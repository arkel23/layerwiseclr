import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from .heads import ProjectionMLPHead, MultiScaleToSingleScaleHead
from .custom_losses import SupConLoss
from .model_selection import load_model
from .lit_evaluator import freeze_layers
from .optim_utils import WarmupCosineSchedule, create_optim
        
class LitContDistill(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # one model for giving/generating layer-wise views
        # another for receiving and evaluating them
        # teacher can be different (controlled by --model_name_teacher arg)
        self.distill = True if args.model_name_teacher else False
        self.backbone = load_model(args, ret_interm_repr=False, pretrained=False)  
        self.backbone_teacher = load_model(args, ret_interm_repr=True, pretrained=args.pretrained_teacher, distill=self.distill)
        
        if self.args.freeze_teacher:
            freeze_layers(self.backbone_teacher)             

        in_features = self.backbone.configuration.hidden_size
        in_features_teacher = self.backbone_teacher.configuration.hidden_size
        hidden_size = self.args.projector_hidden_size
        out_features = self.args.projector_output_size
        
        self.rescaler = MultiScaleToSingleScaleHead(args, model=self.backbone_teacher, distill=self.distill)
        
        self.projector = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=in_features, hidden_size=hidden_size, out_features=out_features)
        self.projector_teacher = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=in_features_teacher, hidden_size=hidden_size, out_features=out_features)
        
        self.cls_head = ProjectionMLPHead(linear=True, no_layers=1, 
                in_features=in_features_teacher, out_features=args.num_classes)

        self.criterion_teacher = nn.CrossEntropyLoss()
        self.criterion_student = SupConLoss(
            temperature=args.temperature, base_temperature=args.temperature, contrast_mode='all')
           
    def forward(self, x):
        return self.backbone(x)

    def cd_single(self, interm_feats_teacher, feats_student):
        if self.args.random_layer_contrast:
            last_layer = self.backbone.configuration.num_hidden_layers - 1
            feats_teacher = interm_feats_teacher[
                random.randint(last_layer - self.args.cont_layers_range + 1, last_layer)]#.detach()
        else:
            feats_teacher = interm_feats_teacher[self.args.layer_contrast]#.detach()
        
        z_teacher = self.projector_teacher(feats_teacher)
        z_student = self.projector(feats_student)
        
        z = torch.cat([z_student.unsqueeze(1), z_teacher.unsqueeze(1)], dim=1)
        loss = self.criterion_student(F.normalize(z, dim=2))
        return loss

    def cd_multi(self, interm_feats_teacher, feats_student):
        interm_feats_teacher = interm_feats_teacher[-self.args.cont_layers_range:]
        
        #z_teacher = [self.projector_teacher(feats.detach()) for feats in interm_feats_teacher]
        z_teacher = [self.projector_teacher(feats) for feats in interm_feats_teacher]
        z_student = self.projector(feats_student)
        
        z = torch.cat([z_student.unsqueeze(1), torch.stack(z_teacher, dim=1)], dim=1)        
        loss = self.criterion_student(F.normalize(z, dim=2))
        return loss
        
    def shared_step(self, batch):
        x, y = batch
        
        interm_feats_teacher = self.backbone_teacher(x)
        feats_student = self.backbone(x)
        
        # loss for teacher network
        logits = self.cls_head(interm_feats_teacher[-1])
        loss_teacher = self.criterion_teacher(logits, y)
        teacher_acc = accuracy(logits.softmax(-1), y)

        # loss for student network
        interm_feats_teacher = self.rescaler(interm_feats_teacher)
        if self.args.mode == 'cd_single':
            loss = self.cd_single(interm_feats_teacher, feats_student)    
        else:
            loss = self.cd_multi(interm_feats_teacher, feats_student)
        
        return loss, loss_teacher, teacher_acc 

    def training_step(self, batch, batch_idx):
        loss, loss_teacher, teacher_acc = self.shared_step(batch)    
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_loss_teacher', loss_teacher, on_epoch=True, on_step=False)
        self.log('train_acc_teacher', teacher_acc, on_epoch=True, on_step=False)

        return loss + loss_teacher

    def validation_step(self, batch, batch_idx):
        loss, loss_teacher, teacher_acc = self.shared_step(batch) 
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_teacher', loss_teacher, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_acc_teacher', teacher_acc, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss + loss_teacher

    def test_step(self, batch, batch_idx):
        loss, loss_teacher, teacher_acc = self.shared_step(batch) 
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_loss_teacher', loss_teacher, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc_teacher', teacher_acc, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss + loss_teacher

    def configure_optimizers(self):
        optimizer = create_optim(self, self.args)
        
        scheduler = {'scheduler': WarmupCosineSchedule(
        optimizer, warmup_steps=self.args.warmup_steps, 
        t_total=self.args.total_steps),
        'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        
        return [optimizer], [scheduler]
    
