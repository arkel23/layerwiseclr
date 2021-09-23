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
#from .optim_utils import WarmupCosineSchedule, create_optim
from .optim_utils import create_optim, create_scheduler
        
class LitContDistillFull(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # one model for giving/generating layer-wise views
        # another for receiving and evaluating them
        # teacher can be different (controlled by --model_name_teacher arg)
        self.distill = True if args.model_name_teacher else False
        self.backbone = load_model(args, ret_interm_repr=False, pretrained=False)  
        self.backbone_teacher = load_model(
            args, ret_interm_repr=True, pretrained=args.pretrained_teacher, 
            distill=self.distill, ckpt_path=args.checkpoint_path)
        
        if self.args.freeze_teacher:
            freeze_layers(self.backbone_teacher)             

        in_features = self.backbone.configuration.hidden_size
        in_features_teacher = self.backbone_teacher.configuration.hidden_size
        hidden_size = self.args.projector_hidden_size
        out_features = self.args.projector_output_size
        
        if ((self.args.mode == 'cd_full_single' and (self.args.random_layer_contrast or self.args.layer_contrast != -1)) or 
        (self.args.mode == 'cd_full_multi' and self.args.cont_layers_range != 1)):
            self.rescaler = MultiScaleToSingleScaleHead(args, model=self.backbone_teacher, distill=self.distill)
        
        self.projector = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=in_features, hidden_size=hidden_size, out_features=out_features)
        self.projector_teacher = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=in_features_teacher, hidden_size=hidden_size, out_features=out_features)
        
        self.cls_head = ProjectionMLPHead(linear=True, no_layers=1, 
                in_features=in_features, out_features=args.num_classes)
        self.cls_head_teacher = ProjectionMLPHead(linear=True, no_layers=1, 
                in_features=in_features_teacher, out_features=args.num_classes)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_student = SupConLoss(
            temperature=args.temperature, base_temperature=args.temperature, contrast_mode='all')
           
    def forward(self, x):
        return self.backbone(x)

    def cd_full_single(self, interm_feats_teacher, feats_student):
        if self.args.random_layer_contrast:
            last_layer = self.backbone.configuration.num_hidden_layers - 1
            feats_teacher = interm_feats_teacher[
                random.randint(last_layer - self.args.cont_layers_range + 1, last_layer)]#.detach()
        else:
            feats_teacher = interm_feats_teacher[self.args.layer_contrast]#.detach()
        
        z_teacher = self.projector_teacher(feats_teacher)
        z_student = self.projector(feats_student)
        
        z = torch.cat([z_student.unsqueeze(1), z_teacher.unsqueeze(1)], dim=1)
        loss_cont = self.criterion_student(F.normalize(z, dim=2))
        return loss_cont

    def cd_full_multi(self, interm_feats_teacher, feats_student):
        interm_feats_teacher = interm_feats_teacher[-self.args.cont_layers_range:]
        
        #z_teacher = [self.projector_teacher(feats.detach()) for feats in interm_feats_teacher]
        z_teacher = [self.projector_teacher(feats) for feats in interm_feats_teacher]
        z_student = self.projector(feats_student)
        
        z = torch.cat([z_student.unsqueeze(1), torch.stack(z_teacher, dim=1)], dim=1)        
        loss_cont = self.criterion_student(F.normalize(z, dim=2))
        return loss_cont
        
    def shared_step(self, batch):
        x, y = batch
        
        interm_feats_teacher = self.backbone_teacher(x)
        feats_student = self.backbone(x)
        
        # loss for teacher network
        logits_teacher = self.cls_head_teacher(interm_feats_teacher[-1])
        loss_teacher = self.criterion_cls(logits_teacher, y)
        teacher_acc = accuracy(logits_teacher.softmax(-1), y)

        # loss for student network
        if hasattr(self, 'rescaler'):
            interm_feats_teacher = self.rescaler(interm_feats_teacher)
        else:
            interm_feats_teacher = [interm_feats_teacher[-1].detach()]
            
        if self.args.mode == 'cd_full_single':
            loss_cont = self.cd_full_single(interm_feats_teacher, feats_student)    
        else:
            loss_cont = self.cd_full_multi(interm_feats_teacher, feats_student)
        
        logits = self.cls_head(feats_student)
        loss_fs = self.criterion_cls(logits, y)
        acc = accuracy(logits.softmax(-1), y)

        loss = (self.args.fs_weight * loss_fs) + (self.args.cont_weight * loss_cont)

        return loss, loss_teacher, acc, teacher_acc 

    def training_step(self, batch, batch_idx):
        loss, loss_teacher, acc, teacher_acc = self.shared_step(batch)    
        
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        metrics = {'train_acc': acc, 'train_acc_teacher': teacher_acc, 'train_loss_teacher': loss_teacher}
        self.log_dict(metrics, on_epoch=True, on_step=False)

        return loss + loss_teacher

    def validation_step(self, batch, batch_idx):
        loss, loss_teacher, acc, teacher_acc = self.shared_step(batch)    
        
        metrics = {'val_acc': acc, 'val_acc_teacher': teacher_acc, 
                   'val_loss': loss, 'val_loss_teacher': loss_teacher}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

        return loss + loss_teacher

    def test_step(self, batch, batch_idx):
        loss, loss_teacher, acc, teacher_acc = self.shared_step(batch)    
        
        metrics = {'test_acc': acc, 'test_acc_teacher': teacher_acc, 
                   'test_loss': loss, 'test_loss_teacher': loss_teacher}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

        return loss + loss_teacher

    def configure_optimizers(self):
        optimizer = create_optim(self, self.args)
        
        scheduler = create_scheduler(self.args, optimizer, self.args.total_steps)
        #scheduler = {'scheduler': WarmupCosineSchedule(
        #optimizer, warmup_steps=self.args.warmup_steps, 
        #t_total=self.args.total_steps),
        #'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        
        return [optimizer], [scheduler]
    
