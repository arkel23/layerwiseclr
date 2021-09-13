import random
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .model_selection import load_model
from .lit_lwclr_cont import SimContrastiveHead, LWContrastiveHead
from .scheduler import WarmupCosineSchedule
from .lit_evaluator import freeze_layers
    
class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, classes: int):
        super(ClassificationHead, self).__init__()
        
        self.class_head = nn.Sequential(
            nn.Linear(in_features, classes)
        )
        
    def forward(self, x):
        return self.class_head(x)
        
        
class LitLWCLRFull(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # one model for giving/generating layer-wise views
        # another for receiving and evaluating them
        self.backbone_aux = load_model(args, ret_interm_repr=True, pretrained=args.pretrained_aux)
        self.backbone = load_model(args, ret_interm_repr=False, pretrained=False)  
        
        if self.args.freeze_aux:
            freeze_layers(self.backbone_aux)             
        
        in_features = self.backbone.configuration.hidden_size
        hidden_size = self.args.projector_hidden_size
        out_features = self.args.projector_output_size
        
        if self.args.mode == 'lwclr_full_single':
            self.contrastive_head = SimContrastiveHead(in_features=in_features,
                out_features=out_features, hidden_size=hidden_size, 
                no_layers=args.no_proj_layers, temp=args.temperature)
        else:
            self.contrastive_head = LWContrastiveHead(in_features=in_features,
                out_features=out_features, hidden_size=hidden_size, 
                no_layers=args.no_proj_layers, temp=args.temperature)
        
        self.aux = ClassificationHead(in_features=in_features,
                classes=args.num_classes)
        
        self.criterion_aux = nn.CrossEntropyLoss()
           
    def forward(self, x):
        return self.backbone(x)

    def shared_step(self, batch):
        x, y = batch
        interm_features = self.backbone_aux(x)
        features = self.backbone(x)
        
        # loss for auxiliary generator/giver network
        logits = self.aux(interm_features[-1])
        loss_aux = self.criterion_aux(logits, y)
        
        # loss for lwclr network
        if self.args.mode == 'lwclr_full_single':
            if self.args.random_layer_contrast:
                last_layer = self.backbone.configuration.num_hidden_layers - 1
                features_aux = interm_features[
                    random.randint(last_layer - self.args.cont_layers_range + 1, last_layer)].detach()
            else:
                features_aux = interm_features[self.args.layer_contrast].detach()
            loss = self.contrastive_head(features, features_aux)
        else:
            features_aux = interm_features[-self.args.cont_layers_range:]
            loss = self.contrastive_head(features, features_aux)    
        
        return loss, loss_aux             

    def training_step(self, batch, batch_idx):
        loss, loss_aux = self.shared_step(batch)    
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_loss_aux', loss_aux, on_epoch=True, on_step=False)
        
        return loss + loss_aux

    def validation_step(self, batch, batch_idx):
        loss, loss_aux = self.shared_step(batch) 
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_aux', loss_aux, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss + loss_aux

    def test_step(self, batch, batch_idx):
        loss, loss_aux = self.shared_step(batch) 
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_loss_aux', loss_aux, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss + loss_aux

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)  
        else: 
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, 
            momentum=0.9, weight_decay=self.args.weight_decay)

        scheduler = {'scheduler': WarmupCosineSchedule(
        optimizer, warmup_steps=self.args.warmup_steps, 
        t_total=self.args.total_steps),
        'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        
        return [optimizer], [scheduler]
    
