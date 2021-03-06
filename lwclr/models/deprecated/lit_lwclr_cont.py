import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .lit_simclr import SimCLR
from .heads import ProjectionHead, ProjectionMLP
from .model_selection import load_model
from .custom_losses import NT_XentSimCLR, SupConLoss
from .scheduler import WarmupCosineSchedule
from .lit_evaluator import freeze_layers

class LitLWCLRCont(pl.LightningModule):
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
        
        if self.args.mode == 'lwclr_cont_single':
            self.contrastive_head = SimContrastiveHead(in_features=in_features,
                out_features=out_features, hidden_size=hidden_size, 
                no_layers=args.no_proj_layers, temp=args.temperature, bn_proj=self.args.bn_proj)
        else:
            self.contrastive_head = LWContrastiveHead(in_features=in_features,
                out_features=out_features, hidden_size=hidden_size, 
                no_layers=args.no_proj_layers, temp=args.temperature, bn_proj=self.args.bn_proj)
                    
        self.aux = SimCLR(self.backbone_aux, 
            no_layers=args.no_proj_layers, in_features=in_features, 
            out_features=out_features, hidden_size=hidden_size,
            ret_interm_repr=True, bn_proj=self.args.bn_proj, layers_contrast=[-1, -1])
        
        self.criterion_aux = NT_XentSimCLR(temp=args.temperature)
            
    def forward(self, x_i):
        return self.backbone(x_i)

    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        h_i, h_j, z_i, z_j = self.aux(x_i, x_j)
        
        # loss for auxiliary generator/giver network
        loss_aux = self.criterion_aux(z_i, z_j)
        
        # loss for lwclr network
        if self.args.mode == 'lwclr_cont_single':
            if self.args.random_layer_contrast:
                last_layer = self.backbone.configuration.num_hidden_layers - 1
                
                features_i = self.backbone(x_i)
                features_aux_j = h_j[random.randint(
                    last_layer - self.args.cont_layers_range + 1, last_layer)].detach()
                
                features_j = self.backbone(x_j)
                features_aux_i = h_i[random.randint(
                    last_layer - self.args.cont_layers_range + 1, last_layer)].detach()
            
            else:
                features_i = self.backbone(x_i)
                features_aux_j = h_j[self.args.layer_contrast].detach()
                
                features_j = self.backbone(x_j)
                features_aux_i = h_i[self.args.layer_contrast].detach()
            
            loss_ij = self.contrastive_head(features_i, features_aux_j)
            loss_ji = self.contrastive_head(features_j, features_aux_i)
            loss = loss_ij + loss_ji
        else:
            raise NotImplementedError    
        
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
    
