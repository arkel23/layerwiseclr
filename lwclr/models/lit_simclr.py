# https://github.com/Spijkervet/SimCLR/blob/master/main_pl.py
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .simclr import SimCLR
from .model_selection import load_model
from .custom_losses import NT_XentSimCLR
from .scheduler import WarmupCosineSchedule

class LitSimCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=False)                
        
        self.n_features = self.backbone.configuration.hidden_size
        self.representation_size = self.backbone.configuration.representation_size
        
        self.model = SimCLR(self.backbone, 
            projection_dim=self.backbone.configuration.representation_size,
            n_features=self.n_features, ret_interm_repr=False)

        self.criterion = NT_XentSimCLR(temp=args.temperature)
        
    def forward(self, x_i):
        return self.model.inference(x_i)

    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss

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
    