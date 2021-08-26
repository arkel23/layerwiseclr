# https://github.com/Spijkervet/SimCLR/blob/master/main_pl.py
from argparse import ArgumentParser

import torch
import torch.nn as nn
import  pytorch_lightning as pl

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule
from .simclr import SimCLR
from .custom_losses import NT_XentSimCLR

class LitSimCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=False)                
        
        self.n_features = self.backbone.configuration.hidden_size
        self.representation_size = self.backbone.configuration.representation_size
        
        self.model = SimCLR(self.backbone, 
            projection_dim=self.backbone.configuration.representation_size,
            n_features=self.n_features)
        self.criterion = NT_XentSimCLR(batch_size=args.batch_size, 
            temperature=args.temperature, world_size=1)

    def forward(self, x_i):
        return self.model.inference(x_i)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch

        h_i, h_j, z_i, z_j = self.model(x_i, x_j)

        loss = self.criterion(z_i, z_j)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        
        curr_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', curr_lr, on_epoch=False, on_step=True)

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
    