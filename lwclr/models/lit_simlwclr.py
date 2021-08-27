from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .model_selection import load_model
from .custom_losses import NT_XentSimCLR
from .scheduler import WarmupCosineSchedule

class SimLWCLR(nn.Module):
    def __init__(self, encoder, projection_dim, n_features, layers_contrast=[0, -1]):
        super(SimLWCLR, self).__init__()

        self.encoder = encoder

        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.GELU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )
        
        self.layers_contrast = layers_contrast
    
    def inference(self, x):
        return self.encoder(x)[-1][:, 0]
    
    def forward(self, x):
        interm_features = self.encoder(x)
        h_i = interm_features[self.layers_contrast[0]]
        h_j = interm_features[self.layers_contrast[1]]

        z_i = self.projector(h_i[:, 0])
        z_j = self.projector(h_j[:, 0])
        return h_i, h_j, z_i, z_j


class LitSimLWCLR(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=True)                
        
        self.n_features = self.backbone.configuration.hidden_size
        self.representation_size = self.backbone.configuration.representation_size
        
        self.layers_contrast = [args.layer_pair_1, args.layer_pair_2]

        self.model = SimLWCLR(self.backbone, 
            projection_dim=self.backbone.configuration.representation_size,
            n_features=self.n_features, layers_contrast=self.layers_contrast)

        self.criterion = NT_XentSimCLR(temp=args.temperature)
        
    def forward(self, x):
        # return last layer cls token features
        return self.model.inference(x)
        
    def shared_step(self, batch):
        x, _ = batch
        h_i, h_j, z_i, z_j = self.model(x)
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