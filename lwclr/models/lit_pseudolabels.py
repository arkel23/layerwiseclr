from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import einops
from einops.layers.torch import Rearrange

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule

class CLSHead(nn.Module):
    def __init__(self, args, configuration):
        super().__init__()
        
        if args.interm_features_fc:
            self.inter_class_head = nn.Sequential(
                nn.Linear(configuration.num_hidden_layers, 1),
                Rearrange(' b d 1 -> b d'),
                nn.GELU(),
                nn.LayerNorm(configuration.hidden_size, eps=configuration.layer_norm_eps),
                nn.Linear(configuration.hidden_size, configuration.num_classes)
            )
        else:
            self.class_head = nn.Sequential(
                nn.Linear(configuration.hidden_size, configuration.representation_size),
                nn.GELU(),
                nn.LayerNorm(configuration.hidden_size, eps=configuration.layer_norm_eps),
                nn.Linear(configuration.representation_size, configuration.num_classes)
                )  
    
    def forward(self, x):
        if hasattr(self, 'inter_class_head'):
            return self.inter_class_head(torch.stack(x, dim=-1))    
        else:
            return self.class_head(x)
       

class PLHead(nn.Module):
    def __init__(self, args, configuration):
        super().__init__()
        
        if configuration.load_repr_layer:
            self.repr_layer = nn.Sequential(
                nn.Linear(configuration.hidden_size, configuration.representation_size),
                nn.ReLU
            )
            pre_logits_size = configuration.representation_size
        else:
            pre_logits_size = configuration.hidden_size
        
        self.class_head = nn.Sequential(
            nn.LayerNorm(pre_logits_size, eps=configuration.layer_norm_eps),
            nn.Linear(configuration.representation_size, args.batch_size)
        )
    
    def forward(self, interm_features):
        class_batch = torch.cat((interm_features), dim=0)[:, 0, :]
        
        if hasattr(self, 'repr_layer'):
            class_batch = self.repr_layer(class_batch)
        
        return self.class_head(class_batch)

        
class LitPseudoLabels(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.backbone = load_model(args)
        self.pl_head = PLHead(args, self.backbone.configuration)

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # return last layer cls token features
        interm_features = self.backbone(x)
        return interm_features[-1][:, 0]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        labels = torch.tensor([i for i in range(x.shape[0])], device=self.device)
        new_labels = torch.cat([labels for _ in range(self.backbone.configuration.num_hidden_layers)], dim=0)
        
        interm_features= self.backbone(x)
        logits = self.pl_head(interm_features)
        
        loss = self.criterion(logits, new_labels)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        
        curr_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', curr_lr, on_epoch=False, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        labels = torch.tensor([i for i in range(x.shape[0])], device=self.device)
        new_labels = torch.cat([labels for _ in range(self.backbone.configuration.num_hidden_layers)], dim=0)
        
        interm_features= self.backbone(x)
        logits = self.pl_head(interm_features)
        
        loss = self.criterion(logits, new_labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        x, _ = batch
        labels = torch.tensor([i for i in range(x.shape[0])], device=self.device)
        new_labels = torch.cat([labels for _ in range(self.backbone.configuration.num_hidden_layers)], dim=0)
        
        interm_features= self.backbone(x)
        logits = self.pl_head(interm_features)
        
        loss = self.criterion(logits, new_labels)
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

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
    