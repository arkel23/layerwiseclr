from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import einops
from einops.layers.torch import Rearrange

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule

class LWCLR(nn.Module):
    def __init__(self, args, configuration):
        super().__init__()
        
        if self.configuration.load_repr_layer:
            self.repr_layer = nn.Sequential(
                nn.Linear(configuration.hidden_size, configuration.representation_size),
                nn.ReLU)
            )
            pre_logits_size = config.representation_size
        else:
            pre_logits_size = self.config.hidden_size
        
        self.class_head = nn.Sequential(
            nn.LayerNorm(pre_logits_size, eps=configuration.layer_norm_eps),
            nn.Linear(configuration.representation_size, args.batch_size)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.labels = torch.tensor([i for i in range(args.batch_size)])
        self.new_labels = [labels for _ in range(configuration.num_hidden_layers)]
        
    
    def forward(self, interm_features):
        class_batch = torch.cat((interm_features), dim=0)[:, 0, :]
        
        if hasattr(self, 'repr_layer'):
            class_batch = self.repr_layer(class_batch)
        
        logits = self.class_head(class_batch)
        return self.criterion(logits, self.new_labels)

        
class LayerWiseCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.backbone = load_model(args)
        self.LWCLR = LWCLR(args, configuration)
        
    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        
        return embedding

    def training_step(self, batch, batch_idx):
        # forward and backward pass and log
        x, _ = batch
        
        interm_features= self.backbone(x)
        loss = self.LWCLR(interm_features)
        
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        
        curr_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', curr_lr, on_epoch=False, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        
        interm_features= self.backbone(x)
        loss = self.LWCLR(interm_features)
        
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        
    def test_step(self, batch, batch_idx):
        x, _ = batch
        
        interm_features= self.backbone(x)
        loss = self.LWCLR(interm_features)
        
        self.log('test_loss', loss, on_epoch=True, on_step=False)

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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
        parser.add_argument('--learning_rate', default=0.001, type=float,
                            help='Initial learning rate.')  
        parser.add_argument('--weight_decay', type=float, default=0.00)
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR scheduler.')
        
        parser.add_argument('--model_name', choices=['B_16', 'B_32', 'L_16', 'L_32'], default='B_16',
                        help='Which model architecture to use')
        parser.add_argument('--pretrained_checkpoint',action='store_true',
                            help='Loads pretrained model if available')
        parser.add_argument('--checkpoint_path', type=str, default=None)     
        parser.add_argument('--transfer_learning', action='store_true',
                            help='Load partial state dict for transfer learning'
                            'Resets the [embeddings, logits and] fc layer for ViT')    
        parser.add_argument('--load_partial_mode', choices=['full_tokenizer', 'patchprojection', 
                            'posembeddings', 'clstoken', 'patchandposembeddings', 
                            'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                            help='Load pre-processing components to speed up training')
        
        return parser

