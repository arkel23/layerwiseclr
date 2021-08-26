from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule
from .custom_losses import NT_XentSimCLR

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

        self.backbone = load_model(args)                
        
        self.n_features = self.backbone.configuration.hidden_size
        self.representation_size = self.backbone.configuration.representation_size
        
        self.layers_contrast = [args.layer_pair_1, args.layer_pair_2]

        self.model = SimLWCLR(self.backbone, 
            projection_dim=self.backbone.configuration.representation_size,
            n_features=self.n_features, layers_contrast=self.layers_contrast)

        self.criterion = NT_XentSimCLR(temperature=args.temperature)
        
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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature parameter for ntxent loss')        

        parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
        parser.add_argument('--learning_rate', default=3e-4, type=float,
                        help='Initial learning rate.')  
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR scheduler.')
        parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='If doing warmup in terms of epochs instead of steps.')

        parser.add_argument('--model_name', choices=['B_16', 'B_32', 'L_16', 'L_32'], default='B_16',
                        help='Which model architecture to use')
        parser.add_argument('--projection_layers', type=int, choices=[1, 2, 3], default=1,
                            help='Number of layers for projection head.')
        parser.add_argument('--layer_pair_1', type=int, default=0, 
                        help='Layer features for pairs')
        parser.add_argument('--layer_pair_2', type=int, default=-1, 
                        help='Layer features for pairs')

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

        parser.add_argument('--interm_features_fc', action='store_true', 
                        help='If use this flag creates FC using intermediate features instead of only last layer.')
        parser.add_argument('--conv_patching', action='store_true', 
                        help='If use this flag uses a small convolutional stem instead of single large-stride convolution for patch projection.')
        
        return parser

