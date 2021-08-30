# https://github.com/Spijkervet/SimCLR/blob/master/main_pl.py
# https://github.com/Spijkervet/SimCLR/blob/04bcf2baa1fb5631a0a636825aabe469865ad8a9/simclr/simclr.py#L8
# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/simclr/simclr_module.py
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .heads import ProjectionHead
from .model_selection import load_model
from .custom_losses import NT_XentSimCLR
from .scheduler import WarmupCosineSchedule

class SimCLR(nn.Module):
    def __init__(self, encoder, 
                 in_features: int, out_features: int, 
                 hidden_size: int, no_layers: int = 2, 
                 ret_interm_repr: bool = False, layers_contrast=[-1, -1]):
        super(SimCLR, self).__init__()

        self.encoder = encoder

        self.projector = ProjectionHead(no_layers=no_layers, in_features=in_features, 
                            out_features=out_features, hidden_size=hidden_size)

        self.ret_interm_repr = ret_interm_repr
        self.layers_contrast = layers_contrast

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # use cls token for projection
        if not self.ret_interm_repr:
            z_i = self.projector(h_i)
            z_j = self.projector(h_j)
        else:
            # if using layerwise repr. choose which layer repr. to use
            z_i = self.projector(h_i[self.layers_contrast[0]])
            z_j = self.projector(h_j[self.layers_contrast[1]])
        return h_i, h_j, z_i, z_j


class LitSimCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=False)                
        
        in_features = self.backbone.configuration.hidden_size
        repr_size = self.backbone.configuration.representation_size
        
        self.model = SimCLR(self.backbone, 
            no_layers=args.no_proj_layers, in_features=in_features, 
            out_features=repr_size, hidden_size=repr_size)

        self.criterion = NT_XentSimCLR(temp=args.temperature)
        
    def forward(self, x_i):
        if not self.args.ret_interm_repr:
            return self.backbone(x_i)
        return self.backbone(x_i)[-1]
        
    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        _, _, z_i, z_j = self.model(x_i, x_j)
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

        parser.add_argument('--model_name', 
                        choices=['B_16', 'B_32', 'L_16', 'L_32', 
                                 'effnet_b0', 'resnet18', 'resnet50'], 
                        default='B_16', help='Which model architecture to use')
        parser.add_argument('--vit_avg_pooling', action='store_true',
                            help='If use this flag then uses average pooling instead of cls token of ViT')
        
        parser.add_argument('--no_proj_layers', type=int, choices=[1, 2, 3], default=2,
                            help='Number of layers for projection head.')
        parser.add_argument('--layer_contrast', type=int, default=-1, 
                        help='Layer features for pairs')
        parser.add_argument('--random_layer_contrast', action='store_true',
                            help='If use this flag then at each step chooses a random layer from gen to contrast against')
        parser.add_argument('--cont_layers_range', type=int, default=3,
                        help='Choose which last N layers to contrast from.')
        
        parser.add_argument('--fs_weight', type=float, default=1, 
                        help='Weight for fully supervised loss')
        parser.add_argument('--pl_weight', type=float, default=1, 
                        help='Wegith for layer-wise pseudolabels loss')
        parser.add_argument('--cont_weight', type=float, default=1, 
                        help='Weight for contrastive loss')

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
