# https://github.com/Spijkervet/SimCLR/blob/master/main_pl.py
# https://github.com/Spijkervet/SimCLR/blob/04bcf2baa1fb5631a0a636825aabe469865ad8a9/simclr/simclr.py#L8
# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/simclr/simclr_module.py
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from .heads import ProjectionMLPHead
from .model_selection import load_model
from .custom_losses import NT_XentSimCLR
from .optim_utils import WarmupCosineSchedule, create_optim

class LitSimCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=False)                
        
        self.projector = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=self.backbone.configuration.hidden_size, 
                            hidden_size=args.projector_hidden_size, out_features=args.projector_output_size)

        self.criterion = NT_XentSimCLR(temp=args.temperature)
        
    def forward(self, x_i):
        return self.backbone(x_i)
        
    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        
        h_i = self.backbone(x_i)
        h_j =  self.backbone(x_j)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
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
        optimizer = create_optim(self, self.args)
        
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
                        choices=['Ti_4', 'Ti_8', 'Ti_16', 'Ti_32', 'S_4', 'S_8', 'S_16', 'S_32', 
                                 'B_4', 'B_8', 'B_16', 'B_32', 'L_16', 'L_32', 'B_16_in1k', 
                                 'effnet_b0', 'resnet18', 'resnet50', 'alexnet'], 
                        default='B_16_in1k', help='Which model architecture to use')
        parser.add_argument('--vit_avg_pooling', action='store_true',
                            help='If use this flag then uses average pooling instead of cls token of ViT')
        parser.add_argument('--model_name_teacher', type=str, default=None, 
                            help='By default uses same architecture as main network, but can choose any other')
        
        parser.add_argument('--layer_contrast', type=int, default=-1, 
                        help='Layer features for pairs')
        parser.add_argument('--random_layer_contrast', action='store_true',
                            help='If use this flag then at each step chooses a random layer from gen to contrast against')
        parser.add_argument('--cont_layers_range', type=int, default=2,
                        help='Choose which last N layers to contrast from (def last 2 layers).')
        parser.add_argument('--freeze_teacher', action='store_true',
                            help='If use this flag then freeze teacher network')
        
        parser.add_argument('--bn_proj', action='store_true',
                            help='If use this flag then uses projector MLP with BN instead of LN')
        parser.add_argument('--no_proj_layers', type=int, choices=[1, 2, 3], default=3,
                            help='Number of layers for projection head.')
        parser.add_argument('--projector_hidden_size', type=int, default=2048,
                        help='Number of units in hidden layer of MLP projector')
        parser.add_argument('--projector_output_size', type=int, default=2048,
                        help='Number of units in output layer of MLP projector')

        parser.add_argument('--fs_weight', type=float, default=1, 
                        help='Weight for fully supervised loss')
        parser.add_argument('--pl_weight', type=float, default=1, 
                        help='Wegith for layer-wise pseudolabels loss')
        parser.add_argument('--cont_weight', type=float, default=1, 
                        help='Weight for contrastive loss')

        parser.add_argument('--pretrained_teacher',action='store_true',
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
