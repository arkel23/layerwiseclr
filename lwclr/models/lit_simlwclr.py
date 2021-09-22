import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .heads import ProjectionMLPHead, MultiScaleToSingleScaleHead
from .custom_losses import SupConLoss
from .model_selection import load_model
from .optim_utils import WarmupCosineSchedule, create_optim

class LitSimLWCLR(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        # if contrast only last layer then becomes simclr
        #assert self.args.cont_layers_range >= 2, 'Need to contrast at least 2 layers'

        self.backbone = load_model(args, ret_interm_repr=True)                
        
        stop_gradient = not self.args.no_stop_gradient
        self.rescaler = MultiScaleToSingleScaleHead(args, model=self.backbone, detach=stop_gradient)
        
        self.projector = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=self.backbone.configuration.hidden_size, 
                            hidden_size=args.projector_hidden_size, out_features=args.projector_output_size)
        
        self.criterion = SupConLoss(
            temperature=args.temperature, base_temperature=args.temperature, contrast_mode='all')
        
    def forward(self, x_i):
        return self.backbone(x_i)[-1]
        
    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        
        interm_feats_i = self.backbone(x_i)
        z_i = self.projector(interm_feats_i[-1])
        
        interm_feats_j =  self.backbone(x_j)
        interm_feats_j = self.rescaler(interm_feats_j)
        interm_feats_j = interm_feats_j[-self.args.cont_layers_range:]
        #z_j = [self.projector(feats.detach()) for feats in interm_feats_j]
        z_j = [self.projector(feats) for feats in interm_feats_j]
        
        z = torch.cat([z_i.unsqueeze(1), torch.stack(z_j, dim=1)], dim=1)
        loss = self.criterion(F.normalize(z, dim=2))
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