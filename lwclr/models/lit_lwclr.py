import torch
import pytorch_lightning as pl

from .heads import ProjectionMLPHead
from .custom_losses import SupConLoss
from .model_selection import load_model
from .scheduler import WarmupCosineSchedule

class LitLWCLR(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert self.args.cont_layers_range >= 2, 'Need to contrast at least 2 layers'

        self.backbone = load_model(args, ret_interm_repr=True)                
        
        self.projector = ProjectionMLPHead(batch_norm=args.bn_proj, no_layers=args.no_proj_layers,
                            in_features=self.backbone.configuration.hidden_size, 
                            hidden_size=args.projector_hidden_size, out_features=args.projector_output_size)
        # to do: add momentum encoder
        self.criterion_student = SupConLoss(
            temperature=args.temperature, base_temperature=args.temperature, contrast_mode='all')
        
    def forward(self, x):
        return self.backbone(x)
        
    def shared_step(self, batch):
        x, _ = batch
        interm_feats = self.backbone(x)
        interm_feats = interm_feats[-self.args.cont_layers_range:]
        
        z_last = self.projector(interm_feats[-1])
        z_others = [self.projector(feats.detach()) for feats in interm_feats[:-1]]
        z = torch.cat([z_last.unsqueeze(1), torch.stack(z_others, dim=1)], dim=1)
        
        loss = self.criterion(z)
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