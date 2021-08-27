import torch
import torch.nn as nn
import pytorch_lightning as pl

from .simclr import SimCLR
from .heads import ProjectionHead
from .model_selection import load_model
from .custom_losses import NT_XentSimCLR
from .scheduler import WarmupCosineSchedule

NO_AUGS = 2
PROJECTION_LAYERS = 2
PL_WEIGHT = 0.1    

class PLClassificationHead(nn.Module):
    def __init__(self, n_layers, n_features, n_classes, hidden_size):
        super().__init__()
        
        self.projection_class_head = ProjectionHead( 
                n_layers=n_layers, n_features=n_features, n_classes=n_classes, 
                hidden_size=hidden_size)
        
    def forward(self, interm_features_i, interm_features_j):
        cat_features_i = torch.cat((interm_features_i), dim=0)[:, 0, :]
        cat_features_j = torch.cat((interm_features_j), dim=0)[:, 0, :]
        cat_features = torch.cat((cat_features_i, cat_features_j), dim=0)

        return self.projection_class_head(cat_features)

        
class LitLWPLCLR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.backbone = load_model(args, ret_interm_repr=True)                
        
        self.n_features = self.backbone.configuration.hidden_size
        self.representation_size = self.backbone.configuration.representation_size
        
        self.model = SimCLR(self.backbone, 
            projection_dim=self.backbone.configuration.representation_size,
            n_features=self.n_features, ret_interm_repr=True)

        self.criterion_contrastive = NT_XentSimCLR(temp=args.temperature)
        
        self.pl_class_head = PLClassificationHead(n_layers=PROJECTION_LAYERS, 
            n_features=self.n_features, n_classes=args.batch_size,
            hidden_size=self.representation_size)

        self.criterion_pseudosupervised = nn.CrossEntropyLoss()

    def forward(self, x_i):
        return self.model.inference(x_i)

    def get_pseudolabels(self, batch_size, no_augs=NO_AUGS):
        pseudolabels = torch.tensor([i for i in range(batch_size)], device=self.device)
        pseudolabels = torch.cat([pseudolabels for _ in range(
            self.backbone.configuration.num_hidden_layers)], dim=0)
        pseudolabels = torch.cat([pseudolabels for _ in range(no_augs)])
        return pseudolabels

    def shared_step(self, batch):
        (x_i, x_j), _ = batch
        pseudolabels = self.get_pseudolabels(batch_size=x_i.shape[0])

        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss_contrastive = self.criterion_contrastive(z_i, z_j)

        logits = self.pl_class_head(h_i, h_j)
        loss_pseudosupervised = self.criterion_pseudosupervised(logits, pseudolabels)
        
        loss = loss_contrastive + PL_WEIGHT * loss_pseudosupervised
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
    