# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/ssl_finetuner.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import  pytorch_lightning as pl
from torchmetrics.functional import accuracy

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule
from .heads import LinearHead

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


class LitEvaluator(pl.LightningModule):
    '''
    To evaluate semi-supervised (fine-tuning with limited labels) use
    PL's --limit_train_batches argument 0.1 for 10% or 0.01 for 1% of train samples
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.backbone = load_model(args, ret_interm_repr=False)

        if args.mode == 'linear_eval':
            freeze_layers(self.backbone)               
        
        self.n_features = self.backbone.configuration.hidden_size
        self.num_classes = self.backbone.configuration.num_classes
        
        self.linear_head =  LinearHead(
            n_input=self.n_features,
            n_classes=self.num_classes,
        )

    def forward(self, x):
        return self.linear_head(self.backbone(x)[:, 0])

    def shared_step(self, batch):
        x, y = batch
        logits = self.linear_head(self.backbone(x)[:, 0])
        
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits.softmax(-1), y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)        
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)        
        
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)        
        
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

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
    