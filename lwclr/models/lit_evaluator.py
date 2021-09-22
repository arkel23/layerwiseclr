# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/ssl_finetuner.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import  pytorch_lightning as pl
from torchmetrics.functional import accuracy

from .heads import ProjectionMLPHead
from .model_selection import load_model
from .optim_utils import WarmupCosineSchedule, create_optim

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

        self.backbone = load_model(args, ret_interm_repr=False, ckpt_path=args.checkpoint_path)

        if args.mode == 'linear_eval':
            freeze_layers(self.backbone)               
        
        in_features = self.backbone.configuration.hidden_size
        num_classes = self.backbone.configuration.num_classes
        
        self.cls_head =  ProjectionMLPHead(
            linear=True, no_layers=1, in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.cls_head(self.backbone(x))

    def shared_step(self, batch):
        x, y = batch
        logits = self.cls_head(self.backbone(x))
        
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
        optimizer = create_optim(self, self.args)
        
        scheduler = {'scheduler': WarmupCosineSchedule(
        optimizer, warmup_steps=self.args.warmup_steps, 
        t_total=self.args.total_steps),
        'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        
        return [optimizer], [scheduler]
    