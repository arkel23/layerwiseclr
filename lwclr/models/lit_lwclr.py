import os
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

from .model_selection import load_model
from lwclr.utilities.scheduler import WarmupCosineSchedule

class LitClassifier(pl.LightningModule):
    def __init__(self, args, total_steps, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.backbone = VisionTransformer(args)
        print(str(self.backbone.configuration))

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        curr_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', curr_lr, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        metrics = {'val_acc': self.val_acc, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)  
        else: 
            optimizer = torch.optim.SGD(self.parameters(), 
            lr=self.hparams.learning_rate, momentum=0.9, 
            weight_decay=self.hparams.weight_decay)
        scheduler = {'scheduler': WarmupCosineSchedule(
        optimizer, warmup_steps=self.hparams.warmup_steps, 
        t_total=self.hparams.total_steps),
        'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
    '''    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=str, default=1e-3)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--weight_decay', type=float, default=0.00)
        parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
        parser.add_argument('--model_name', choices=['B_16', 'B_32', 'L_16', 'L_32'], default='B_16')
        parser.add_argument("--checkpoint_path", type=str, default=None)  
        parser.add_argument("--pretrained", type=bool, default=True,
                            help="For models with pretrained weights available"
                            "Default=False")
        parser.add_argument("--load_partial_mode", choices=['full_tokenizer', 'patchprojection', 
                            'posembeddings', 'clstoken', 'patchandposembeddings', 
                            'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                            help="Load pre-processing components to speed up training")
        return parser
    '''

