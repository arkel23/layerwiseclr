# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/callbacks/ssl_online.py
# https://github.com/PyTorchLightning/lightning-bolts/blob/47eb2aae677350159c9ec0dc8ccdb6eef4217fff/pl_bolts/models/self_supervised/evaluator.py
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, device
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy

from .heads import ProjectionMLPHead

class SSLOnlineLinearEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
        )
    """

    def __init__(
        self,
        mode: str,
        z_dim: int = None,
        num_classes: int = None,
        lr: float = 3e-4,
    ):
        """
        Args:
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.optimizer: Optimizer

        self.mode = mode
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.lr = lr

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.linear_evaluator = ProjectionMLPHead(
            linear=True, no_layers=1, in_features=self.z_dim, out_features=self.num_classes,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.linear_evaluator.parameters(), lr=self.lr)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, test: bool, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        inputs, y = batch

        if self.mode in ['simclr', 'simlwclr'] and (not test):
            x = inputs[0]
            x = x.to(device)
            y = y.to(device)
        else:
            x = inputs.to(device)
            y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(test=False, batch=batch, device=pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        logits = pl_module.linear_evaluator(representations)  # type: ignore[operator]
        loss = F.cross_entropy(logits, y)

        # update finetune weights
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        train_acc = accuracy(logits.softmax(-1), y)
        pl_module.log("online_train_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(test=True, batch=batch, device=pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        with torch.no_grad():
            logits = pl_module.linear_evaluator(representations)  # type: ignore[operator]
            loss = F.cross_entropy(logits, y)

        # log metrics
        val_acc = accuracy(logits.softmax(-1), y)
        pl_module.log("online_val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(test=True, batch=batch, device=pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        with torch.no_grad():
            logits = pl_module.linear_evaluator(representations)  # type: ignore[operator]
            loss = F.cross_entropy(logits, y)

        # log metrics
        test_acc = accuracy(logits.softmax(-1), y)
        pl_module.log("online_test_acc", test_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
