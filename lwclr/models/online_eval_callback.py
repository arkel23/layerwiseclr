from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, device
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy

class LogisticRegressionLinearEvaluator(nn.Module):
    def __init__(self, n_input, n_classes):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        # use linear classifier
        self.block_forward = nn.Sequential(Flatten(), nn.Linear(n_input, n_classes, bias=True))

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


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
            dataset='imagenet'
        )
    """

    def __init__(
        self,
        dataset: str,
        z_dim: int = None,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.non_linear_evaluator = LogisticRegressionLinearEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(), lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        inputs, y = batch

        x = inputs[0]
        x = x.to(device)
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
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_logits = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        train_acc = accuracy(mlp_logits.softmax(-1), y)
        pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_logits = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        # log metrics
        val_acc = accuracy(mlp_logits.softmax(-1), y)
        pl_module.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
