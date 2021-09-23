# https://github.com/PyTorchLightning/lightning-bolts/blob/0.3.0/pl_bolts/callbacks/knn_online.py#L17-L121
# https://github.com/PatrickHua/SimSiam/blob/01d7e7811ac7b864bf8adccc8005208878208994/tools/knn_monitor.py
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier

class KNNOnlineEvaluator(Callback):  # pragma: no cover
    """
    Evaluates self-supervised K nearest neighbors.
    
    Example::
    
        # your model must have 1 attribute
        model = Model()
        model.num_classes = ... # the num of classes in the model
    
        online_eval = KNNOnlineEvaluator(
            num_classes=model.num_classes,
        )
    
    """

    def __init__(
        self,
        mode: str,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Args:
            num_classes: Number of classes
        """
        
        super().__init__()
        
        self.mode = mode
        self.num_classes = num_classes
        
        self.knn_k = 200 # min(args.train.knn_k, len(memory_loader.dataset))
        self.knn_t = 0.1 
        
    def get_representations(self, pl_module: LightningModule, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def get_all_representations(
        self, split,
        pl_module: LightningModule,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_representations = None
        ys = None
        test = True if split in ['val', 'test'] else False
        
        for batch in dataloader:
            x, y = self.to_device(test=test, batch=batch, device=pl_module.device)

            with torch.no_grad():
                representations = F.normalize(self.get_representations(pl_module, x), dim=1)

            if all_representations is None:
                all_representations = representations.detach()
            else:
                all_representations = torch.cat([all_representations, representations.detach()])

            if ys is None:
                ys = y
            else:
                ys = torch.cat([ys, y])

        return all_representations.t().contiguous(), ys
        #return all_representations.cpu().numpy(), ys.cpu().numpy()  # type: ignore[union-attr]
    
    def knn_monitor(self, split, pl_module, feature_bank, feature_labels, dataloader):
        pl_module.eval()
    
        total_top1, total_num = 0.0, 0
        test = True if split in ['val', 'test'] else False
        
        with torch.no_grad():
            # loop test data to predict the label by weighted knn search
            #test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
            #for data, target in test_bar:
            #    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            for batch in dataloader:
                x, y = self.to_device(test=test, batch=batch, device=pl_module.device)
                
                feature = F.normalize(pl_module(x), dim=1)
                
                #feature = net(data)
                #feature = F.normalize(feature, dim=1)
                
                pred_labels = self.knn_predict(feature, feature_bank, feature_labels)

                total_num += x.size(0)
                total_top1 += (pred_labels[:, 0] == y).float().sum().item()
                #test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
        return total_top1 / total_num
    
    def knn_predict(self, feature, feature_bank, feature_labels):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * self.knn_k, self.num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def to_device(self, test: bool, batch: torch.Tensor, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:        
        #print(len(batch), batch)
        #print(len(batch))
        inputs, y = batch

        if self.mode in ['simclr', 'simlwclr'] and (not test):
            x = inputs[0]
            x = x.to(device)
            y = y.to(device)
        else:
            x = inputs.to(device)
            y = y.to(device)

        return x, y

    '''
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.knn_evaluator = KNeighborsClassifier(n_neighbors=self.num_classes, n_jobs=-1)

        val_dataloader = pl_module.val_dataloader()
        representations, y = self.get_all_representations(pl_module, val_dataloader)  # type: ignore[arg-type]

        # knn fit
        pl_module.knn_evaluator.fit(representations, y)  # type: ignore[union-attr,operator]

        # knn val acc
        val_acc = pl_module.knn_evaluator.score(representations, y)  # type: ignore[union-attr,operator]

        # log metrics
        pl_module.log('online_knn_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
    '''
    
    #'''
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        #train_dataloader = pl_module.train_dataloader()
        val_dataloader = pl_module.val_dataloader()

        representations_bank, y = self.get_all_representations(pl_module, val_dataloader)
        val_acc = self.knn_monitor(split='val', pl_module=pl_module, feature_bank=representations_bank, feature_labels=y, dataloader=val_dataloader)

        # log metrics
        pl_module.log('online_knn_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
    #'''