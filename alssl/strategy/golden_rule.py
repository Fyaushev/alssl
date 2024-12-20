import numpy as np
import torch
from torch import nn

from ..data.base import ALDataModule
from .base import BaseStrategy
from .utils import predict


class GRStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):
        
        unlabeled_dataset = dataset.unlabeled_dataloader()
        
        scores = predict(
            model, 
            unlabeled_dataset, 
            scoring="individual", 
            scoring_function=self.scoring_function)

        unlabeled_ids = dataset.get_unlabeled_ids()
        return np.array(unlabeled_ids)[np.argsort(-scores)][:budget].tolist()

    def scoring_function(self, gt, pred, embeddings):
        return nn.functional.cross_entropy(torch.Tensor(pred), gt.cpu().long(), reduction='none').numpy()