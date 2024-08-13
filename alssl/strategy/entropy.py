import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from ..data.base import ALDataModule
from ..utils import predict
from .base import BaseStrategy


class EntropyStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int):
        
        unlabeled_dataset = dataset.unlabeled_dataloader()
        
        scores = predict(
            model, 
            unlabeled_dataset, 
            scoring="individual", 
            scoring_function=self.scoring_function)

        unlabeled_ids = dataset.get_unlabeled_ids()
        return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()

    def scoring_function(self, gt, pred, embeddings):
        """
        calculate inverse entropy: min is worst
        """
        proba = softmax(pred, 1)
        log_proba = np.log(proba)
        U = (proba*log_proba).sum(1)
        return U