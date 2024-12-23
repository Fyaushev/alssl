import numpy as np
import torch
from scipy.special import softmax
from torch import nn

from ..data.base import ALDataModule
from .base import BaseStrategy
from .utils import predict


class EntropyStrategy(BaseStrategy):
    def __init__(self, random_proportion: float = .9, iter_weight: float = 0):
        self.random_proportion = random_proportion
        self.iter_weight = iter_weight

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, iter_n: int):
        
        unlabeled_dataset = dataset.unlabeled_dataloader()
        
        scores = predict(
            model, 
            unlabeled_dataset, 
            scoring="individual", 
            scoring_function=self.scoring_function)

        unlabeled_ids = dataset.get_unlabeled_ids()

        random_proportion = max(self.random_proportion - iter_n * self.iter_weight, 0)

        entropy_selected_ids = np.array(unlabeled_ids)[np.argsort(scores)][:int(budget * (1-random_proportion))].tolist()

        unselected_ids = list(set(unlabeled_ids) ^ set(entropy_selected_ids))

        random_selected_ids = np.random.choice(unselected_ids, size=int(budget * random_proportion), replace=False).tolist()

        return entropy_selected_ids + random_selected_ids

    def scoring_function(self, gt, pred, embeddings):
        """
        calculate inverse entropy: min is worst
        """
        proba = softmax(pred, 1)
        log_proba = np.log(proba)
        U = (proba*log_proba).sum(1)
        return U