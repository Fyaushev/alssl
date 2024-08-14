import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from torch import nn

from ..data.base import ALDataModule
from ..utils import predict
from .base import BaseStrategy
from .coreset import furthest_first


class CDALStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):
        _, y_preds_unlabeled, _ = predict(
            model.get_lightning_module(), 
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        _, y_preds_train, _ = predict(
            model.get_lightning_module(), 
            dataset.train_dataloader(), 
            scoring="none", desc='train')
        
        proba_unlabeled = softmax(y_preds_unlabeled, 1)
        proba_train = softmax(y_preds_train, 1)

        chosen_idxs = furthest_first(proba_unlabeled, proba_train, budget)
        unlabeled_ids = dataset.get_unlabeled_ids()

        return np.array(unlabeled_ids)[chosen_idxs.astype(int)].tolist()