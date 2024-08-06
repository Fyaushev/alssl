import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from torch import nn

from ..data.base import ALDataModule
from ..utils import predict
from .base import BaseStrategy


def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for _ in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs


class CoresetStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int):
        _, _, embeddings_unlabeled = predict(
            model.get_lightning_module(), 
            dataset.unlabeled_dataloader(), 
            scoring="none")
        
        _, _, embeddings_train = predict(
            model.get_lightning_module(), 
            dataset.train_dataloader(), 
            scoring="none")

        chosen_idxs = furthest_first(embeddings_unlabeled, embeddings_train, budget)
        unlabeled_ids = dataset.get_unlabeled_ids()

        return unlabeled_ids[chosen_idxs]
