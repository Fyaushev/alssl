import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from .base import BaseStrategy
from .utils import predict


def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for _ in tqdm(range(n), desc="Coreset distances calculation"):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return np.array(idxs)


class CoresetStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):
        _, _, embeddings_unlabeled = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        _, _, embeddings_train = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc='train')

        chosen_idxs = furthest_first(embeddings_unlabeled, embeddings_train, budget)
        unlabeled_ids = dataset.get_unlabeled_ids()

        return np.array(unlabeled_ids)[chosen_idxs.astype(int)].tolist()
