
import os

import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from torch import nn

from ..data.base import ALDataModule
from ..strategy.alssl.utils import load_or_compute
from ..strategy.utils import predict
from .base import BaseColdStart

os.environ["OPENBLAS_NUM_THREADS"] = "1"


class KMeansColdStart(BaseColdStart):
    """
    Random sampling of initial ids
    """
    def __init__(self, initial_train_size: int, random_seed: int, num_classes: int, samples_per_class: int = 3, is_random: bool = True):
        self.initial_train_size = initial_train_size
        self.random_seed = random_seed
        self.num_classes = num_classes

        self.samples_per_class = samples_per_class
        assert samples_per_class > 0, f"Number of samples per class should be positive. Current: {samples_per_class}"

        self.is_random = is_random

    def select_ids(self, model: nn.Module, dataset: ALDataModule, **kwargs) -> list:
        all_ids = np.array(dataset.get_unlabeled_ids())
        cluster_labels, distances_to_centroids = self.run_kmeans(model, dataset)

        train_ids = []

        np.random.seed(self.random_seed)
        for cluster in np.unique(cluster_labels):
            cluster_inds = np.argwhere(cluster_labels == cluster).ravel()

            if self.is_random:
                selected_cluster_inds = np.random.choice(cluster_inds, self.samples_per_class, replace=False)
            elif not self.is_random and (self.samples_per_class == 1):
                selected_cluster_inds = [cluster_inds[np.argmin(distances_to_centroids[cluster_inds])]]
            else:
                raise ValueError('Poor KMeans setup, check `samples_per_class` and `is_random` parameters.')
            
            train_ids.extend(all_ids[selected_cluster_inds])

        return train_ids
    
    def run_kmeans(self, model: nn.Module, dataset: ALDataModule):
    
        def _predict_unlabeled():
            _, _, embeddings = predict(
                model,
                dataset.unlabeled_dataloader(), 
                scoring="none", desc="KMeans coldstart")
            return embeddings
        
        embeddings = load_or_compute(["embeddings_unlabeled.npy"], _predict_unlabeled)

        kmeans = KMeans(n_clusters=self.num_classes, random_state=self.random_seed, n_init="auto").fit(embeddings)
        kmeans_labels = kmeans.predict(embeddings)
        centroids = kmeans.cluster_centers_
        closest, distances_to_centroids = vq(embeddings, centroids)
        return kmeans_labels, distances_to_centroids
    
    