
import numpy as np
from scipy.cluster.vq import vq
from scipy.special import softmax
from sklearn.cluster import KMeans
from torch import nn

from ..data.base import ALDataModule
from ..strategy.alssl.utils import load_or_compute
from .base import BaseStrategy
from .utils import predict


class KMeansStrategy(BaseStrategy):
    """
    Random sampling of initial ids
    """
    def __init__(self, num_classes: int, samples_per_class: int = 1, is_random: bool = False, scoring=None):
        self.num_classes = num_classes

        self.samples_per_class = samples_per_class
        assert samples_per_class > 0, f"Number of samples per class should be positive. Current: {samples_per_class}"

        self.is_random = is_random
        self.scoring = scoring
        if self.scoring is not None and self.is_random:
            raise ValueError('Poor KMeans setup, check `scoring` and `is_random` parameters.')

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, *args) -> list:
        all_ids = np.array(dataset.get_unlabeled_ids())
        
        def _predict_unlabeled():
            _, y_preds, embeddings = predict(
                model,
                dataset.unlabeled_dataloader(), 
                scoring="none", desc="KMeans strategy")
            return embeddings, y_preds
        
        embeddings, y_preds = load_or_compute(["embeddings_unlabeled.npy"], _predict_unlabeled)
        
        cluster_labels, distances_to_centroids = self.run_kmeans(embeddings)

        train_ids = []

        for cluster in np.unique(cluster_labels):
            cluster_inds = np.argwhere(cluster_labels == cluster).ravel()

            if self.is_random:
                selected_cluster_inds = np.random.choice(cluster_inds, self.samples_per_class, replace=False)
            elif not self.is_random and (self.samples_per_class == 1) and (self.scoring is None):
                selected_cluster_inds = [cluster_inds[np.argmin(distances_to_centroids[cluster_inds])]]
            elif self.scoring == 'entropy':
                entropy_scores = entropy(y_preds[cluster_inds])
                selected_cluster_inds = cluster_inds[np.argsort(-entropy_scores)[:self.samples_per_class]]
            else:
                raise ValueError('Poor KMeans setup, check `samples_per_class` and `is_random` parameters.')
            
            train_ids.extend(all_ids[selected_cluster_inds])

        return train_ids
    
    def run_kmeans(self, embeddings):
        kmeans = KMeans(n_clusters=self.num_classes, n_init="auto").fit(embeddings)
        kmeans_labels = kmeans.predict(embeddings)
        centroids = kmeans.cluster_centers_
        closest, distances_to_centroids = vq(embeddings, centroids)
        return kmeans_labels, distances_to_centroids
    
    
def entropy(pred):
    """Calculate entropy: max is worst."""
    proba = softmax(pred, axis=1)
    return -np.sum(proba * np.log(proba), axis=1)