# import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from .base import BaseStrategy
from .utils import predict


def get_nn(features: np.ndarray, num_neighbors, metric='cosine'):
    neigh = NearestNeighbors(n_neighbors=num_neighbors+1, metric=metric, n_jobs=-1)
    neigh.fit(X=features)
    dists, neighbours = neigh.kneighbors(X=features, return_distance=True)
    return dists[:, 1:], neighbours[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClustStrategy(BaseStrategy):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.concatenate([dataset.train_ids, np.array(dataset.get_unlabeled_ids())])
        num_clusters = min(len(dataset.train_ids) + self.num_classes, self.MAX_NUM_CLUSTERS)

        m = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        _, y_preds, features_unlabeled = predict(
            m,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="TypiClust strategy (unlabeled)")
        
        _, y_preds_train, features_train = predict(
            m,
            dataset.train_dataloader(), 
            scoring="none", desc="TypiClust strategy (train)")
        
        features = np.concatenate([features_train, features_unlabeled])
        labels = kmeans(features, num_clusters=num_clusters)
        existing_indices = np.arange(len(dataset.train_ids))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        for i in range(budget):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        return all_ids[selected]
