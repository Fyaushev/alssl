# import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from .base import BaseStrategy
from .utils import get_cluster_acc, predict


def get_nn(features: np.ndarray, num_neighbors, metric='cosine'):
    neigh = NearestNeighbors(n_neighbors=num_neighbors+1, metric=metric, n_jobs=-1)
    neigh.fit(X=features)
    dists, neighbours = neigh.kneighbors(X=features, return_distance=True)
    return dists[:, 1:], neighbours[:, 1:]


def calculate_nn_score(features_e0, features_e1, num_neighbors):
    _, neighbors_e0 = get_nn(features_e0, num_neighbors)
    _, neighbors_e1 = get_nn(features_e1, num_neighbors)
    scores = np.array([
        len(set(orig) & set(finetuned))
        for orig, finetuned in tqdm(
            zip(neighbors_e0, neighbors_e1),
            total=len(neighbors_e1),
            desc="Calculating neighbor intersections",
        )
    ])
    return scores

def calculate_clust_consist_score(features_e0, features_e1, labels, cluster_curr, num_clusters, cluster_ids):
    labels_ = kmeans(features_e1 if not cluster_curr else features_e0, num_clusters=num_clusters)
    if cluster_curr:
        e0_labelling, e1_labelling = labels_, labels
    else:
        e0_labelling, e1_labelling = labels, labels_
    acc, mean_per_class_acc, e0_labelling = get_cluster_acc(e0_labelling, e1_labelling, return_matching=True)
    print(f"Cluster matching: {acc:.4f}%")
    
    cluster_consistency_scores = []
    for cluster_i in cluster_ids:
        e0_cluster_ids = np.argwhere(e0_labelling == cluster_i).ravel()
        e1_cluster_ids = np.argwhere(e1_labelling == cluster_i).ravel()
        consistent_cluster_ids = set(e0_cluster_ids) & set(e1_cluster_ids)
        consistency_score = len(consistent_cluster_ids) / len(e0_cluster_ids) * 100
        
        cluster_consistency_scores.append(consistency_score)
    return np.array(cluster_consistency_scores)


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class EmbClustStrategy(BaseStrategy):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 500

    def __init__(self, num_classes: int, cluster_curr: bool = False, clust_consist: bool = False, inverse: bool=False):
        self.num_classes = num_classes
        self.cluster_curr = cluster_curr
        self.clust_consist = clust_consist
        self.inverse = inverse
    
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.concatenate([dataset.train_ids, np.array(dataset.get_unlabeled_ids())])
        num_clusters = min(len(dataset.train_ids) + self.num_classes, self.MAX_NUM_CLUSTERS)

        m = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        _, _, features_unlabeled_e0 = predict(
            m,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="TypiClust strategy (unlabeled)")
        
        _, _, features_train_e0 = predict(
            m,
            dataset.train_dataloader(), 
            scoring="none", desc="TypiClust strategy (train)")
        
        _, _, features_unlabeled_e1 = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="TypiClust strategy (unlabeled)")
        
        _, _, features_train_e1 = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc="TypiClust strategy (train)")
        
        features_e0 = np.concatenate([features_train_e0, features_unlabeled_e0])
        features_e1 = np.concatenate([features_train_e1, features_unlabeled_e1])
        labels = kmeans(features_e1 if self.cluster_curr else features_e0, num_clusters=num_clusters)
        existing_indices = np.arange(len(dataset.train_ids))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        clust_consist_score = calculate_clust_consist_score(features_e0, features_e1, labels, self.cluster_curr, num_clusters, cluster_ids)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes, 'clust_consist_score': clust_consist_score})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        if self.clust_consist:
            clusters_df = clusters_df.sort_values(['clust_consist_score'], ascending=self.inverse)
        labels[existing_indices] = -1

        selected = []

        for i in range(budget):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            # in case we have too small cluster, calculate score among half of the cluster
            nn_score = calculate_nn_score(features_e0[indices], features_e1[indices], min(self.K_NN, len(indices) // 2))
            idx = indices[nn_score.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        return all_ids[selected]
