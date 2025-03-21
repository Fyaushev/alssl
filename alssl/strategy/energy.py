# import faiss
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
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


def calculate_nn_score(features_e0, features_e1, num_neighbors, nnorm):
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

    mean_nn_scores_e0 = np.array([scores[orig].mean() for orig in neighbors_e0])
    mean_nn_scores_e1 = np.array([scores[fine].mean() for fine in neighbors_e1])

    if not nnorm:
        return mean_nn_scores_e0
    
    return mean_nn_scores_e0 / mean_nn_scores_e1

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


def calculate_clust_spread_score(features_e0, features_e1, e0_labelling, e1_labelling, cluster_ids):
    
    cluster_spread_scores = []
    for cluster_i in cluster_ids:
        e0_cluster_ids = np.argwhere(e0_labelling == cluster_i).ravel()
        e1_cluster_ids = np.argwhere(e1_labelling == cluster_i).ravel()
        spread_score = np.linalg.norm(features_e0[e0_cluster_ids] - features_e1[e0_cluster_ids], axis=1).mean()
        
        cluster_spread_scores.append(spread_score)
    return np.array(cluster_spread_scores)


def calculate_clust_energy_score(e0_labelling, e1_labelling, cluster_ids, energy_scores):
    cluster_energy_scores = []
    for cluster_i in cluster_ids:
        e0_cluster_ids = np.argwhere(e0_labelling == cluster_i).ravel()
        e1_cluster_ids = np.argwhere(e1_labelling == cluster_i).ravel()
        energy_score = energy_scores[e0_cluster_ids].mean()
        
        cluster_energy_scores.append(energy_score)
    return np.array(cluster_energy_scores)


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

def get_cluster_matching(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))
    return match

class EnergyStrategy(BaseStrategy):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 50

    def __init__(self, num_classes: int, cluster_curr: bool = False, add_nn: bool = False, sort_inverse: bool = False,  nnorm: bool = True):
        self.num_classes = num_classes
        self.cluster_curr = cluster_curr
        self.add_nn = add_nn
        self.sort_inverse = sort_inverse
        self.nnorm = nnorm

    def compute_class_centroids(self, embeddings, labels):
        """Compute class centroids and radii."""
        unique_labels = np.unique(labels)
        centroids = []
        radii = []
        
        for label in unique_labels:
            class_points = embeddings[labels == label]
            centroid = np.mean(class_points, axis=0)
            radius = np.max(cosine_distances(class_points, centroid[None]))
            centroids.append(centroid)
            radii.append(radius)
        
        return np.array(centroids), np.array(radii)
    
    def compute_potential_energy(self, embeddings, centroids, radii):
        """Compute the potential energy of each sample."""
        distances = cosine_distances(embeddings, centroids)
        energy = np.zeros(embeddings.shape[0])
        
        for i in range(embeddings.shape[0]):
            for j in range(len(centroids)):
                overlap = max(radii[j] - distances[i, j], 0)
                energy[i] += 0.5 * overlap ** 2
        
        return energy
    
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

        labels_e0 = kmeans(features_e0, num_clusters=num_clusters)
        labels_e1 = kmeans(features_e1, num_clusters=num_clusters)
        labels = labels_e1 if self.cluster_curr else labels_e0

        labels_e0 = get_cluster_matching(labels_e0, labels_e1)

        existing_indices = np.arange(len(dataset.train_ids))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        # clust_consist_score = calculate_clust_consist_score(features_e0, features_e1, labels, self.cluster_curr, num_clusters, cluster_ids)
        # if self.sort_inverse:
        #     clust_consist_score = -clust_consist_score
        
        centroids_e0, radii_e0 = self.compute_class_centroids(features_e0, labels_e0)
        centroids_e1, radii_e1 = self.compute_class_centroids(features_e1, labels_e1)

        energy_E0 = self.compute_potential_energy(features_e0, centroids_e0, radii_e0)
        energy_E1 = self.compute_potential_energy(features_e1, centroids_e1, radii_e1)
        energy_scores = energy_E0 - energy_E1

        nn_scores = calculate_nn_score(features_e0, features_e1, self.K_NN, self.nnorm)
        clust_energy_score = - calculate_clust_energy_score(labels_e0, labels_e1, cluster_ids, nn_scores)
        clust_spread_score = calculate_clust_spread_score(features_e0, features_e1, labels_e0, labels_e1, cluster_ids)

        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes, 'clust_energy_score': clust_energy_score})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        # clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        
        clusters_df = clusters_df[clusters_df.existing_count > 0]
        clusters_df = clusters_df.sort_values(['neg_cluster_size' ]) # 'clust_energy_score',
        # if self.clust_consist:
        # clusters_df = clusters_df.sort_values(['clust_consist_score'], ascending=True)
        labels[existing_indices] = -1

        selected = []

        for i in range(budget):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]

            energy_score = energy_scores[indices]

            # in case we have too small cluster, calculate score among half of the cluster
            # in case we have too small cluster, calculate density among half of the cluster
            
            typicality_e0 = calculate_typicality(features_e0[indices], min(self.K_NN, len(indices) // 2))
            # typicality_e1 = calculate_typicality(features_e1[indices], min(self.K_NN, len(indices) // 2))

            nn_score = nn_scores[indices]
            
            # if self.typinorm:
            #     typiscore = typicality_e0  / typicality_e1
            # else:
            #     typiscore = typicality_e0
            # typiscore = typiscore / typiscore.max()

            # if self.add_nn:
            #     nn_score = calculate_nn_score(features_e0[indices], features_e1[indices], min(self.K_NN, len(indices) // 2), self.nnorm)
            #     nn_score = nn_score / nn_score.max()
            #     score = energy_score + nn_score
            # else:
            score =  nn_score / nn_score.max()

            idx = indices[score.argmax()]
            # nn_score = calculate_nn_score(features_e0[indices], features_e1[indices], min(self.K_NN, len(indices) // 2))
            # nn_score = nn_score / nn_score.max()
            # idx = indices[(typicality  * nn_score).argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        return all_ids[selected]
