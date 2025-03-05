
import os

import numpy as np
from scipy.cluster.vq import vq
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from ..strategy.alssl.utils import load_or_compute
from .alssl.utils import (get_current_iteration, get_neighbours,
                          get_previous_interation_state_dict)
from .base import BaseStrategy
from .umaplike import construct_graph
from .utils import get_cluster_acc, predict

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def calculate_nn_scores(model, dataset, almodel, num_neighbours=250, metric="cosine", comb_score=False, load_from_prev_iter=False):
    """
    Calculate nearest-neighbor-based scores and return neighbors for further operations.
    """
    # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
    if get_current_iteration() and load_from_prev_iter:
        prev_model.load_state_dict(get_previous_interation_state_dict())

    # Compute embeddings and neighbors for original and finetuned models
    e0, neighbors_original, y_gt, y_pred_original, kmeans_original = get_neighbours(
        prev_model, dataset, "original", num_neighbours=num_neighbours, metric=metric, return_predicts_full=True
    )
    e1, neighbors_finetuned, y_gt, y_pred_finetuned, kmeans_finetuned = get_neighbours(
        model, dataset, "finetuned", num_neighbours=num_neighbours, metric=metric, return_predicts_full=True
    )

    # Calculate intersection scores between neighbors
    scores = np.array([
        len(set(orig) & set(finetuned))
        for orig, finetuned in tqdm(
            zip(neighbors_original, neighbors_finetuned),
            total=len(neighbors_finetuned),
            desc="Calculating neighbor intersections",
        )
    ])

    # Adjust scores if combined scoring is enabled
    if comb_score:
        mean_nn_scores = np.array([scores[orig].mean() for orig in neighbors_original])
        scores = scores / (mean_nn_scores + 1e-10)
    
    return scores, kmeans_finetuned, neighbors_finetuned


def calculate_nn_clust_scores(model, all_ids, dataset, almodel, inverse, cluster_curr, num_classes, num_classes_scale=2, num_neighbours=5, metric="cosine", comb_score=False, load_from_prev_iter=False):
    """
    Calculate nearest-neighbor-based scores and return neighbors for further operations.
    """
    # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
    if get_current_iteration() and load_from_prev_iter:
        prev_model.load_state_dict(get_previous_interation_state_dict())

    # Compute embeddings and neighbors for original and finetuned models
    _, _, e0 = predict(
        prev_model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc="original prediction")
    
    _, _, e1 = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc="finetuned prediction")
    
    def _run_kmeans(embeddings, _num_classes):
        kmeans = KMeans(n_clusters=_num_classes, n_init="auto").fit(embeddings)
        kmeans_labels = kmeans.predict(embeddings)
        centroids = kmeans.cluster_centers_
        closest, distances_to_centroids = vq(embeddings, centroids)
        return kmeans_labels, distances_to_centroids
    
    if cluster_curr:
        kmeans_labels, distances_to_centroids = _run_kmeans(e1, num_classes*num_classes_scale)
    else:
        kmeans_labels, distances_to_centroids = _run_kmeans(e0, num_classes*num_classes_scale)
        
    kmeans_centroids = []
    for cluster in np.unique(kmeans_labels):
        cluster_inds = np.argwhere(kmeans_labels == cluster).ravel()
        kmeans_centroids += [int(cluster_inds[np.argmin(distances_to_centroids[cluster_inds])])]

    neigh = NearestNeighbors(n_neighbors=num_neighbours+1, metric=metric, n_jobs=-1)

    neigh.fit(X=e1[kmeans_centroids])
    nn_finetuned = neigh.kneighbors(X=e1[kmeans_centroids], return_distance=False)[:, 1:]

    neigh.fit(X=e0[kmeans_centroids])
    nn_original = neigh.kneighbors(X=e0[kmeans_centroids], return_distance=False)[:, 1:]

    nn_scores = np.array([
        len(set(orig) & set(finetuned)) for orig, finetuned in zip(nn_original, nn_finetuned)
    ])
    if inverse:
        nn_scores = -nn_scores

    return all_ids[np.array(kmeans_centroids)[np.argsort(nn_scores)][:num_classes]]

def calculate_stability_clust_scores(model, all_ids, dataset, almodel, inverse, cluster_curr, num_classes, num_classes_scale=2, num_neighbours=5, metric="cosine", comb_score=False, load_from_prev_iter=False):
     # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
    if get_current_iteration() and load_from_prev_iter:
        prev_model.load_state_dict(get_previous_interation_state_dict())

    # Compute embeddings and neighbors for original and finetuned models
    _, _, e0 = predict(
        prev_model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc="original prediction")
    
    _, _, e1 = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc="finetuned prediction")
    
    def _run_kmeans(embeddings, _num_classes):
        kmeans = KMeans(n_clusters=_num_classes, n_init="auto").fit(embeddings)
        kmeans_labels = kmeans.predict(embeddings)
        centroids = kmeans.cluster_centers_
        closest, distances_to_centroids = vq(embeddings, centroids)
        return kmeans_labels, distances_to_centroids
    
    
    kmeans_labels_e1, distances_to_centroids_e1 = _run_kmeans(e1, num_classes*num_classes_scale)
    kmeans_labels_e0, distances_to_centroids_e0 = _run_kmeans(e0, num_classes*num_classes_scale)

    if cluster_curr:
        kmeans_labels, distances_to_centroids = kmeans_labels_e1, distances_to_centroids_e1
    else:
        kmeans_labels, distances_to_centroids = kmeans_labels_e0, distances_to_centroids_e0
        
    kmeans_centroids = []
    for cluster in np.unique(kmeans_labels):
        cluster_inds = np.argwhere(kmeans_labels == cluster).ravel()
        kmeans_centroids += [int(cluster_inds[np.argmin(distances_to_centroids[cluster_inds])])]
    kmeans_centroids = np.array(kmeans_centroids)
    
    acc, mean_per_class_acc, e0_labelling = get_cluster_acc(kmeans_labels_e0, kmeans_labels_e1, return_matching=True)
    e1_labelling = kmeans_labels_e1
    
    cluster_consistency_scores = []

    for cluster_i in np.unique(kmeans_labels):
        e0_cluster_ids = np.argwhere(e0_labelling == cluster_i).ravel()
        e1_cluster_ids = np.argwhere(e1_labelling == cluster_i).ravel()
        consistent_cluster_ids = set(e0_cluster_ids) & set(e1_cluster_ids)
        consistency_score = len(consistent_cluster_ids) / len(e0_cluster_ids) * 100
        
        cluster_consistency_scores.append(consistency_score)

    cluster_consistency_scores = np.array(cluster_consistency_scores)

    if inverse:
        cluster_consistency_scores = -cluster_consistency_scores

    return all_ids[kmeans_centroids[np.argsort(cluster_consistency_scores)][:num_classes]]


def calculate_umap_scores(model, dataset, almodel, num_neighbours=250, num_neighbours_umap=50, metric="cosine", load_from_prev_iter=False):
    """
    Calculate nearest-neighbor-based scores and return neighbors for further operations.
    """
    # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
    if get_current_iteration() and load_from_prev_iter:
        prev_model.load_state_dict(get_previous_interation_state_dict())

    # Compute embeddings and neighbors for original and finetuned models
    e0, neighbors_original, y_gt, y_pred_original, kmeans_original = get_neighbours(
        prev_model, dataset, "original", num_neighbours=num_neighbours, metric=metric, return_predicts_full=True
    )
    e1, neighbors_finetuned, y_gt, y_pred_finetuned, kmeans_finetuned = get_neighbours(
        model, dataset, "finetuned", num_neighbours=num_neighbours, metric=metric, return_predicts_full=True
    )
    scores = []
    for idx in tqdm(range(e0.shape[0]), desc='UMAP scores'):
        nn_original = neighbors_original[idx]
        nn_finetuned = neighbors_finetuned[idx]

        combined_neighbours = np.array(list(set(nn_original) & set(nn_finetuned)))

        graph_original = construct_graph(e0[combined_neighbours, :], num_neighbours_umap)
        graph_finetuned = construct_graph(e1[combined_neighbours, :], num_neighbours_umap)

        ce = - graph_original * np.log(graph_finetuned + 0.01) - (1 - graph_original) * np.log(1 - graph_finetuned + 0.01)

        scores.append(float(np.median(ce)))
    scores = np.array(scores)
    
    return scores, kmeans_finetuned, neighbors_finetuned

def calculate_typi_scores(model, dataset, almodel, num_neighbours=20, metric="cosine", cluster_curr=True):
    if cluster_curr:
        e1, dists_finetuned, neighbors_finetuned, kmeans_finetuned = get_neighbours(
            model, dataset, "finetuned", num_neighbours=num_neighbours, metric=metric, return_distance=True
        )
        return dists_finetuned.mean(axis=-1), kmeans_finetuned, neighbors_finetuned
    else:
        prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        e0, dists_original, neighbors_original, kmeans_original = get_neighbours(
            prev_model, dataset, "finetuned", num_neighbours=num_neighbours, metric=metric, return_distance=True
        )
        return dists_original.mean(axis=-1), kmeans_original, neighbors_original

class KMeansStrategy(BaseStrategy):
    """
    Random sampling of initial ids
    """
    def __init__(self, num_classes: int, samples_per_class: int = 1, is_random: bool = False, scoring=None, num_neighbours=None, num_neighbours_nms=None, comb_score=False, inverse=False, nms=True, cluster_curr=True, num_classes_scale=None):
        self.num_classes = num_classes

        self.samples_per_class = samples_per_class
        assert samples_per_class > 0, f"Number of samples per class should be positive. Current: {samples_per_class}"

        self.is_random = is_random
        self.scoring = scoring
        self.num_neighbours = num_neighbours
        self.num_neighbours_nms = num_neighbours_nms
        self.comb_score = comb_score
        self.inverse = inverse
        self.nms = nms
        self.cluster_curr = cluster_curr
        self.num_classes_scale = num_classes_scale
        if self.scoring is not None and self.is_random:
            raise ValueError('Poor KMeans setup, check `scoring` and `is_random` parameters.')

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.array(dataset.get_unlabeled_ids())
        if self.scoring == 'nn_clust':
            return calculate_nn_clust_scores(model, all_ids, dataset, almodel, self.inverse, self.cluster_curr, self.num_classes, self.num_classes_scale, self.num_neighbours)
        
        if self.scoring == 'clust_consist':
            return calculate_stability_clust_scores(model, all_ids, dataset, almodel, self.inverse, self.cluster_curr, self.num_classes, self.num_classes_scale, self.num_neighbours)
 
        def _predict_unlabeled():
            m = model if self.cluster_curr else almodel.get_lightning_module()(**almodel.get_hyperparameters())
            _, y_preds, embeddings = predict(
                m,
                dataset.unlabeled_dataloader(), 
                scoring="none", desc="KMeans strategy")
            return embeddings, y_preds
        
        embeddings, y_preds = load_or_compute(["embeddings_unlabeled.npy"], _predict_unlabeled)
        
        cluster_labels, distances_to_centroids = self.run_kmeans(embeddings)

        train_ids = []

        if self.scoring == 'nn':
            nn_scores, kmeans_finetuned, neighbors_finetuned = calculate_nn_scores(
                model, dataset, almodel, num_neighbours=self.num_neighbours, metric="cosine", comb_score=self.comb_score)
        elif self.scoring == 'umap':
            umap_scores, kmeans_finetuned, neighbors_finetuned = calculate_umap_scores(
                model, dataset, almodel, num_neighbours=self.num_neighbours, metric="cosine")
        elif self.scoring == 'typiclust':
            typi_scores, kmeans_finetuned, neighbors_finetuned = calculate_typi_scores(
                model, dataset, almodel, num_neighbours=self.num_neighbours, metric="cosine")
            
        if self.scoring in ['nn', 'typiclust', 'umap']:
            _, y_preds_train, embeddings_train = predict(
                model,
                dataset.train_dataloader(), 
                scoring="none", desc="KMeans strategy")
            neighbours_train = kmeans_finetuned.kneighbors(X=embeddings_train, return_distance=False)[:, 1:]
             

        for cluster in np.unique(cluster_labels):
            cluster_inds = np.argwhere(cluster_labels == cluster).ravel()

            if self.is_random:
                selected_cluster_inds = np.random.choice(cluster_inds, self.samples_per_class, replace=False)
            elif not self.is_random and (self.samples_per_class == 1) and (self.scoring is None):
                selected_cluster_inds = [cluster_inds[np.argmin(distances_to_centroids[cluster_inds])]]
            elif self.scoring == 'entropy':
                scores = entropy(y_preds[cluster_inds])
                if self.inverse:
                    scores = -scores
                if self.nms:
                    selected_cluster_inds = cluster_inds[nms_all_points(scores, neighbors_finetuned[cluster_inds], neighbours_train, self.samples_per_class, self.num_neighbours_nms)]
                else:
                    selected_cluster_inds = cluster_inds[np.argsort(-scores)[:self.samples_per_class]]
            elif self.scoring == 'nn':
                scores = nn_scores[cluster_inds]
                if self.inverse:
                    scores = -scores # most stable
                if self.nms:
                    selected_cluster_inds = cluster_inds[nms_all_points(-scores, neighbors_finetuned[cluster_inds], neighbours_train, self.samples_per_class, self.num_neighbours_nms)]
                else:
                    selected_cluster_inds = cluster_inds[np.argsort(scores)[:self.samples_per_class]]
            elif self.scoring == 'typiclust':
                scores = typi_scores[cluster_inds]
                if self.inverse:
                    scores = -scores
                if self.nms:
                    selected_cluster_inds = cluster_inds[nms_all_points(scores, neighbors_finetuned[cluster_inds], neighbours_train, self.samples_per_class, self.num_neighbours_nms)]
                else:
                    selected_cluster_inds = cluster_inds[np.argsort(scores)[:self.samples_per_class]]
            elif self.scoring == 'umap':
                scores = -umap_scores[cluster_inds]
                if self.inverse:
                    scores = -scores # most stable
                if self.nms:
                    selected_cluster_inds = cluster_inds[nms_all_points(scores, neighbors_finetuned[cluster_inds], neighbours_train, self.samples_per_class, self.num_neighbours_nms)]
                else:
                    selected_cluster_inds = cluster_inds[np.argsort(scores)[:self.samples_per_class]]
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


def nms_all_points(scores, neighbors_finetuned, neighbours_train, samples_per_class, num_neighbours_nms):
    """Apply NMS to both train and unlabeled data"""
    n_train = neighbours_train.shape[0]

    scores_train = [-np.inf for _ in range(n_train)]
    scores_combined = np.concatenate([scores_train, scores])
    print('scores_combined.shape', scores_combined.shape)

    neighbors_combined = np.concatenate([neighbours_train, neighbors_finetuned])
    print('neighbors_combined.shape', neighbors_combined.shape)
    selected_inds = non_max_suppression(-scores_combined, neighbors_combined[:, :num_neighbours_nms], max_boxes=n_train+samples_per_class)
    
    selected_inds = [idx - n_train for idx in selected_inds if idx not in range(n_train)][:samples_per_class]
    return selected_inds


def non_max_suppression(scores: np.ndarray, neighbors: np.ndarray, max_closeness: float = None, min_score=-np.inf, max_boxes=np.inf):
    """Select points with max scores, applying non-maximum suppression."""
    indices = np.arange(scores.size)
    mask = scores >= min_score
    scores, neighbors, indices = scores[mask], neighbors[mask], indices[mask]

    results = []
    while indices.size and len(results) < max_boxes:
        idx = scores.argmax()
        reference_neighbors = neighbors[idx]
        far_enough = ~np.isin(indices, reference_neighbors)

        if max_closeness is not None:
            far_enough &= far_enough < max_closeness

        far_enough[idx] = False
        results.append(indices[idx])

        scores, neighbors, indices = scores[far_enough], neighbors[far_enough], indices[far_enough]

    return results