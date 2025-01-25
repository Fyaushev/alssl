
import os

import numpy as np
from scipy.cluster.vq import vq
from scipy.special import softmax
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from ..strategy.alssl.utils import load_or_compute
from .alssl.utils import (get_current_iteration, get_neighbours,
                          get_previous_interation_state_dict)
from .base import BaseStrategy
from .utils import predict

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def calculate_nn_scores(model, dataset, almodel, num_neighbours=500, metric="cosine", comb_score=False, load_from_prev_iter=False):
    """
    Calculate nearest-neighbor-based scores and return neighbors for further operations.
    """
    # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
    if get_current_iteration() and load_from_prev_iter:
        prev_model.load_state_dict(get_previous_interation_state_dict())

    # Compute embeddings and neighbors for original and finetuned models
    e0, neighbors_original, y_gt, y_pred_original = get_neighbours(
        prev_model, dataset, "original", num_neighbours=num_neighbours, metric=metric, return_predicts_full=True
    )
    e1, neighbors_finetuned, y_gt, y_pred_finetuned = get_neighbours(
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
    
    return scores


class KMeansStrategy(BaseStrategy):
    """
    Random sampling of initial ids
    """
    def __init__(self, num_classes: int, samples_per_class: int = 1, is_random: bool = False, scoring=None, num_neighbours=None, comb_score=False, inverse=False):
        self.num_classes = num_classes

        self.samples_per_class = samples_per_class
        assert samples_per_class > 0, f"Number of samples per class should be positive. Current: {samples_per_class}"

        self.is_random = is_random
        self.scoring = scoring
        self.num_neighbours = num_neighbours
        self.comb_score = comb_score
        self.inverse = inverse
        if self.scoring is not None and self.is_random:
            raise ValueError('Poor KMeans setup, check `scoring` and `is_random` parameters.')

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
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

        if self.scoring == 'nn':
            nn_scores = calculate_nn_scores(
                model, dataset, almodel, num_neighbours=self.num_neighbours, metric="cosine", comb_score=self.comb_score)

        for cluster in np.unique(cluster_labels):
            cluster_inds = np.argwhere(cluster_labels == cluster).ravel()

            if self.is_random:
                selected_cluster_inds = np.random.choice(cluster_inds, self.samples_per_class, replace=False)
            elif not self.is_random and (self.samples_per_class == 1) and (self.scoring is None):
                selected_cluster_inds = [cluster_inds[np.argmin(distances_to_centroids[cluster_inds])]]
            elif self.scoring == 'entropy':
                entropy_scores = entropy(y_preds[cluster_inds])
                selected_cluster_inds = cluster_inds[np.argsort(-entropy_scores)[:self.samples_per_class]]
            elif self.scoring == 'nn':
                scores = nn_scores[cluster_inds]
                if self.inverse:
                    scores = -scores
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