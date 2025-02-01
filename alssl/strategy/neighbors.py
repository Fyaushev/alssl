
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


class NNStrategy(BaseStrategy):
    """
    Random sampling of initial ids
    """
    def __init__(self, num_classes: int, is_random: bool = False, num_neighbours=None, num_neighbours_nms=None, comb_score=False, inverse=False):
        self.num_classes = num_classes
        self.is_random = is_random
        self.num_neighbours = num_neighbours
        self.num_neighbours_nms = num_neighbours_nms
        self.comb_score = comb_score
        self.inverse = inverse

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.array(dataset.get_unlabeled_ids())
        
        def _predict_unlabeled():
            _, y_preds, embeddings = predict(
                model,
                dataset.unlabeled_dataloader(), 
                scoring="none", desc="KMeans strategy")
            return embeddings, y_preds
        
        embeddings, y_preds = load_or_compute(["embeddings_unlabeled.npy"], _predict_unlabeled)
        
        nn_scores, kmeans_finetuned, neighbors_finetuned = calculate_nn_scores(
            model, dataset, almodel, num_neighbours=self.num_neighbours, metric="cosine", comb_score=self.comb_score)
    
        
        _, y_preds_train, embeddings_train = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc="KMeans strategy")
        neighbours_train = kmeans_finetuned.kneighbors(X=embeddings_train, return_distance=False)[:, 1:]
            
        scores = nn_scores
        if self.inverse:
            scores = -scores

        train_ids = all_ids[nms_all_points(scores, neighbors_finetuned, neighbours_train, budget, self.num_neighbours_nms)]

        return train_ids


def nms_all_points(scores, neighbors_finetuned, neighbours_train, budget, num_neighbours_nms):
    """Apply NMS to both train and unlabeled data"""
    n_train = neighbours_train.shape[0]

    scores_train = [-np.inf for _ in range(n_train)]
    scores_combined = np.concatenate([scores_train, scores])
    print('scores_combined.shape', scores_combined.shape)

    neighbors_combined = np.concatenate([neighbours_train, neighbors_finetuned])
    print('neighbors_combined.shape', neighbors_combined.shape)
    selected_inds = non_max_suppression(-scores_combined, neighbors_combined[:, :num_neighbours_nms], max_boxes=n_train+budget)
    
    selected_inds = [idx - n_train for idx in selected_inds if idx not in range(n_train)][:budget]
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