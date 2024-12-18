from pathlib import Path

import numpy as np
from scipy.special import softmax
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ...model.base import BaseALModel
from ..base import BaseStrategy
from .utils import (get_current_iteration, get_neighbours,
                    get_previous_interation_state_dict, load_or_compute)


def entropy(pred):
    """Calculate entropy: max is worst."""
    proba = softmax(pred, axis=1)
    return -np.sum(proba * np.log(proba), axis=1)


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


class NeighboursStrategy(BaseStrategy):
    def __init__(self, num_neighbours, metric="minkowski", fixed_budget=True, load_from_prev_iter=True, 
                 finetune=True, p_loss=False, nms=True, nms_e0=True, comb_score=True):
        self.num_neighbours = num_neighbours + 1
        self.metric = metric
        self.fixed_budget = fixed_budget
        self.nn_thr = int(num_neighbours * 0.2)
        self.load_from_prev_iter = load_from_prev_iter
        self.finetune = finetune
        self.include_param_loss = p_loss
        self.nms = nms
        self.nms_e0 = nms_e0
        self.comb_score = comb_score

    def _build_filename(self, prefix, short=True, ext="npy"):
        """Construct a dynamic filename based on strategy parameters."""
        options = {
            "num_neighbours": self.num_neighbours,
            "metric": self.metric,
            "fixed_budget": self.fixed_budget,
            "finetune": self.finetune,
            "p_loss": self.include_param_loss,
            "nms": self.nms,
            "nms_e0": self.nms_e0,
            "comb_score": self.comb_score,
        }
        options_str = "_".join(f"{k}={v}" for k, v in options.items() if v)
        return f"{prefix}.{ext}" if short else f"{prefix}_{options_str}.{ext}"

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):
        def compute_original_embeddings():
            prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
            if get_current_iteration() and self.load_from_prev_iter:
                prev_model.load_state_dict(get_previous_interation_state_dict())
            return get_neighbours(prev_model, dataset, "original", num_neighbours=self.num_neighbours, metric=self.metric, return_predicts=False)

        e0, neighbours_original_inds = load_or_compute(
            [self._build_filename("embeddings_original"), self._build_filename("neighbours_original_inds")],
            compute_original_embeddings,
        )

        def compute_finetuned_embeddings():
            e1, neighbors, pred = get_neighbours(
                model, dataset, "finetuned", num_neighbours=self.num_neighbours, metric=self.metric, return_predicts=True
            )
            entropy_scores = entropy(pred)
            return e1, neighbors, entropy_scores

        e1, neighbours_finetuned_inds, entropy_scores = load_or_compute(
            [
                self._build_filename("embeddings_finetuned", short=False),
                self._build_filename("neighbours_finetuned_inds", short=False),
                self._build_filename("entropy_scores"),
            ],
            compute_finetuned_embeddings,
        )

        scores = np.array([
            len(set(neighbors_orig) & set(neighbors_finetuned))
            for neighbors_orig, neighbors_finetuned in tqdm(
                zip(neighbours_original_inds, neighbours_finetuned_inds),
                total=neighbours_finetuned_inds.shape[0],
                desc="Finding neighbors intersection for every unlabeled data point",
            )
        ])
        np.save(self._build_filename("scores", short=False), scores)

        if self.comb_score:
            scores = -(entropy_scores / entropy_scores.max()) * (1 - scores / scores.max())
            np.save(self._build_filename("scores_combined", short=False), scores)

        unlabeled_ids = dataset.get_unlabeled_ids()
        if self.nms:
            neighbors = neighbours_original_inds if self.self.nms_e0 else neighbours_finetuned_inds
            nms_indices = non_max_suppression(-scores, neighbors, max_boxes=budget)
            return np.array(unlabeled_ids)[nms_indices].tolist()

        sorting = np.argsort(scores)
        mask = np.ones_like(sorting, dtype=bool) if self.fixed_budget else scores[sorting] < self.nn_thr
        return np.array(unlabeled_ids)[sorting][mask][:budget].tolist()
