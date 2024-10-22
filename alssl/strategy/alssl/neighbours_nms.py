
from typing import Callable

import numpy as np
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ...model.base import BaseALModel
from ..base import BaseStrategy
from .utils import (get_current_iteration, get_neighbours,
                    get_previous_interation_state_dict,
                    get_previous_iteration_dir)


def make_closeness(min_distance):
    return lambda p, ps: np.linalg.norm(p - ps, axis=-1) > min_distance

def closeness(reference_neighbors, keypoints):
    return np.isin(keypoints, reference_neighbors)


def non_max_suppression(keypoints: np.ndarray, scores: np.ndarray, neighbors: np.ndarray, max_closeness: float = None,
                        min_score=-np.inf, max_boxes=np.inf):
    """
    Parameters
    ----------
    keypoints
        shape (*spatial, *any)
    scores
        shape (*spatial)
        algorithm selects points with max scores
    closeness: Callable
        (*any), (N, *any) -> (N,)
        closeness of a single instance to a set of instances.
        if max_closeness is not provided - must return a bool mask directly.
    max_closeness
    min_score
    max_boxes

    Returns
    -------
    indices: (N,) * dim
        an array of indices in the initial `keypoints` array
    """
    shape = scores.shape
    indices = np.arange(scores.size)

    # initial filtering
    confident = scores >= min_score  # changed from >= due to ReLU!!
    scores = scores[confident]
    keypoints = keypoints[confident]
    neighbors = neighbors[confident]
    indices = indices[confident.flatten()]

    results = []
    while keypoints.size and len(results) < max_boxes:
        idx = scores.argmax()
        reference = keypoints[idx]
        reference_neighbors = neighbors[idx][1:] # first one is our point
        far_enough = np.isin(keypoints, reference_neighbors)
        # closeness(reference_neighbors, keypoints)
        if max_closeness is not None:
            assert far_enough.dtype != bool, far_enough.dtype
            far_enough = far_enough < max_closeness
        else:
            assert far_enough.dtype == bool, far_enough.dtype
        results.append(np.unravel_index(indices[idx], shape))

        scores = scores[far_enough]
        keypoints = keypoints[far_enough]
        neighbors = neighbors[far_enough]
        indices = indices[far_enough]

    if not results:
        return ([],) * len(shape)

    return tuple(list(axis) for axis in zip(*results))

class NeighboursNMSStrategy(BaseStrategy):

    def __init__(self, num_neighbours: int, metric='minkowski'):
        self.num_neighbours = num_neighbours + 1 # NearestNeighbors outputs point itself as neighbour
        self.metric = metric

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        # get neighbours for previous iteration and save for later
        # if get_current_iteration():
        #     neighbours_original_inds = np.load(get_previous_iteration_dir() / 'neighbours_inds.npy')    
        # else:
        previous_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        # load weights from previous iteration if available
        if get_current_iteration():
            previous_model.load_state_dict(get_previous_interation_state_dict())

        embeddings_original, neighbours_original_inds = get_neighbours(previous_model, dataset, desc="original", num_neighbours=self.num_neighbours, metric=self.metric)

        # generate neighbours for current iteration and save for later
        embeddings_finetuned, neighbours_finetuned_inds = get_neighbours(model, dataset, desc="finetuned", num_neighbours=self.num_neighbours, metric=self.metric)
        np.save('neighbours_inds.npy', neighbours_finetuned_inds)

        scores = []
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds), 
                                                            total=neighbours_finetuned_inds.shape[0], 
                                                            desc="Finding neighbours intersection for every unlabeled data point"):
            # find number of intersecting neighbours
            number_saved_neighbours = len(set(neighbours_original) & set(neighbours_finetuned))

            scores.append(number_saved_neighbours)

        nms_indices = non_max_suppression(embeddings_finetuned, np.array(scores), neighbours_finetuned_inds)


        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the lowest scores
        return np.array(unlabeled_ids)[nms_indices][:budget].tolist()

