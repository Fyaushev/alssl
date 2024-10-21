import os
from pathlib import Path

import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn

from ...al.utils import last_checkpoint
from ...data.base import ALDataModule
from ..utils import predict


def get_current_iteration():
    curr_dir = Path(os.getcwd()).name #.split('_')
    if curr_dir == 'zero_iteration':
        return 0
    else:
        return int(curr_dir.split('_')[-1])

def get_previous_iteration_dir():
    curr_dir = Path(os.getcwd())
    curr_iter = get_current_iteration()
    if curr_iter == 0:
        return
    elif curr_iter == 1:
        return curr_dir.parent.parent.parent / 'zero_iteration'
    else:
        return curr_dir.parents[0] / f'iter_{curr_iter - 1}'
    # experiment_name = curr_dir.name.split('_')[0]
    # current_iteration = int(curr_dir.name.split('_')[-1])
    # previous_iteration_dir = curr_dir.parents[0] / f'{experiment_name}_iter_{current_iteration-1}'
    # return previous_iteration_dir

def get_previous_interation_state_dict():
    previous_iteration_dir = get_previous_iteration_dir()
    print("previous_iteration_dir", previous_iteration_dir)
    return torch.load(last_checkpoint(previous_iteration_dir))["state_dict"]


def get_neighbours(model: nn.Module, dataset: ALDataModule, desc: str, num_neighbours: int, return_distance: bool = False, metric: str = 'minkowski'):
    '''
    Obtain embeddings and find `num_neighbours` nearest neighbours in the same embedding space
    '''
    _, _, embeddings = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc=desc)

    # fit KN 
    neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric)
    neigh.fit(X=embeddings)

    if return_distance:
        dists, neighbours = neigh.kneighbors(X=embeddings, return_distance=return_distance)
        return embeddings, dists[:, 1:], neighbours[:, 1:]
    else:
        return embeddings, neigh.kneighbors(X=embeddings, return_distance=return_distance)[:, 1:]
