import os
from pathlib import Path

import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn

from ...data.base import ALDataModule
from ..utils import predict


def get_current_iteration():
    curr_dir = Path(os.getcwd())
    return int(curr_dir.name.split('_')[-1])

def get_previous_iteration_dir():
    curr_dir = Path(os.getcwd())
    experiment_name = curr_dir.name.split('_')[0]
    current_iteration = int(curr_dir.name.split('_')[-1])
    previous_iteration_dir = curr_dir.parents[0] / f'{experiment_name}_iter_{current_iteration-1}'
    return previous_iteration_dir

def get_previous_interation_state_dict():
    previous_iteration_dir = get_previous_iteration_dir()
    checkpoints = previous_iteration_dir.glob('*/*/checkpoints/*.ckpt')
    checkpoint_path = max(checkpoints, key=lambda t: os.stat(t).st_mtime)
    return torch.load(checkpoint_path)["state_dict"]


def get_neighbours(model: nn.Module, dataset: ALDataModule, desc: str, num_neighbours: int, return_distance: bool = False):
    '''
    Obtain embeddings and find `num_neighbours` nearest neighbours in the same embedding space
    '''
    _, _, embeddings = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc=desc)

    # fit KN 
    neigh = NearestNeighbors(n_neighbors=num_neighbours)
    neigh.fit(X=embeddings)

    if return_distance:
        dists, neighbours = neigh.kneighbors(X=embeddings, return_distance=return_distance)
        return embeddings, dists[:, 1:], neighbours[:, 1:]
    else:
        return embeddings, neigh.kneighbors(X=embeddings, return_distance=return_distance)[:, 1:]
