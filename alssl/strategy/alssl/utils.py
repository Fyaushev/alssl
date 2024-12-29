import os
from pathlib import Path

import numpy as np
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

def get_previous_interation_state_dict():
    previous_iteration_dir = get_previous_iteration_dir()
    print("previous_iteration_dir", previous_iteration_dir)
    return torch.load(last_checkpoint(previous_iteration_dir))["state_dict"]


def load_or_compute(filepaths, compute_fn, to_save:bool = False, *args, **kwargs):
    """
    Load data from multiple files if they all exist, otherwise compute and save.
    
    Parameters:
        filepaths (list of str): List of file paths corresponding to the data to be loaded/saved.
        compute_fn (callable): Function to compute the data if files are missing.
    
    Returns:
        list of np.ndarray: Loaded or computed data.
    """
    if all(Path(fp).exists() for fp in filepaths):
        if len(filepaths) == 1:
            return np.load(filepaths[0])
        return [np.load(fp) for fp in filepaths]
    
    # Compute data and save to all files
    results = compute_fn(*args, **kwargs)
    if to_save:
        for fp, result in zip(filepaths, results):
            np.save(fp, result)
    return results


def get_neighbours(
        model: nn.Module, 
        dataset: ALDataModule, 
        desc: str, 
        num_neighbours: int, 
        return_distance: bool = False, 
        metric: str = 'minkowski',
        return_predicts: bool = False,
    ):
    '''
    Obtain embeddings and find `num_neighbours` nearest neighbours in the same embedding space
    '''
    ys, y_preds, embeddings = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc=desc)
    np.save('y_gt.npy', ys)
    np.save('y_preds.npy', y_preds)

    # fit KN 
    neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric, n_jobs=-1)
    neigh.fit(X=embeddings)

    if return_distance:
        dists, neighbours = neigh.kneighbors(X=embeddings, return_distance=return_distance)
        return embeddings, dists[:, 1:], neighbours[:, 1:]
    elif return_predicts:
        return embeddings, neigh.kneighbors(X=embeddings, return_distance=return_distance)[:, 1:], y_preds
    else:
        return embeddings, neigh.kneighbors(X=embeddings, return_distance=return_distance)[:, 1:]
