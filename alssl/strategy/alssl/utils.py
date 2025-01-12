import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
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
    # TODO: fix
    # if all(Path(fp).exists() for fp in filepaths):
    #     if len(filepaths) == 1:
    #         return np.load(filepaths[0])
    #     return [np.load(fp) for fp in filepaths]
    
    # Compute data and save to all files
    results = compute_fn(*args, **kwargs)
    if to_save:
        if len(filepaths) == 1:
            np.save(filepaths[0], results)
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
        neigh_base: str = 'embeddings',
        return_predicts: bool = False,
        return_predicts_full: bool = False,
    ):
    '''
    Obtain embeddings and find `num_neighbours` nearest neighbours in the same embedding space
    '''
    ys, y_preds, embeddings = predict(
        model,
        dataset.unlabeled_dataloader(), 
        scoring="none", desc=desc)

    # fit KN 
    neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric, n_jobs=-1)
    if neigh_base == 'embeddings':
        to_fit = embeddings
    elif neigh_base == 'proba':
        proba = softmax(y_preds, 1)
        to_fit = proba

    neigh.fit(X=to_fit)

    if return_distance:
        dists, neighbours = neigh.kneighbors(X=to_fit, return_distance=return_distance)
        return embeddings, dists[:, 1:], neighbours[:, 1:]
    elif return_predicts:
        return embeddings, neigh.kneighbors(X=to_fit, return_distance=return_distance)[:, 1:], y_preds
    elif return_predicts_full:
        return embeddings, neigh.kneighbors(X=to_fit, return_distance=return_distance)[:, 1:], ys, y_preds
    else:
        return embeddings, neigh.kneighbors(X=to_fit, return_distance=return_distance)[:, 1:]
