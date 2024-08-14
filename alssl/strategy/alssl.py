import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from ..utils import predict
from .base import BaseStrategy


def get_current_iteration():
    curr_dir = Path(os.getcwd())
    return int(curr_dir.name.split('_')[-1])

def get_previous_interation_state_dict():
    curr_dir = Path(os.getcwd())
    experiment_name = curr_dir.name.split('_')[0]
    current_iteration = int(curr_dir.name.split('_')[-1])
    previous_iteration_dir = curr_dir.parents[0] / f'{experiment_name}_iter_{current_iteration-1}'
    checkpoints = previous_iteration_dir.glob('*/*/checkpoints/*.ckpt')
    checkpoint_path = max(checkpoints, key=lambda t: os.stat(t).st_mtime)
    return torch.load(checkpoint_path)


class ALSSLStrategy(BaseStrategy):

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        previous_model = almodel.get_lightning_module()
        # load weights from previous iteration if available
        if get_current_iteration():
            previous_model.load_state_dict(get_previous_interation_state_dict())

        ys_unlabeled, _, original_embeddings = predict(
            previous_model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='original')
        
        _, _, finetuned_embeddings = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='finetuned')
        
        # fit KN 
        neigh_original = NearestNeighbors(n_neighbors=self.num_neighbours)
        neigh_original.fit(X=original_embeddings)

        neigh_finetuned = NearestNeighbors(n_neighbors=self.num_neighbours)
        neigh_finetuned.fit(X=finetuned_embeddings)

        scores = []
        for original_embedding, finetuned_embedding in tqdm(zip(original_embeddings, finetuned_embeddings), 
                                                            total=len(ys_unlabeled), desc="Finding neighbours for every unlabeled data point"):
            # for unlabeled point, find indices of closest neighbours in unlabeled set
            neighbours_original_inds = neigh_original.kneighbors(X=original_embedding[None], return_distance=False).flatten()[1:]
            neighbours_finetuned_inds = neigh_finetuned.kneighbors(X=finetuned_embedding[None], return_distance=False).flatten()[1:]

            # find number of intersecting neighbours
            number_saved_neighbours = len(set(neighbours_original_inds) & set(neighbours_finetuned_inds))

            scores.append(number_saved_neighbours)

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the lowest scores
        return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()
