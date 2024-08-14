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
from .base import BaseStrategy
from .utils import predict


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


class ALSSLStrategy(BaseStrategy):

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours + 1 # NearestNeighbors outputs point itself as neighbour

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        # get neighbours for previous iteration and save for later
        if get_current_iteration():
            neighbours_original_inds = np.load(get_previous_iteration_dir() / 'neighbours_inds.npy')    
        else:
            previous_model = almodel.get_lightning_module()
            # load weights from previous iteration if available
            if get_current_iteration():
                previous_model.load_state_dict(get_previous_interation_state_dict())

            neighbours_original_inds = self.get_neighbours(previous_model, dataset, desc="original")

        # generate neighbours for current iteration and save for later
        neighbours_finetuned_inds = self.get_neighbours(model, dataset, desc="finetuned")
        np.save('neighbours_inds.npy', neighbours_finetuned_inds)

        scores = []
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds), 
                                                            total=neighbours_finetuned_inds.shape[0], desc="Finding neighbours intersection for every unlabeled data point"):
            # find number of intersecting neighbours
            number_saved_neighbours = len(set(neighbours_original) & set(neighbours_finetuned))

            scores.append(number_saved_neighbours)

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the lowest scores
        return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()
    

    def get_neighbours(self, model: nn.Module, dataset: ALDataModule, desc: str):
        _, _, embeddings = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc=desc)

        # fit KN 
        neigh = NearestNeighbors(n_neighbors=self.num_neighbours)
        neigh.fit(X=embeddings)
        
        return neigh.kneighbors(X=embeddings, return_distance=False)[:, 1:]
