
from pathlib import Path

import numpy as np
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ...model.base import BaseALModel
from ..base import BaseStrategy
from .utils import (get_current_iteration, get_neighbours,
                    get_previous_interation_state_dict,
                    get_previous_iteration_dir)


class NeighboursStrategy(BaseStrategy):

    def __init__(self, num_neighbours: int, metric='minkowski', fixed_budget:bool=True, load_from_prev_iter:bool=True):
        self.num_neighbours = num_neighbours + 1 # NearestNeighbors outputs point itself as neighbour
        self.metric = metric
        self.fixed_budget = fixed_budget
        self.nn_thr = int(num_neighbours * 0.2)
        self.load_from_prev_iter = load_from_prev_iter

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        if Path('embeddings_original.npy').exists() and Path('neighbours_original_inds.npy').exists():
            e0 = np.load('embeddings_original.npy')
            neighbours_original_inds = np.load('neighbours_original_inds.npy')
        else:
            previous_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
            # load weights from previous iteration if available
            if get_current_iteration() and self.load_from_prev_iter:
                previous_model.load_state_dict(get_previous_interation_state_dict())

            e0, neighbours_original_inds = get_neighbours(previous_model, dataset, desc="original", num_neighbours=self.num_neighbours, metric=self.metric)
            np.save('embeddings_original.npy', e0)
            np.save('neighbours_original_inds.npy', neighbours_original_inds)
        
        # generate neighbours for current iteration and save for later
        if Path(f'embeddings_finetuned_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy').exists() and Path(f'neighbours_finetuned_inds_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy').exists():
            e1 = np.load(f'embeddings_finetuned_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy')
            neighbours_finetuned_inds = np.load(f'neighbours_finetuned_inds_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy')
        else:
            e1, neighbours_finetuned_inds = get_neighbours(model, dataset, desc="finetuned", num_neighbours=self.num_neighbours, metric=self.metric)
            np.save(f'embeddings_finetuned_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy', e1)
            np.save(f'neighbours_finetuned_inds_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy', neighbours_finetuned_inds)

        scores = []
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds), 
                                                            total=neighbours_finetuned_inds.shape[0], 
                                                            desc="Finding neighbours intersection for every unlabeled data point"):
            # find number of intersecting neighbours
            number_saved_neighbours = len(set(neighbours_original) & set(neighbours_finetuned))

            scores.append(number_saved_neighbours)
        scores = np.array(scores)
        np.save(f'scores_{self.num_neighbours}_{self.metric}_{self.fixed_budget}.npy', np.array(scores))
        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the lowest scores
        sorting = np.argsort(scores)
        if not self.fixed_budget:
            mask = (scores < self.nn_thr)[sorting]
        else:
            mask = np.ones_like(scores, dtype=bool)
        
        return np.array(unlabeled_ids)[sorting][mask][:budget].tolist()

