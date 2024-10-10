
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

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours + 1 # NearestNeighbors outputs point itself as neighbour

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        # get neighbours for previous iteration and save for later
        # if get_current_iteration():
        #     neighbours_original_inds = np.load(get_previous_iteration_dir() / 'neighbours_inds.npy')    
        # else:
        previous_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        # load weights from previous iteration if available
        if get_current_iteration():
            previous_model.load_state_dict(get_previous_interation_state_dict())

        _, neighbours_original_inds = get_neighbours(previous_model, dataset, desc="original", num_neighbours=self.num_neighbours)

        # generate neighbours for current iteration and save for later
        _, neighbours_finetuned_inds = get_neighbours(model, dataset, desc="finetuned", num_neighbours=self.num_neighbours)
        np.save('neighbours_inds.npy', neighbours_finetuned_inds)

        scores = []
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds), 
                                                            total=neighbours_finetuned_inds.shape[0], 
                                                            desc="Finding neighbours intersection for every unlabeled data point"):
            # find number of intersecting neighbours
            number_saved_neighbours = len(set(neighbours_original) & set(neighbours_finetuned))

            scores.append(number_saved_neighbours)

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the lowest scores
        return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()

