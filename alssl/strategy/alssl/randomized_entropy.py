import numpy as np
from scipy.special import softmax
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ..base import BaseStrategy
from .utils import get_neighbours


def entropy(pred):
    """Calculate entropy: max is worst."""
    proba = softmax(pred, axis=1)
    return -np.sum(proba * np.log(proba), axis=1)


class RandEntropyStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, al_model, iter_n: int):

        e1, neighbors, pred = get_neighbours(
                model, dataset, "finetuned", num_neighbours=self.num_neighbours, metric=self.metric, return_predicts=True
        )
        entropy_scores = entropy(pred)
        
        rand_selected = np.random.choice(range(neighbors.shape[0]), size=budget, replace=False)

        strategy_selected = []

        for idx in tqdm(rand_selected):
            selected_neighbors = [neigh for neigh in neighbors[idx] if neigh not in strategy_selected]
            selected_neighbor = selected_neighbors[np.argmax(entropy_scores[selected_neighbors])]
            strategy_selected.append(selected_neighbor)

        unlabeled_ids = dataset.get_unlabeled_ids()

        return np.array(unlabeled_ids)[strategy_selected].tolist()