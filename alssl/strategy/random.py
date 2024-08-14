import numpy as np
from torch import nn

from ..data.base import ALDataModule
from .base import BaseStrategy


class RandomStrategy(BaseStrategy):
    """
    Random Strategy for Active Learning.

    This strategy randomly selects a specified number of unlabeled data points
    from the dataset to be labeled next, based on the given budget.
    """

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):
        return np.random.choice(
            dataset.get_unlabeled_ids(), size=budget, replace=False
        ).tolist()
