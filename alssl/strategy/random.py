import numpy as np
from torch import nn

from ..data.base import ALDataModule
from .base import BaseStrategy


class RandomStrategy(BaseStrategy):
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int):
        train_ids = dataset.train_ids
        val_ids = dataset.val_ids
        all_ids = dataset.all_ids

        return np.random.choice(
            list(set(all_ids) - set(train_ids + val_ids)), size=budget, replace=False
        ).tolist()
