import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn

from ..data.base import ALDataModule
from .base import BaseColdStart


class RandomColdStart(BaseColdStart):
    """
    Random sampling of initial ids
    """
    def __init__(self, initial_train_size: int, random_seed: int, num_classes: int):
        self.initial_train_size = initial_train_size
        self.random_seed = random_seed
        self.num_classes = num_classes

    def select_ids(self, model: nn.Module, dataset: ALDataModule,) -> list:
        ids = dataset.get_unlabeled_ids()
        if len(ids) <= self.initial_train_size:
            return ids
        train_ids, _ = train_test_split(
                    ids,
                    train_size=self.initial_train_size,
                    random_state=self.random_seed,
                    stratify=None,
                )
        return train_ids