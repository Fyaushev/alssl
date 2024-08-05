from abc import ABC, abstractmethod

from torch import nn

from ..data.base import ALDataModule


class BaseStrategy(ABC):
    @abstractmethod
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int) -> list:
        pass
