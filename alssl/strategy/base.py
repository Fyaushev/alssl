from abc import ABC, abstractmethod

from torch import nn

from ..data.base import ALDataModule
from ..model.base import BaseALModel


class BaseStrategy(ABC):
    @abstractmethod
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel) -> list:
        pass
