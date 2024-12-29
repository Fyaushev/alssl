from abc import ABC, abstractmethod

import numpy as np
from torch import nn

from ..data.base import ALDataModule
from ..strategy.utils import predict


class BaseColdStart(ABC):
    @abstractmethod
    def select_ids(self, model: nn.Module, dataset: ALDataModule, **kwargs) -> list:
        pass