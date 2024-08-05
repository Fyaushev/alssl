from abc import ABC, abstractmethod

import lightning as L


class BaseALModel(ABC):
    @abstractmethod
    def get_lightning_module(self) -> L.LightningModule:
        pass
