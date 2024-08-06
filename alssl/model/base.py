from abc import ABC, abstractmethod

import lightning as L
import torch


class BaseALModel(ABC):
    @abstractmethod
    def get_lightning_module(self) -> L.LightningModule:
        pass

    # def predict(self, batch):
    #     images, masks = batch
    #     logits = self(images)