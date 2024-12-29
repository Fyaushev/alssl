from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from torch import nn

from ...data.base import ALDataModule
from ...strategy.alssl.utils import load_or_compute
from ...strategy.utils import predict
from ..base import BaseColdStart
from .turtle_backend import get_labels, train_turtle


class TurtleColdStart(BaseColdStart):
    """
    Random sampling of initial ids
    """
    def __init__(self, initial_train_size: int, random_seed: int, num_classes: int, samples_per_class: int = 3, turtle_lr: float = .01):
        self.initial_train_size = initial_train_size
        self.random_seed = random_seed
        self.num_classes = num_classes

        self.samples_per_class = samples_per_class
        assert samples_per_class > 0, f"Number of samples per class should be positive. Current: {samples_per_class}"
        self.turtle_lr = turtle_lr

    def select_ids(self, model: nn.Module, dataset: ALDataModule, **kwargs) -> list:
        all_ids = np.array(dataset.get_unlabeled_ids())
        cluster_labels = self.run_turtle(model, dataset)

        train_ids = []

        np.random.seed(self.random_seed)
        for cluster in np.unique(cluster_labels):
            cluster_inds = np.argwhere(cluster_labels == cluster).ravel()

            selected_cluster_inds = np.random.choice(cluster_inds, self.samples_per_class, replace=False)
            
            train_ids.extend(all_ids[selected_cluster_inds])

        return train_ids
    
    def run_turtle(self, model: nn.Module, dataset: ALDataModule):
    
        def _predict_unlabeled():
            _, _, embeddings = predict(
                model,
                dataset.unlabeled_dataloader(), 
                scoring="none", desc="KMeans coldstart")
            return embeddings
        
        embeddings = load_or_compute(["embeddings_unlabeled.npy"], _predict_unlabeled)

        turtle = train_turtle(embeddings, self.num_classes, self.turtle_lr)
        turtle_labels = get_labels(embeddings, turtle)
        return turtle_labels
    
    