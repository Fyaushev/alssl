import numpy as np
import torch
from scipy.special import kl_div, softmax
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..utils import predict
from .base import BaseStrategy


class CALStrategy(BaseStrategy):
    '''
    Contrastive Active Learning (CAL)
    code adapted from https://github.com/mourga/contrastive-active-learning
    '''

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):

        ys_unlabeled, ys_unlabeled_pred, embeddings_unlabeled = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        ys_train, ys_train_pred, embeddings_train = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc='train')
        
        # fit KN classifier on train embeddings
        neigh = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        neigh.fit(X=embeddings_train, y=ys_train)

        scores = []
        for embedding_unlabeled, y_unlabeled_pred in tqdm(zip(embeddings_unlabeled, ys_unlabeled_pred), 
                                                          total=len(ys_unlabeled), desc="Finding neighbours for every unlabeled data point"):
            # for unlabeled point, find indices of closest neighbours in labeled set
            neighbours = neigh.kneighbors(X=embedding_unlabeled[None], return_distance=False).flatten()

            # calculate output probabilities for labeled examples and the unlabeled point
            neigh_prob = softmax(ys_train_pred[neighbours], axis=-1)            
            candidate_prob = softmax(y_unlabeled_pred, axis=-1)
            
            # compute the Kullback–Leibler divergence (KL) between the output probabilities of the unlabeled point and its closest neighbours in labeled set
            kl_scores = np.array([
                np.sum(kl_div(candidate_prob, n), axis=-1) for n in neigh_prob
            ])
            
            # to obtain a score for a candidate, take the average of all divergence scores
            scores.append(kl_scores.mean())

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the highest scores
        return np.array(unlabeled_ids)[np.argsort(scores)[::-1]][:budget].tolist()
