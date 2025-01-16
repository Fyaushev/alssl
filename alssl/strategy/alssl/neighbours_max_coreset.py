import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ...model.base import BaseALModel
from ..base import BaseStrategy
from ..utils import predict
from .utils import (get_current_iteration, get_previous_interation_state_dict,
                    load_or_compute)


def get_neighbours(X, num_neighbours, metric):
    neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric, n_jobs=-1)
    neigh.fit(X=X)
    return neigh.kneighbors(X=X, return_distance=False) #[:, 1:] include the point

def furthest_first(X, nn, X_set, n, scores):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        # dist_ctr_nn = dist_ctr.copy()
        # for i in range(dist_ctr.shape[0]):
        #     dist_ctr_nn[i] = dist_ctr[nn[i]].mean()

        # min_dist = np.amin(dist_ctr_nn, axis=1)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for _ in tqdm(range(n), desc="Coreset distances calculation"):
        idx_distant = min_dist.argmax()
        ids_neighbors = nn[idx_distant]
        best_neighbor = ids_neighbors[np.argmin(scores[ids_neighbors])]
        print('best_neighbor', best_neighbor)
        idxs.append(best_neighbor)
        dist_new_ctr = pairwise_distances(X, X[[best_neighbor], :])
        # dist_new_ctr_nn = dist_new_ctr.copy()
        # for i in range(dist_new_ctr.shape[0]):
        #     dist_new_ctr_nn[i] = dist_new_ctr[nn[i]].mean()
        
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return np.array(idxs)


class FFNNMaxStrategy(BaseStrategy):
    def __init__(self, num_neighbours, metric='cosine', neigh_base='embeddings', add_nn=False, comb_score=False):
        self.num_neighbours = num_neighbours
        self.metric = metric

        assert neigh_base in ['embeddings', 'proba'], 'Invalid neigh_base argument, needs to be embeddings or proba'
        self.neigh_base = neigh_base
        self.add_nn = add_nn
        self.comb_score = comb_score
    
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, iter_n: int):

        prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        
        _, y_preds_unlabeled_original, embeddings_unlabeled_original = predict(
            prev_model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        _, y_preds_train_original, embeddings_train_original = predict(
            prev_model,
            dataset.train_dataloader(), 
            scoring="none", desc='train')
        
        _, y_preds_unlabeled_finetuned, embeddings_unlabeled_finetuned = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        _, y_preds_train_finetuned, embeddings_train_finetuned = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc='train')
        
        proba_unlabeled_original = softmax(y_preds_unlabeled_original, 1)
        proba_unlabeled_finetuned = softmax(y_preds_unlabeled_finetuned, 1)
        proba_train_original = softmax(y_preds_train_original, 1)
        proba_train_finetuned = softmax(y_preds_train_finetuned, 1)

        if self.neigh_base == 'embeddings':
            X_orig, X_fine = embeddings_unlabeled_original, embeddings_unlabeled_finetuned
            X_set_orig, X_set_fine = embeddings_train_original, embeddings_train_finetuned
        elif self.neigh_base == 'proba':
            X_orig, X_fine = proba_unlabeled_original, proba_unlabeled_finetuned
            X_set_orig, X_set_fine = proba_train_original, proba_train_finetuned

        nn = get_neighbours(X_fine, self.num_neighbours, self.metric)

        if self.add_nn:
            nn_orig = get_neighbours(X_orig, self.num_neighbours, self.metric)
            scores = np.array([
                len(set(orig) & set(finetuned))
                for orig, finetuned in tqdm(
                    zip(nn_orig, nn),
                    total=len(nn),
                    desc="Calculating neighbor intersections",
                )
            ])

            if self.comb_score:
                mean_of_nn_scores = []
                for orig_nns in nn_orig:
                    mean_of_nn_scores.append(scores[orig_nns].mean())
                mean_of_nn_scores = np.array(mean_of_nn_scores)

                scores = scores / (mean_of_nn_scores + 1e-10)
        else:
            scores = np.zeros(len(nn))

        chosen_idxs = furthest_first(X_fine, nn, X_set_fine, budget, scores)
        
        unlabeled_ids = dataset.get_unlabeled_ids()

        return np.array(unlabeled_ids)[chosen_idxs.astype(int)].tolist()
