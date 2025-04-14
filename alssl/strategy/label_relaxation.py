import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from .base import BaseStrategy
from .typiclust import calculate_typicality
from .utils import predict


def pdist(A, B, squared = False, eps = 1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A,B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

def get_cluster_matching(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))
    return match

class LabelRelaxStrategy(BaseStrategy):
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 50
    SIGMA = 1
    DELTA = 1

    def __init__(self, num_classes: int, cluster_curr: bool = False, mode: str = 'both', cluster_mode: str = 'both', inverse_score: bool = False, inverse_cluster_score: bool = False, source='e0'):
        self.num_classes = num_classes
        self.cluster_curr = cluster_curr
        self.inverse_score = inverse_score
        self.inverse_cluster_score = inverse_cluster_score # inverse will sort pandas df from max cluster score to min
        self.source = source
        self.mode = mode
        self.cluster_mode = cluster_mode

        assert mode in ['both', 'pull', 'push', 'typi', 'combined_pull', 'combined_push', 'proba_', 'adj_', 'emb_norm'], f'Mode is {mode}. Please choose both, pull or push.'
        assert cluster_mode in ['both', 'pull', 'push', 'size'], f'Cluster mode is {mode}. Please choose both, pull or push.'
        self.pull_alpha, self.push_alpha = 1, 1
        if mode == 'pull':
            self.push_alpha = 0
        elif mode == 'push':
            self.pull_alpha = 0
        self.cluster_pull_alpha, self.cluster_push_alpha = 1, 1
        if cluster_mode == 'pull':
            self.cluster_push_alpha = -1 #0
        elif cluster_mode == 'push':
            self.cluster_pull_alpha = 0

    
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.concatenate([dataset.train_ids, np.array(dataset.get_unlabeled_ids())])
        num_clusters = min(len(dataset.train_ids) + self.num_classes, self.MAX_NUM_CLUSTERS)

        m = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        _, _, features_unlabeled_e0 = predict(
            m,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="TypiClust strategy (unlabeled)")
        
        _, _, features_train_e0 = predict(
            m,
            dataset.train_dataloader(), 
            scoring="none", desc="TypiClust strategy (train)")
        
        _, _, features_unlabeled_e1 = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="TypiClust strategy (unlabeled)")
        
        _, _, features_train_e1 = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc="TypiClust strategy (train)")
        
        features_e0 = np.concatenate([features_train_e0, features_unlabeled_e0])
        features_e1 = np.concatenate([features_train_e1, features_unlabeled_e1])

        labels_e0 = kmeans(features_e0, num_clusters=num_clusters)
        labels_e1 = kmeans(features_e1, num_clusters=num_clusters)
        labels = labels_e1 if self.cluster_curr else labels_e0

        # labels_e0 = get_cluster_matching(labels_e0, labels_e1)

        existing_indices = np.arange(len(dataset.train_ids))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)

        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))

        if self.source == 'e0':
            cluster_losses = calc_loss(features_e1, features_e0, torch.tensor(labels))
        else:
            cluster_losses = calc_loss(features_e0, features_e1, torch.tensor(labels))

        clusters_loss_mean = {}
        for i, cluster in enumerate(cluster_ids):
            indices = (labels == cluster).nonzero()[0]
            sample_losses = cluster_losses[cluster]
            pull_losses, push_losses = [], []
            for sample_idx, (pull_loss, push_loss) in sample_losses.items():
                pull_losses.append(pull_loss)
                push_losses.append(push_loss)
            clusters_loss_mean[cluster] = np.mean(pull_losses + pull_losses)
        loss_means = list(clusters_loss_mean.values())
        for i, cluster in enumerate(cluster_ids):
            clusters_loss_mean[cluster] = (clusters_loss_mean[cluster]  - np.min(loss_means)) / (np.max(loss_means) - np.min(loss_means) + 1e-8)

        cluster_scores = []
        cluster_selected_inds = {}
        for i, cluster in enumerate(cluster_ids):
            indices = (labels == cluster).nonzero()[0]
            
            sample_losses = cluster_losses[cluster]
            pull_losses, push_losses = [], []
            for sample_idx, (pull_loss, push_loss) in sample_losses.items():
                pull_losses.append(pull_loss)
                push_losses.append(push_loss)
            pull_losses, push_losses = np.array(pull_losses), np.array(push_losses)
            
            pull_losses_norm = (pull_losses - np.min(pull_losses)) / (np.max(pull_losses) - np.min(pull_losses) + 1e-8)
            push_losses_norm = (push_losses - np.min(push_losses)) / (np.max(push_losses) - np.min(push_losses) + 1e-8)

            cluster_typiscores = calculate_typicality(features_e0[indices], min(self.K_NN, len(indices) // 2))
            cluster_typiscores = (cluster_typiscores - np.min(cluster_typiscores)) / (np.max(cluster_typiscores) - np.min(cluster_typiscores) + 1e-8)

            if 'combined' in self.mode:
                score = self.pull_alpha * pull_losses + self.push_alpha * push_losses
                # losses = pull_losses if 'pull' in self.mode else push_losses
                # losses_mask = losses < np.quantile(losses, .1)
                # typi_mask = cluster_typiscores > np.quantile(cluster_typiscores, .9)
                # score = pull_losses if 'pull' in self.mode else push_losses
                score = cluster_typiscores - score * .5
                # score *= typi_mask.astype(int)
            elif 'proba' in self.mode:
                is_poor = np.random.binomial(1, clusters_loss_mean[cluster])
                if is_poor:
                    print('poor')
                    # take most stable point
                    score = cluster_typiscores - (self.pull_alpha * pull_losses + self.push_alpha * push_losses) * .5
                else:
                    print('cool')
                    # take typical point
                    score = cluster_typiscores
            
            elif 'adj' in self.mode:
                score = cluster_typiscores - (self.pull_alpha * pull_losses + self.push_alpha * push_losses) * clusters_loss_mean[cluster] 
            elif self.mode == 'emb_norm':
                score = np.linalg.norm(features_e0[indices], -1)   
            elif self.mode !='typi':
                score = self.pull_alpha * pull_losses + self.push_alpha * push_losses
            else:
                score = cluster_typiscores

            cluster_score = self.cluster_pull_alpha * pull_losses + self.cluster_push_alpha * push_losses
            top_ind = indices[score.argmax()] if not self.inverse_score else indices[score.argmin()]
            top_score = cluster_score.mean() if not self.inverse_cluster_score else -1 * cluster_score.mean()
            if self.cluster_mode == 'size':
                top_score = -cluster_sizes[i]
            cluster_scores.append(top_score)
            cluster_selected_inds[cluster] = top_ind


        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'cluster_scores': cluster_scores})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df[clusters_df.existing_count == 0]
        clusters_df = clusters_df.sort_values(['cluster_scores'])

        labels[existing_indices] = -1

        selected = []

        for i in range(budget):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            idx = cluster_selected_inds[cluster]

            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        return all_ids[selected]


def sample_from_clusters(labels, num_samples_per_cluster=50):
    """Samples a fixed number of embeddings from each cluster."""
    sampled_indices = []
    unique_clusters = torch.unique(labels)
    
    for cluster in unique_clusters:
        cluster_indices = (labels == cluster).nonzero(as_tuple=True)[0]
        sample_size = min(len(cluster_indices), num_samples_per_cluster)  # Avoid exceeding cluster size
        sampled_indices.append(cluster_indices[torch.randperm(len(cluster_indices))[:sample_size]])
    
    return torch.cat(sampled_indices)

def calc_loss(t_emb, s_emb, labels, num_samples_per_cluster=50):
    """Calculates per-sample loss while considering the entire dataset via cluster-wise sampling."""
    
    t_emb, s_emb, labels = torch.tensor(t_emb), torch.tensor(s_emb), torch.tensor(labels)
    t_emb = F.normalize(t_emb, p=2, dim=1)
    s_emb = F.normalize(s_emb, p=2, dim=1)

    # Sample embeddings from each cluster
    sampled_indices = sample_from_clusters(labels, num_samples_per_cluster)
    t_emb_sampled = t_emb[sampled_indices]
    s_emb_sampled = s_emb[sampled_indices]

    # Compute pairwise distances: each sample vs all sampled points
    T_dist = pdist(t_emb, t_emb_sampled, False)  # (N, K)
    dist_mean = T_dist.mean(1, keepdim=True)  
    T_dist = T_dist / dist_mean  # Normalize

    with torch.no_grad():
        S_dist = pdist(s_emb, s_emb_sampled, False)  # (N, K)
        P = torch.exp(-S_dist.pow(2) / 1)

    pos_weight = P
    neg_weight = 1 - P

    pull_losses = torch.relu(T_dist).pow(2) * pos_weight
    push_losses = torch.relu(1 - T_dist).pow(2) * neg_weight

    pull_losses = pull_losses.mean(1)  # Average over sampled distances
    push_losses = push_losses.mean(1)  

    # Store per-sample losses grouped by cluster
    cluster_losses = {cluster.item(): {} for cluster in torch.unique(labels)}
    for i, cluster in enumerate(labels):
        cluster_losses[cluster.item()][i] = (pull_losses[i].item(), push_losses[i].item())

    return cluster_losses  # Returns per-sample loss within each cluster
