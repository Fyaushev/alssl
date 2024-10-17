from collections import deque
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch import nn
from tqdm import tqdm

from ...data.base import ALDataModule
from ...model.base import BaseALModel
from ..base import BaseStrategy
from .utils import (get_current_iteration, get_neighbours,
                    get_previous_interation_state_dict,
                    get_previous_iteration_dir)


def bfs_shortest_path(graph, start, goal):
    # Keep track of visited nodes and the path taken
    visited = set()
    queue = deque([[start]])
    
    if start == goal:
        return 0  # The distance between the same embedding is 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node not in visited:
            neighbors = graph[node]
            
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                
                if neighbor == goal:
                    return len(new_path) - 1  # Return the number of edges (distance)
            
            visited.add(node)

    return -1  # No path found


def prob_high_dim(sigma, dist_row, dist, rho):
    """
    For each row of Euclidean distance matrix (dist_row) compute
    probability in high dimensions (1D array)
    """
    d = dist[dist_row] - rho[dist_row]
    d[d < 0] = 0
    return np.exp(- d / sigma)

def k(prob):
    """
    Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
    """
    return np.power(2, np.sum(prob))

def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Solve equation k_of_sigma(sigma) = fixed_k 
    with respect to sigma by the binary search algorithm
    """
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        if k_of_sigma(approx_sigma) < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
            break
    return approx_sigma

def construct_graph(X_train, N_NEIGHBOR, metric='minkowski'):
    # get martix of squared pairwise Euclidean distances for the initial high-dimensions set
    if metric=='minkowski':
        dist = np.square(euclidean_distances(X_train, X_train))
    elif metric=='cosine':
        dist = np.square(cosine_distances(X_train, X_train))
    else:
        raise Exception(f"selected metric {metric} is not implemented")

    # distance to the nearest neighbour for each point
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])] # [1] because [0] will always be zero

    # get number of objects
    n = X_train.shape[0]

    # build weighted oriented graph
    prob = np.zeros((n,n))
    for dist_row in range(n):
        func = lambda sigma: k(prob_high_dim(sigma, dist_row, dist, rho))
        binary_search_result = sigma_binary_search(func, N_NEIGHBOR)
        prob[dist_row] = prob_high_dim(binary_search_result, dist_row, dist, rho)

    # apply the symmetry condition 
    P = (prob + np.transpose(prob)) / 2
    return P


class NeighboursPathStrategy(BaseStrategy):
    '''
    Алгоритм такой:
    1) находим ближайших соседей для точки i в E0 (nn-0-i) и для E1 (nn-1-i)
    2) находим соседей в nn-0-i, которые отсутствуют в nn-1-i
    3) для каждого такого соседа находим его расстояние до точки i в E1 как минимальное расстояние в графе случайных соседей
    '''

    def __init__(self, num_neighbours: int, metric: str='minkowski', take_lowest: bool = False, num_neighbours_umap: Optional[int]=None):
        self.num_neighbours = num_neighbours
        self.metric = metric
        self.take_lowest = take_lowest
        self.num_neighbours_umap = num_neighbours_umap if num_neighbours_umap is not None else num_neighbours * 2

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel):

        previous_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        # load weights from previous iteration if available
        if get_current_iteration():
            previous_model.load_state_dict(get_previous_interation_state_dict())

        embeddings_original, neighbours_original_inds = get_neighbours(previous_model, dataset, desc="original", num_neighbours=self.num_neighbours, metric=self.metric)

        # generate neighbours for current iteration and save for later
        embeddings_finetuned, neighbours_finetuned_inds = get_neighbours(model, dataset, desc="finetuned", num_neighbours=self.num_neighbours, metric=self.metric)
        np.save('neighbours_inds_original.npy', neighbours_original_inds)
        np.save('neighbours_inds.npy', neighbours_finetuned_inds)
        np.save('embeddings.npy', embeddings_finetuned)

        scores = []
        point_i = 0
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds),
                                                              total=neighbours_finetuned_inds.shape[0], 
                                                              desc="Calculating nn distances for neighbours"):
            
            # for each point i obtain a set of points that take into account the neighbors from E0 and E1.
            original_notpreserved_neighbours = np.array(list(set(neighbours_original) - set(neighbours_finetuned)))

            path_sizes = [bfs_shortest_path(embeddings_finetuned, point_i, lost_neighbour) for lost_neighbour in original_notpreserved_neighbours]
            if len(path_sizes):
                scores.append(np.mean(path_sizes))
            else:
                scores.append(np.mean(path_sizes))
            point_i += 1

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the highest scores
        if self.take_lowest:
            return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()
        else:
            return np.array(unlabeled_ids)[np.argsort(scores)][-budget:].tolist()