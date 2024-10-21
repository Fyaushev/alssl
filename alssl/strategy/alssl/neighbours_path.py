from collections import deque
import heapq
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

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Convert the unweighted graph to a weighted graph
def create_weighted_graph(unweighted_graph, embeddings):
    weighted_graph = {}
    
    for node, neighbors in enumerate(unweighted_graph):
        weighted_graph[node] = []
        for neighbor in neighbors:
            dist = euclidean_distance(embeddings[node], embeddings[neighbor])
            weighted_graph[node].append((neighbor, dist))
    
    return weighted_graph

# Dijkstra's algorithm to find the shortest path in the weighted graph
def dijkstra_shortest_path(weighted_graph, start, goal):
    queue = [(0, start)]  # Priority queue: (distance, node)
    distances = {start: 0}  # Distance from start to each node
    previous_nodes = {start: None}  # To store the path
    
    while queue:
        current_distance, node = heapq.heappop(queue)
        
        if node == goal:
            # Reconstruct the shortest path from start to goal
            path = []
            while node is not None:
                path.append(node)
                node = previous_nodes[node]
            return distances[goal]  # Return distance
        
        for neighbor, weight in weighted_graph[node]:
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = node
                heapq.heappush(queue, (distance, neighbor))
    
    return None  # No path found



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
        
        weighted_graph = create_weighted_graph(neighbours_finetuned_inds, embeddings_finetuned)

        scores = []
        point_i = 0
        for neighbours_original, neighbours_finetuned in tqdm(zip(neighbours_original_inds, neighbours_finetuned_inds),
                                                              total=neighbours_finetuned_inds.shape[0], 
                                                              desc="Calculating nn distances for neighbours"):
            
            # for each point i obtain a set of points that take into account the neighbors from E0 and E1.
            # original_notpreserved_neighbours = np.array(list(set(neighbours_original) - set(neighbours_finetuned)))

            path_sizes = [dijkstra_shortest_path(weighted_graph,  point_i, lost_neighbour) for lost_neighbour in neighbours_original]
            path_sizes = [i for i in path_sizes if i != None] # filter points where the path was not found
            if len(path_sizes):
                scores.append(np.median(path_sizes))
            else:
                scores.append(0)
            point_i += 1

        unlabeled_ids = dataset.get_unlabeled_ids()
        # need to take the highest scores
        if self.take_lowest:
            return np.array(unlabeled_ids)[np.argsort(scores)][:budget].tolist()
        else:
            return np.array(unlabeled_ids)[np.argsort(scores)][-budget:].tolist()