
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


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

def construct_graph(X_train, N_NEIGHBOR, metric='cosine'):
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