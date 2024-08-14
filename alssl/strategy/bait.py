import pdb
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy, rv_discrete
from sklearn.metrics import pairwise_distances
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..utils import predict
from .base import BaseStrategy


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

class BAITStrategy(BaseStrategy):
    '''
    Batch Active learning by Diverse Gradient Embeddings (BADGE)
    code adapted from https://github.com/JordanAsh/badge
    assumes cross-entropy loss
    '''

    def __init__(self, num_classes: int, prefilter_beta: Optional[int]=10):
        self.num_classes = num_classes
        self.prefilter_beta = prefilter_beta

    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, _):

        ys_unlabeled, ys_unlabeled_pred, embeddings_unlabeled = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        # get fisher embedding 
        embDim = embeddings_unlabeled[0].shape[0]
        fisher_embedding = np.zeros((len(ys_unlabeled), embDim * self.num_classes))
        entropy_scores = []
        for ind in tqdm(range(self.num_classes), desc='Calculating gradient embeddings per class:'):
            for y_unlabeled_pred, embedding_unlabeled in zip(ys_unlabeled_pred, embeddings_unlabeled):
                prob = softmax(y_unlabeled_pred, axis=-1)
                maxInd = np.argmax(prob, axis=-1)

                # calculate entropy to pre-filter embeddings
                if self.prefilter_beta is not None:
                    # calculate inverse entropy: min is worst
                    entropy_scores.append( - entropy(prob))

                for c in range(self.num_classes):
                    if c == maxInd:
                        fisher_embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(embedding_unlabeled) * (1 - prob[c])
                    else:
                        fisher_embedding[ind][embDim * c : embDim * (c+1)] = deepcopy(embedding_unlabeled) * (-1 * prob[c])
                fisher_embedding[ind] *= np.sqrt(prob[ind])

        if self.prefilter_beta is not None:
            top_k_entropy = np.argsort(entropy_scores)[:budget * self.prefilter_beta]
            fisher_embedding = fisher_embedding[top_k_entropy, :]
        
        chosen_idxs = init_centers(fisher_embedding, budget)
        unlabeled_ids = dataset.get_unlabeled_ids()
        
        return np.array(unlabeled_ids)[chosen_idxs.astype(int)].tolist()
