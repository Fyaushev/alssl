from typing import Callable, Literal, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def move_to_np(tensor: torch.Tensor):
    return tensor.cpu().numpy()

def _init_arrays(num_samples, embedding_size, output_size):
    ys = np.empty((num_samples,))
    y_preds = np.empty((num_samples, output_size))
    all_embeddings = np.empty((num_samples, embedding_size))
    return ys, y_preds, all_embeddings

def predict(
        model, 
        dataloader, 
        scoring: Literal["common", "individual", "none"]="none", 
        scoring_function: Optional[Callable] = None, 
        device='cuda',
        desc: str = ''
    ):
    '''
    Make prediction from a pytorch model (logits)
    '''
    model.to(device).eval()

    if scoring == "individual":
        scores = []

    sample_idx = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f'strategy prediction {desc}:'):
            x, y = x.to(device), y.to(device)
            y_pred, embeddings = map(move_to_np, model(x))

            batch_size = y.shape[0]
            
            if scoring == "individual":
                score = scoring_function(y, y_pred, embeddings)
                scores.append(score)
            else:
                if sample_idx == 0:
                    num_samples = int(batch_size * len(dataloader))
                    embedding_size = embeddings.shape[-1]
                    output_size = y_pred.shape[-1]

                    ys, y_preds, all_embeddings = _init_arrays(num_samples, embedding_size, output_size)

                y_np = move_to_np(y)

                ys[sample_idx:sample_idx + batch_size] = y_np
                y_preds[sample_idx:sample_idx + batch_size, :] = y_pred
                all_embeddings[sample_idx:sample_idx + batch_size, :] = embeddings
                sample_idx += batch_size
    
    if scoring == "individual":
        scores = np.concatenate(scores, axis=0)
        return scores

    if scoring == "common":
        scores = scoring_function(ys, y_preds, all_embeddings)
        return scores
    else:
        return ys[:sample_idx], y_preds[:sample_idx], all_embeddings[:sample_idx]
    

def get_cluster_acc(y_pred, y_true, return_matching=False):
    """
    code from https://github.com/mlbio-epfl/turtle
    
    Calculate clustering accuracy and clustering mean per class accuracy.
    Requires scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        Accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    match = np.array(list(map(lambda i: col_ind[i], y_pred)))

    mean_per_class = [0 for i in range(D)]
    for c in range(D):
        mask = y_true == c
        mean_per_class[c] = np.mean((match[mask] == y_true[mask]))
    mean_per_class_acc = np.mean(mean_per_class)

    if return_matching:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc, match
    else:
        return w[row_ind, col_ind].sum() / y_pred.size, mean_per_class_acc