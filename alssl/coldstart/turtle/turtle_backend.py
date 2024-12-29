# code from https://github.com/mlbio-epfl/turtle/

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cluster_acc(y_pred, y_true, return_matching=False):
    """
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
    

def train_turtle(embeddings, C, LR: float = .01):
    seed_everything(42)

    embeddings = [embeddings]


    n_tr = embeddings[0].shape[0]
    feature_dims = [Z_train.shape[1] for Z_train in embeddings]
    batch_size = min(1000, n_tr)
    print("Number of training samples:", n_tr)

    # Define task encoder
    task_encoder = [nn.utils.weight_norm(nn.Linear(d, C)).to('cuda') for d in feature_dims] 

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)] # shape of (K, N, C)
        labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, C)
        return labels, label_per_space

    # we use Adam optimizer for faster convergence, other optimziers such as SGD could also work
    optimizer = torch.optim.Adam(sum([list(task_phi.parameters()) for task_phi in task_encoder], []), lr=LR, betas=(0.9, 0.999))

    # Define linear classifiers for the inner loop
    def init_inner():
        W_in = [nn.Linear(d, C).to('cuda') for d in feature_dims] 
        inner_opt = torch.optim.Adam(sum([list(W.parameters()) for W in W_in], []), lr=LR, betas=(0.9, 0.999))

        return W_in, inner_opt

    W_in, inner_opt = init_inner()

    # start training
    iters_bar = tqdm(range(6000))
    for i in iters_bar:
        optimizer.zero_grad()
        # load batch of data
        indices = np.random.choice(n_tr, size=batch_size, replace=False)
        Zs_tr = [torch.from_numpy(Z_train[indices]).to('cuda').float() for Z_train in embeddings]

        labels, label_per_space = task_encoding(Zs_tr)

        # init inner
        if not False: 
            # cold start, re-init every time
            W_in, inner_opt = init_inner()
        # else, warm start, keep previous 

        # inner loop: update linear classifiers
        for idx_inner in range(10):
            inner_opt.zero_grad()
            # stop gradient by "labels.detach()" to perform first-order hypergradient approximation, i.e., Eq. (13) in the paper
            loss = sum([F.cross_entropy(w_in(z_tr), labels.detach()) for w_in, z_tr in zip(W_in, Zs_tr)])
            loss.backward()
            inner_opt.step()

        # update task encoder
        optimizer.zero_grad()
        pred_error = sum([F.cross_entropy(w_in(z_tr).detach(), labels) for w_in, z_tr in zip(W_in, Zs_tr)])

        # entropy regularization 
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])
        
        # final loss, Eq. (12) in the paper
        (pred_error - 10 * entr_reg).backward()
        optimizer.step()

        iters_bar.set_description(f'Training loss {float(pred_error):.3f}, entropy {float(entr_reg):.3f}')

    print(f'Training finished! ')
    print(f'Training loss {float(pred_error):.3f}, entropy {float(entr_reg):.3f}')
    
    return task_encoding

def get_labels(embeddings, task_encoding, device='cuda'):
    labels_all, _ = task_encoding([torch.from_numpy(embeddings).to(device).float(),])
    preds_all = labels_all.argmax(dim=1).detach().cpu().numpy()
    return preds_all
