from functools import partial

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from torch import nn

from ..data.base import ALDataModule
from .base import BaseStrategy
from .coreset import furthest_first
from .utils import predict


class CDALStrategy(BaseStrategy):
    def __init__(self, num_classes):
        self.num_classes = num_classes
		
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, *args):
        _, y_preds_unlabeled, _ = predict(
            model, 
            dataset.unlabeled_dataloader(), 
            scoring="none", desc='unlabeled')
        
        _, y_preds_train, _ = predict(
            model, 
            dataset.train_dataloader(), 
            scoring="none", desc='train')
        
        if y_preds_unlabeled.ndim == 4:
			# segmentation
            B, B_tr = y_preds_unlabeled.shape[0], y_preds_train.shape[0]
            chosen_idxs = furthest_first(
                  y_preds_unlabeled.reshape(B, -1), 
                  y_preds_train.reshape(B_tr, -1), 
                  budget, metric=partial(kl, nc=self.num_classes))
        else:
            proba_unlabeled = softmax(y_preds_unlabeled, 1)
            proba_train = softmax(y_preds_train, 1)
            chosen_idxs = furthest_first(proba_unlabeled, proba_train, budget)
        
        unlabeled_ids = dataset.get_unlabeled_ids()

        return np.array(unlabeled_ids)[chosen_idxs.astype(int)].tolist()
    

def kl(ac,bc, nc):
	kl_classes=[]
	ac=np.reshape(ac,(nc,-1))
	bc=np.reshape(bc,(nc,-1))
	for i in range(nc):
		a=ac[i,:]
		b=bc[i,:]
		kl1=a*np.log(a/b)
		kl2=b*np.log(b/a)
		kl= -0.5*(np.sum(kl1)) - 0.5*(np.sum(kl2))
		if(kl == kl and not np.isinf(kl)):
			kl_classes.append(kl)

	if(len(kl_classes) != 0):
		reward_kl=sum(kl_classes)/len(kl_classes)
	else:
		reward_kl = 0
	return abs(reward_kl)