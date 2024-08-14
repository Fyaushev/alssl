from typing import Callable, Literal, Optional

import numpy as np
import torch
from tqdm import tqdm


def move_to_np(tensor: torch.Tensor):
    return tensor.cpu().numpy()

def np_append(a, to):
    return np.concatenate((to, a), axis=0)


def predict(
        model, 
        dataloader, 
        scoring: Literal["common", "individual", "none"]="none", 
        scoring_function:Optional[Callable]=None, 
        device='cuda',
        desc: str = ''
    ):
    '''
    Make prediction from a pytorch model (logits)
    '''
    model.to(device).eval()

    if scoring == "individual":
        scores = np.array([])
    else:
        ys, y_preds, all_embeddings = np.array([]), np.array([]), np.array([])
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, total=len(dataloader), desc=f'strategy prediction {desc}:'):
            x, y = x.to(device), y.to(device)
            
            y_pred, embeddings = map(move_to_np, model(x))
            
            if scoring == "individual":
                score = scoring_function(y, y_pred, embeddings)
                scores = np_append(score, scores)
            else:
                ys = np_append(move_to_np(y), ys)
                y_preds = np_append(y_pred, y_preds) if y_preds.size else y_pred
                all_embeddings = np_append(embeddings, all_embeddings) if all_embeddings.size else embeddings
    
    if scoring == "common":
        scores = scoring_function(ys, y_preds, all_embeddings)

    if scoring == "none":
        return ys, y_preds, all_embeddings

    return scores