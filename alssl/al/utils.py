import os
import random
from pathlib import Path
from typing import Any, Literal, Union

import lightning as L
import numpy as np
import torch


def efficient_chdir(root: Union[Path, str]):
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)


def fix_seed(seed=0xBadCafe):
    """Lightning's `seed_everything` with addition `torch.backends` configurations"""
    seed = L.seed_everything(seed=seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    return seed, np.random.default_rng(seed=seed)


def last_checkpoint(root: Union[Path, str]) -> Union[Path, Literal["last"]]:
    """
    Load most fresh last.ckpt file based on time.
    Parameters
    ----------
    root: Union[Path, str]
        Path to folder, where last.ckpt or its symbolic link supposed to be.
    Returns
    -------
    checkpoint_path: Union[Path, str]
        If last.ckpt exists - returns Path to it. Otherwise, returns 'last'.
    """
    checkpoints = []
    for p in Path(root).rglob("*"):
        if p.is_symlink():
            p = p.resolve(strict=False)
        if p.suffix == ".ckpt" and p.stem != 'last':
            checkpoints.append(p)
    return max(checkpoints, key=lambda t: os.stat(t).st_mtime, default=None)

def get_checkpoint(root, zero_iteration_dir, iteration, num_epochs):
    curr_dir = zero_iteration_dir if iteration == 0 else root / f"iter_{iteration}"
    checkpoint_path = last_checkpoint(curr_dir)
    is_fully_trained = checkpoint_path is not None and checkpoint_path.stem.startswith(f'epoch={num_epochs - 1}')
    if iteration > 0 and checkpoint_path is None:
        checkpoint_path = last_checkpoint(root / f"iter_{iteration - 1}")
    return checkpoint_path, is_fully_trained