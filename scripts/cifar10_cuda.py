import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from alssl.data.cifar10 import get_dataset

warnings.filterwarnings("ignore")

DATA_PATH = Path("/shared/projects/active_learning/data/cifar10")

if __name__ == "__main__":
    train_ds = get_dataset("train")
    train_dl = DataLoader(train_ds, batch_size=50, shuffle=False, num_workers=8)

    device = "cuda"
    save_root = Path("/shared/projects/active_learning/embeddings/cifar10/")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    for i, batch in tqdm(enumerate(train_dl)):
        images, lbls = batch
        output = model(images.to(device))
        embeddings = output.pooler_output.cpu().detach().numpy()
        lbls = lbls.cpu().detach().numpy()
        assert len(np.unique(lbls)) == 1, np.unique(lbls)
        np.save(
            save_root / f"embeddings_train_{lbls[0]}_{i}.npy",
            embeddings,
            allow_pickle=False,
        )
        torch.cuda.empty_cache()
