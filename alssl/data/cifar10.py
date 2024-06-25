from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_PATH = Path("/shared/projects/active_learning/data/cifar10")

dino_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def get_data_loaders(
    val_size: int | float,
    *,
    batch_size: int = 128,
    random_seed: int = 42,
    num_workers: int = 4,
    transform=dino_transform,
) -> Iterable[DataLoader]:
    torch.manual_seed(random_seed)

    train_ds = ImageFolder(DATA_PATH / "train", transform=transform)
    test_ds = ImageFolder(DATA_PATH / "test", transform=transform)

    if isinstance(val_size, float) and 0 <= val_size <= 1:
        val_size = round(len(train_ds) * val_size)

    train_size = len(train_ds) - int(val_size)
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size, num_workers=num_workers)

    return train_dl, val_dl, test_dl
