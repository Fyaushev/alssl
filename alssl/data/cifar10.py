from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dpipe.io import save

DATA_PATH = Path("/shared/projects/active_learning/data/cifar10")

dino_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def get_dataset(subset="train", transform=dino_transform):
    assert subset in ["train", "test"]
    return ImageFolder(DATA_PATH / subset, transform=transform)


def get_data_loaders(
    val_size: int | float,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    transform=dino_transform,
    shuffle=False,
) -> Iterable[DataLoader]:

    train_ds = ImageFolder(DATA_PATH / "train", transform=transform)
    test_ds = ImageFolder(DATA_PATH / "test", transform=transform)

    if isinstance(val_size, float) and 0 <= val_size <= 1:
        val_size = round(len(train_ds) * val_size)

    train_size = len(train_ds) - int(val_size)
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_dl = DataLoader(
        train_ds, batch_size, num_workers=num_workers, shuffle=shuffle
    )
    val_dl = DataLoader(val_ds, batch_size, num_workers=num_workers, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size, num_workers=num_workers, shuffle=False)

    return train_dl, val_dl, test_dl


class CIFAR10LightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        *,
        trains_size=950,
        val_size=50,
        train_transform=dino_transform,
        test_transform=dino_transform,
        random_state=0,
        num_workers=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_dataset = ImageFolder(DATA_PATH / "train", transform=train_transform)
        test_dataset = ImageFolder(DATA_PATH / "test", transform=test_transform)

        train_ids, val_ids = train_test_split(
            np.arange(len(train_dataset)),
            train_size=trains_size,
            test_size=val_size,
            random_state=random_state,
            stratify=train_dataset.targets,
        )

        save(train_ids, "train_ids.json")
        save(val_ids, "val_ids_ids.json")

        self.train_dataset = Subset(train_dataset, train_ids)
        self.val_dataset = Subset(train_dataset, val_ids)

        self.val_dataset.transform = test_transform
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
