from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torchvision
from dpipe.io import save
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def get_num_classes():
    return 100

def get_dataset(subset="train", data_path=Path("/shared/projects/active_learning/cifar100")):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    return torchvision.datasets.CIFAR100(
        root=data_path, train=subset == "train", download=True, transform=transform
    )


# class CIFAR100LightningDataModule(L.LightningDataModule):
#     def __init__(
#         self,
#         batch_size,
#         *,
#         train_ids=None,
#         val_ids=None,
#         trains_size=1000,
#         val_size=200,
#         train_transform=dino_transform,
#         test_transform=dino_transform,
#         random_state=0,
#         num_workers=8,
#     ):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         train_dataset = get_dataset(subset="train", transform=train_transform)
#         test_dataset = get_dataset(subset="test", transform=test_transform)

#         if train_ids is None and val_ids is None:
#             train_ids, val_ids = train_test_split(
#                 np.arange(len(train_dataset)),
#                 train_size=trains_size,
#                 test_size=val_size,
#                 random_state=random_state,
#                 stratify=train_dataset.targets,
#             )

#         save(train_ids, "train_ids.json")
#         save(val_ids, "val_ids.json")

#         self.train_dataset = Subset(train_dataset, train_ids)
#         self.val_dataset = Subset(train_dataset, val_ids)

#         self.val_dataset.transform = test_transform
#         self.test_dataset = test_dataset

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             persistent_workers=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             persistent_workers=True,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#         )
