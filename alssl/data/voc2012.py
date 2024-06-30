import cv2
import numpy as np
from torchvision.datasets import VOCSegmentation
import torch
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

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class VOCDataset(VOCSegmentation):
    def __init__(
        self,
        root="/shared/projects/active_learning/data/voc-2012",
        image_set="train",
        download=False,
        transform=None,
    ):
        super().__init__(
            root=root, image_set=image_set, download=download, transform=transform
        )

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)), dtype=np.float32
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, np.ndarray):
            mask = np.argmax(mask, -1)
        elif isinstance(mask, torch.Tensor):
            mask = torch.argmax(mask, -1)
        else:
            raise ValueError

        return image, mask


class VOCLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_transform,
        test_transform,
        *,
        trains_size=200,
        val_size=10,
        random_state=0,
        num_workers=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_dataset = VOCDataset(image_set="train", transform=train_transform)
        test_dataset = VOCDataset(image_set="val", transform=train_transform)

        train_ids, val_ids = train_test_split(
            np.arange(len(train_dataset)),
            train_size=trains_size,
            test_size=val_size,
            random_state=random_state,
        )

        save(train_ids, "train_ids.json")
        save(val_ids, "val_ids.json")

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
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
