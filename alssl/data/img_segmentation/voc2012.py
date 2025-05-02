import os
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

resize_s = (490,490)

transform_train = transforms.Compose(
    [
        # transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize(resize_s, antialias=True),
        # transforms.RandomRotation(10),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        # transforms.ToTensor(), 
        # transforms.Resize(resize_s, antialias=True),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def get_num_classes():
    return 21


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


class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root: str = "./data",
        year: str = "2012",
        image_set: str = "train",
        download: bool = True,
        transform: Optional[A.Compose] = None,
        use_index_label: bool = True,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
        )
        self.n_classes = 21
        self.transform = transform
        self.use_index_label = use_index_label
        self.targets = np.ones(len(self.images))

    @staticmethod
    def _convert_to_segmentation_mask(
        mask: np.ndarray, use_index_label: bool = True
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)),
            dtype=np.float32,
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)

        if use_index_label:
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)
        return segmentation_mask

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path = self.images[index]
        mask_path = image_path.replace('JPEGImages', 'SegmentationObject').replace('jpg','png')
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = cv2.resize(np.array(mask), resize_s)
        mask = self._convert_to_segmentation_mask(mask, self.use_index_label)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # image = np.moveaxis(image, -1, 0) / 255

        image = cv2.resize(np.array(image), resize_s) / 255
        image = (image - np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.24, 0.225])
        image = image.transpose(2, 0, 1)
        return torch.tensor(np.array(image).astype(np.float32)), mask


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/voc2012")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = PascalVOCDataset(root=data_path, image_set=subset if subset == "train" else "val")
    return ds, transform

