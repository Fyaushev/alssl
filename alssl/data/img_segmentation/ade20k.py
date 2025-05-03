import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

resize_s = (280,280)

transform_train = transforms.Compose(
    []
)

transform_train = transforms.Compose(
    [
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
    return 150

class ADE20kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[A.Compose] = None,
    ):
        self.root = root
        self.split = split
        self.n_classes = 150
        self.transform = transform

        root = os.path.join(root, "ADEChallengeData2016")
        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        # Check if the dataset is already downloaded
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            self.download_and_extract_dataset()

        self.image_files = os.listdir(self.images_dir)
        self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=(-10, 10)),
            ])
        self.targets = np.ones(len(self.image_files))

    def download_and_extract_dataset(self) -> None:
        dataset_url = (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )
        zip_path = os.path.join(self.root, "ADEChallengeData2016.zip")
        os.makedirs(self.root, exist_ok=True)

        logging.info("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)

        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        logging.info("Dataset extracted!")
        os.remove(zip_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(np.array(image), resize_s) / 255
        image = (image - np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.24, 0.225])

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:, :, 0]
        mask = cv2.resize(np.array(mask), resize_s, cv2.INTER_NEAREST)

        # if self.split == 'training':
        #     augmented = self.augmentations(image=image, mask=mask)
        #     # Extract results
        #     image = augmented['image']
        #     mask = augmented['mask']

        image = np.moveaxis(image, -1, 0)
        return torch.tensor(np.array(image).astype(np.float32)), np.array(mask).astype(np.int32) - 1
    
def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/ade20k")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = ADE20kDataset(root=data_path, split="training" if subset == "train" else "validation")
    return ds, transform