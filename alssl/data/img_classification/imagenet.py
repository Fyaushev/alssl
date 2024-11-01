from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
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
    return 1000

import numpy as np


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/imagenet")
):
    assert subset in ["train", "test"]

    transform = transform_train if subset == "train" else transform_test

    subset = subset if subset == "train" else "val"

    if subset == "train":
        ds = ImageNet(data_path, split=subset)
        all_ids = list(range(len(ds)))
        size = int(len(all_ids) * 0.1)
        np.random.seed(0)
        new_ids = np.random.choice(
            all_ids, size=size, replace=False, 
        ).tolist()
        return Subset(ds, new_ids), transform

    return ImageNet(data_path, split=subset), transform


# def get_dataset(
#     subset="train", data_path=Path("/shared/projects/active_learning/imagenet")
# ):
#     assert subset in ["train", "test"]

#     transform = transform_train if subset == "train" else transform_test

#     subset = subset if subset == "train" else "val"
#     return ImageNet(data_path, split=subset), transform
