from pathlib import Path

import numpy as np
from medmnist import INFO, PathMNIST
from torchvision import transforms

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
    len(INFO["pathmnist"]["label"])
    return 9


class PathMNISTDataset(PathMNIST):
    def __init__(self, root, split, download, size=224):
        super().__init__(
            root=root, split=split, download=download, size=size
        )
        self.targets = self.labels.ravel()

    def __getitem__(self, index):
        # image = np.moveaxis(self.imgs[index], -1, 0)  # [224, 224, 3] to [3, 224, 224]
        image = self.imgs[index]
        label = self.labels[index][0]  # [label] to label
        return image, label

def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/pathmnist")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = PathMNISTDataset(root=data_path, split=subset, download=True)
    return ds, transform
