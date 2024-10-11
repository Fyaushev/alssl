from pathlib import Path

import numpy as np
from medmnist import INFO, BloodMNIST
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(10),
        # images in the dataset are already normalized
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        # images in the dataset are already normalized
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def get_num_classes():
    len(INFO['bloodmnist']['label'])
    return 8

class BloodMNISTDataset(BloodMNIST):
    def __init__(self, root, split, download, transform, size=224):
        super().__init__(
            root=root, split=split, download=download, transform=transform, size=size
        )

    def __getitem__(self, index):
        image = np.moveaxis(self.imgs[index], -1, 0) # [224, 224, 3] to [3, 224, 224]
        label = self.labels[index][0] # [label] to label
        return image, label

def get_dataset(subset="train", data_path=Path("/shared/projects/active_learning/bloodmnist")):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    return BloodMNISTDataset(
        root=data_path, split=subset, download=True, transform=transform
    )
