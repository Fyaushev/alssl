from pathlib import Path

import torchvision
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
    return 10

def get_dataset(subset="train", data_path=Path("/shared/projects/active_learning/svhn")):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    return torchvision.datasets.SVHN(
        root=data_path, split=subset, download=True, transform=transform
    )
