from pathlib import Path

import torchvision
from torchvision import transforms

from .utils import CustomImageFolder

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
    return 345


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/domainnet"), domainnet_subset="real"
):
    assert subset in ["train", "test"]
    data_path = data_path / 'DomainNet'
    
    transform = transform_train if subset == "train" else transform_test
    return (
        CustomImageFolder(
            root=str(data_path), 
            file_path=str(data_path/f'{domainnet_subset}_{subset}.txt')),
        transform,
    )
