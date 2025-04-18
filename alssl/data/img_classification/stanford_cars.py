from pathlib import Path

import torchvision
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
    return 196


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/stanford_cars")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = torchvision.datasets.StanfordCars(
            root=data_path.parent, split=subset, #download=True
        )
    ds.targets = [i[1] for i in ds._samples]
    return (
        ds,
        transform,
    )