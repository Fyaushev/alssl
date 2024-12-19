from pathlib import Path

from sklearn.model_selection import train_test_split
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


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/imagenet")
):
    assert subset in ["train", "test"]

    transform = transform_train if subset == "train" else transform_test

    subset = subset if subset == "train" else "val"

    ds = ImageNet(data_path, split=subset)

    if subset == "train":
        all_ids = list(range(len(ds)))
        targets = ds.targets

        _, new_ids, _, new_ids_targets = train_test_split(all_ids, targets,
                                                        test_size=0.1,
                                                        random_state=0,
                                                        stratify=targets)
        ds_subset = Subset(ds, new_ids)
        ds_subset.targets = new_ids_targets
        return ds_subset, transform

    return ds, transform


# def get_dataset(
#     subset="train", data_path=Path("/shared/projects/active_learning/imagenet")
# ):
#     assert subset in ["train", "test"]

#     transform = transform_train if subset == "train" else transform_test

#     subset = subset if subset == "train" else "val"
#     return ImageNet(data_path, split=subset), transform
