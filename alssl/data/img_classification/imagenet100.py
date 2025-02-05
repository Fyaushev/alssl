from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
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
    return 100


class ImageNet100Dataset(Dataset):
    def __init__(self, split):
        self.ds = load_dataset("clane9/imagenet-100", split=split)
        self.targets = self.ds['label']

    def __getitem__(self, index):
        out_dict = self.ds[index]
        image, label = out_dict['image'], out_dict['label']
        if np.asarray(image).ndim == 2:
            image_3ch = np.zeros( ( np.array(image).shape[0], np.array(image).shape[1], 3 ) )
            image_3ch[:,:,0] = image
            image_3ch[:,:,1] = image
            image_3ch[:,:,2] = image

            image = image_3ch

        if np.asarray(image).shape[-1] == 4:
            image = np.asarray(image)[..., :-1]
        
        return image, label

    def __len__(self):
        return len(self.ds)


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/cifar100")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = ImageNet100Dataset(split = subset if subset == "train" else "validation")
    return (ds, transform)
