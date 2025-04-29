import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

resize_s = (700,700)

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
    return 2

class WeakPolypDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, video_idx, to_transform=True):
        self.video_idx = video_idx
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.to_transform = to_transform
        self.image_filenames = sorted([i for i in Path(image_dir).glob('*.jpg')], key=lambda f: int(f.stem))
        self.annotation_filenames = sorted([i for i in Path(annotation_dir).glob('*.txt')], key=lambda f: int(f.stem))

    def __len__(self):
        return len(self.image_filenames)

    
    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        ann_path = self.annotation_filenames[idx]

        image = Image.open(img_path).convert("RGB")
        with open(ann_path) as f:
            label = int(f.readlines()[0].strip())

        image = cv2.resize(np.array(image), resize_s) / 255
        if self.to_transform:
            image = (image - np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.24, 0.225])
            image = image.transpose(2, 0, 1)


        return torch.tensor(np.array(image).astype(np.float32)), label


class WeakPolypNormaDataset(Dataset):
    def __init__(self, image_dir, video_idx, to_transform=True):
        self.video_idx = video_idx
        self.image_dir = image_dir
        self.to_transform = to_transform
        self.image_filenames = sorted([i for i in Path(image_dir).glob('*.jpg')], key=lambda f: int(f.stem))

    def __len__(self):
        return len(self.image_filenames)

    
    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]

        image = Image.open(img_path).convert("RGB")

        image = cv2.resize(np.array(image), resize_s) / 255
        if self.to_transform:
            image = (image - np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.24, 0.225])
            image = image.transpose(2, 0, 1)


        return torch.tensor(np.array(image).astype(np.float32)), 0 # label is zero for the norm
    

class WeakPolypDatasetCollection:
    def __init__(self, root_dir, subset='abnormal', split='test', to_transform=True):
        self.image_dir = os.path.join(root_dir, f"MICCAI_2022/{split}_set/{subset}_test/Images")
        self.annotations_dir = os.path.join(root_dir, f"MICCAI_2022/{split}_set/{subset}_test/Annotations")

        self.dataset_collection = {}
        if subset == 'abnormal':
            self.video_ids = [i.stem for i in Path(self.annotations_dir).glob('*')]
            for video_idx in self.video_ids:
                self.dataset_collection[int(video_idx)] = WeakPolypDataset(
                    image_dir=os.path.join(self.image_dir, str(video_idx)),
                    annotation_dir=os.path.join(self.annotations_dir, str(video_idx)),
                    video_idx=video_idx,
                    to_transform=to_transform
                )
        elif subset == 'normal':
            self.video_ids = [i.stem for i in Path(self.annotations_dir).parent.glob('*')]
            for video_idx in self.video_ids:
                self.dataset_collection[video_idx] = WeakPolypNormaDataset(
                    image_dir=os.path.join(Path(self.image_dir).parent, str(video_idx)),
                    video_idx=video_idx,
                    to_transform=to_transform
                )

    
def get_dataset(
    subset="test", data_path=Path("/shared/projects/active_learning/weak_polyp_data"), to_transform=True, path_subset='abnormal'
):
    assert subset in ["test"], 'This dataset is not used for training in our experiments'
    transform = transform_train if subset == "train" else transform_test
    ds = WeakPolypDatasetCollection(root_dir=data_path, split=subset, to_transform=to_transform, subset=path_subset)
    return ds, transform

