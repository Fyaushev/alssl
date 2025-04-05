import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        # transforms.RandomRotation(10),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Resize((224, 224), antialias=True),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def get_num_classes():
    return 2

class CVCClinicDBDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', test_size=100):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "PNG/Original")
        self.mask_dir = os.path.join(root_dir, "PNG/Ground Truth")
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        np.random.seed(42)
        dataset_size = len(self.image_filenames)
        split_size = test_size if split == 'test' else dataset_size - test_size
        selected_indices = np.random.choice(range(dataset_size), split_size, replace=False)

        self.image_filenames = [f for i, f in enumerate(self.image_filenames) if i in selected_indices]
        self.mask_filenames = [f for i, f in enumerate(self.mask_filenames) if i in selected_indices]

        self.targets = np.ones(split_size)

        self.transform = transform

        self.mask_transform = transforms.Resize((224, 224), interpolation=Image.NEAREST)

        self.class_dict = {
            (0, 0, 0): 0,       # Background
            (255, 255, 255): 1  # Polyp
        }

    def __len__(self):
        return len(self.image_filenames)

    def mask_to_class(self, mask):
        mask = np.array(mask)
        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for rgb, class_id in self.class_dict.items():
            matches = (mask == rgb).all(axis=-1)
            class_mask[matches] = class_id

        return class_mask
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") 

        mask = self.mask_transform(mask) 
        mask = self.mask_to_class(mask)  
        
        if self.transform:
            image = self.transform(image)
        # mask = torch.tensor(mask, dtype=torch.long)  

        return np.array(image).astype(np.float32), mask
    
def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/cvc")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds = CVCClinicDBDataset(root_dir=data_path, split=subset)
    return ds, transform

