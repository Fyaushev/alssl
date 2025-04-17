from pathlib import Path

import numpy as np
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
    return 7


class FER2013Dataset(torchvision.datasets.FER2013):
    def __init__(self, root, split, test_size=1000):
        super().__init__(
            root=root.parent, split="train",
        )
        
        dataset_size = len(self._samples)
        np.random.seed(42)
        self.split_size = test_size if split == 'test' else dataset_size - test_size
        self.selected_indices = np.random.choice(range(dataset_size), self.split_size, replace=False)
        
        self.images = [sample[0] for i, sample in enumerate(self._samples) if i in self.selected_indices]
        self.targets = np.array([sample[1] for i, sample in enumerate(self._samples) if i in self.selected_indices])

    def __len__(self):
        return self.split_size

    def __getitem__(self, index):
        image = self.images[index]

        image_3ch = np.zeros( ( np.array(image).shape[0], np.array(image).shape[1], 3 ) )
        image_3ch[:,:,0] = image
        image_3ch[:,:,1] = image
        image_3ch[:,:,2] = image

        label = self.targets[index]
        return image_3ch.astype(np.float32), label


def get_dataset(
    subset="train", data_path=Path("/shared/projects/active_learning/fer2013")
):
    assert subset in ["train", "test"]
    transform = transform_train if subset == "train" else transform_test
    ds =FER2013Dataset(
            root=data_path, split=subset,
        )

    return (
        ds,
        transform,
    )