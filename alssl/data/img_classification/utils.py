# same as https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/utils.py

import os

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader


class CustomImageFolder(DatasetFolder):
    def __init__(
        self,
        root,
        file_path,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = []

        with open(file_path, "r") as f:
            self.targets = []
            for line in f.readlines():
                img_path, label = line.strip().split()
                self.samples.append((os.path.join(root, img_path), int(label)))
                self.targets.append(int(label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)