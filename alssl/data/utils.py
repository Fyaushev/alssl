from torch.utils.data import Dataset, Subset


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.subset = Subset(dataset, indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
