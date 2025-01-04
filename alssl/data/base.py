import lightning as L
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader

from .utils import TransformSubset


class ALDataModule(L.LightningDataModule):
    """
    A Dataset for Active Learning
    """

    def __init__(
        self,
        full_train_dataset: Dataset,
        full_test_dataset: Dataset,
        transform_train,
        transform_test,
        batch_size: int,
        batch_size_prediction: int,
        *,
        train_ids: list = [],
        val_ids: list = [],
        test_ids: list | None = None,
        num_workers: int = 8,
        shuffle: bool = False,
    ):
        """
        Args:
            full_train_dataset (Dataset): The complete training dataset.
            full_test_dataset (Dataset): The complete test dataset.
            batch_size (int): The batch size for DataLoaders.
            batch_size_prediction (int): The batch size for inference DataLoaders.
            train_ids (List[int], optional): Indices for the training dataset (default is an empty list).
            val_ids (List[int], optional): Indices for the validation dataset (default is an empty list).
            test_ids (Optional[List[int]], optional): Indices for the test dataset, if any (default is None).
            num_workers (int, optional): Number of worker processes for data loading (default is 8).
            shuffle (bool, optional): Whether to shuffle the training dataset (default is False).
        """
        super().__init__()

        self.all_ids = list(range(len(full_train_dataset)))

        assert set(train_ids).isdisjoint(
            val_ids
        ), "train_ids and val_ids should be disjoint"
        assert set(train_ids + val_ids) <= set(
            self.all_ids
        ), "train_ids and val_ids should be a subset of dataset indices"

        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = (
            list(range(len(full_test_dataset))) if test_ids is None else test_ids
        )

        self.batch_size = batch_size
        self.batch_size_prediction = batch_size_prediction
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.full_train_dataset = full_train_dataset
        self.full_test_dataset = full_test_dataset

        self.transform_train = transform_train
        self.transform_test = transform_test

    def set_train_ids(self, ids):
        assert set(ids) <= set(self.all_ids)
        assert set(ids).isdisjoint(self.val_ids)
        self.train_ids = ids

    def set_val_ids(self, ids):
        assert set(ids) <= set(self.all_ids)
        assert set(ids).isdisjoint(self.train_ids)
        self.val_ids = ids

    def set_test_ids(self, ids):
        self.test_ids = ids

    def get_unlabeled_ids(self):
        return list(set(self.all_ids) - set(self.train_ids + self.val_ids))

    def get_train_dataset(self) -> Dataset:
        """
        Get the test dataset subset the train dataset.
        """
        return TransformSubset(
            self.full_train_dataset, self.train_ids, transform=self.transform_train
        )

    def get_val_dataset(self) -> Dataset:
        """
        Get the validation dataset subset with the same transform as the test dataset.
        """
        return TransformSubset(
            self.full_train_dataset, self.val_ids, transform=self.transform_test
        )

    def get_test_dataset(self) -> Dataset:
        """
        Get the test dataset subset or the test dataset.
        """
        return TransformSubset(
            self.full_test_dataset, self.test_ids, transform=self.transform_test
        )

    def get_unlabeled_dataset(self) -> Dataset:
        """
        Get the unlabeled dataset subset or the train dataset.
        """
        return TransformSubset(
            self.full_train_dataset,
            self.get_unlabeled_ids(),
            transform=self.transform_test,
        )

    def update_train_ids(self, active_learning_ids):
        """
        Update the training dataset with new active learning indices.

        Args:
            active_learning_ids (list): List of new indices to be added to the training dataset.
        """
        assert set(self.train_ids + self.val_ids).isdisjoint(active_learning_ids)
        assert set(active_learning_ids) <= set(self.all_ids)

        self.train_ids += list(active_learning_ids)

    def train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def unlabeled_dataloader(self):
        return DataLoader(
            self.get_unlabeled_dataset(),
            batch_size=self.batch_size_prediction,
            shuffle=False,
            num_workers=self.num_workers,
        )
