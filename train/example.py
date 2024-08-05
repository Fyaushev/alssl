from pathlib import Path

import torchvision
from torchvision import transforms

from alssl.al.train import ALTrainer
from alssl.data.base import ALDataModule
from alssl.model.base import BaseALModel
from alssl.model.dino import LightningDinoClassifier
from alssl.strategy.random import RandomStrategy

ALTrainer, RandomStrategy, BaseALModel, ALDataModule


DATA_PATH = Path("/shared/projects/active_learning/data/cifar100")

# DATA
# Define the transformations for training and test datasets
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Load the CIFAR-100 datasets with the specified transformations
train_dataset = torchvision.datasets.CIFAR100(
    root=DATA_PATH, train=True, download=False, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR100(
    root=DATA_PATH, train=False, download=False, transform=test_transform
)

# Initialize the Active Learning data module with the datasets and batch size
batch_size = 100
data_module = ALDataModule(
    full_train_dataset=train_dataset,
    full_test_dataset=test_dataset,
    batch_size=batch_size,
)

# MODEL
# Define some model parameters
num_epochs = 51
learning_rate = 1e-5

num_classes = 100
random_state = 0
num_workers = 16
checkpoint_every_n_epochs = 50
check_val_every_n_epoch = 1


class Model(BaseALModel):
    def get_lightning_module(self):
        module = LightningDinoClassifier(
            learning_rate=learning_rate,
            num_classes=num_classes,
            optimizer_kwargs={"weight_decay": 1e-3},
            scheduler_kwargs={"gamma": 0.1, "milestones": [20]},
        )
        return module


model = Model()


# TRAIN
# Initialize the Active Learning trainer
budget_size = 5000
trainer = ALTrainer(
    exp_root_path="/shared/experiments/active_learning/alssl/test",
    exp_name="example",
    al_strategy=RandomStrategy(),
    al_datamodule=data_module,
    al_model=model,
    budget_size=budget_size,
    initial_train_size=5000,
    initial_val_size=200,
    n_iter=10,
    random_seed=0,
    num_epochs=51,
    checkpoint_every_n_epochs=50,
)


trainer.run()
