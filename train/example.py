import argparse
from functools import partial
from pathlib import Path

from torchvision import transforms

from alssl.al.train import ALTrainer
from alssl.data.base import ALDataModule
from alssl.data.cifar100 import get_dataset, get_num_classes
from alssl.model.base import BaseALModel
from alssl.model.dino import LightningDinoClassifier
from alssl.strategy import strategies
from alssl.utils import parse_run_config

ALTrainer, BaseALModel, ALDataModule


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config path', required=True)
args = parser.parse_args()

config = parse_run_config(args.config)


# DATA
# Define the transformations for training and test datasets
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(config.dataset_transforms["crop_size"], padding=config.dataset_transforms["padding"]),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config.dataset_transforms["image_size"], antialias=config.dataset_transforms["antialias"]),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=config.dataset_transforms["normalize_mean"], std=config.dataset_transforms["normalize_std"]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(config.dataset_transforms["image_size"], antialias=config.dataset_transforms["antialias"]),
        transforms.Normalize(mean=config.dataset_transforms["normalize_mean"], std=config.dataset_transforms["normalize_std"]),
    ]
)

# Load the CIFAR-100 datasets with the specified transformations
train_dataset = get_dataset(subset="train", transform=train_transform)
test_dataset = get_dataset(subset="test", transform=train_transform)

# Initialize the Active Learning data module with the datasets and batch size
data_module = ALDataModule(
    full_train_dataset=train_dataset,
    full_test_dataset=test_dataset,
    batch_size=config.training["batch_size"],
)

# MODEL
# Define some model parameters

num_classes = get_num_classes()


class Model(BaseALModel):
    def get_lightning_module(self):
        return LightningDinoClassifier
    def get_hyperparameters(self):
        return {
            'learning_rate':config.training["learning_rate"],
            'num_classes':num_classes,
            'optimizer_kwargs':config.training["optimizer_kwargs"],
            'scheduler_kwargs':config.training["scheduler_kwargs"],
        }


model = Model()


# TRAIN
# Initialize the Active Learning trainer
trainer = ALTrainer(
    exp_root_path=Path(config.experiment["exp_root_path"]) / str(config.strategy["initial_train_size"]) / str(config.training["random_seed"]) / str(config.strategy["budget_size"]),
    exp_name=config.strategy["strategy_name"],
    al_strategy=partial(strategies[config.strategy["strategy_name"]], **config.strategy["strategy_params"])(),
    al_datamodule=data_module,
    al_model=model,
    
    budget_size=config.strategy["budget_size"],
    initial_train_size=config.strategy["initial_train_size"],
    initial_val_size=config.strategy["initial_val_size"],
    n_iter=config.strategy["n_iter"],
    
    random_seed=config.training["random_seed"],
    num_epochs=config.training["num_epochs"],
    checkpoint_every_n_epochs=config.training["checkpoint_every_n_epochs"],
)

if __name__ == "__main__":
    trainer.run()
