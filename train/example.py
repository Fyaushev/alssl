import argparse
import importlib
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from alssl.al.train import ALTrainer
from alssl.data.base import ALDataModule
from alssl.model.base import BaseALModel
from alssl.model.dino import LightningDinoClassifier
from alssl.strategy import strategies

ALTrainer, BaseALModel, ALDataModule


@hydra.main(version_base=None, config_path=".", config_name="config")
def run_exp(config: DictConfig) -> None:
    # Load the dataset
    ds_utils = importlib.import_module(f'alssl.data.img_classification.{config.experiment.dataset}')
    root_path = Path(config.experiment.exp_root_path) / config.experiment.dataset
    data_path = Path(config.experiment.data_path) / config.experiment.dataset

    train_dataset = ds_utils.get_dataset(subset="train", data_path=data_path)
    test_dataset = ds_utils.get_dataset(subset="test", data_path=data_path)

    # Initialize the Active Learning data module with the datasets and batch size
    data_module = ALDataModule(
        full_train_dataset=train_dataset,
        full_test_dataset=test_dataset,
        batch_size=config.training.batch_size,
    )

    # MODEL
    num_classes = ds_utils.get_num_classes()

    class Model(BaseALModel):
        def get_lightning_module(self):
            return LightningDinoClassifier
        def get_hyperparameters(self):
            return {
                'learning_rate':config.training.learning_rate,
                'num_classes':num_classes,
                'optimizer_kwargs':config.training.optimizer_kwargs,
                'scheduler_kwargs':config.training.scheduler_kwargs,
            }

    model = Model()

    # TRAIN

    N = len(train_dataset)
    budget_size = int(config.strategy.budget_percent / 100 * N)
    initial_train_size = int(config.strategy.initial_train_percent / 100 * N)

    # Initialize the Active Learning trainer
    trainer = ALTrainer(
        exp_root_path=root_path / str(config.strategy.initial_train_percent) / str(config.training.random_seed) / str(config.strategy.budget_percent),
        exp_name=config.strategy.strategy_name,
        al_strategy=partial(strategies[config.strategy.strategy_name], **config.strategy.strategy_params)(),
        al_datamodule=data_module,
        al_model=model,
        
        budget_size=budget_size,
        initial_train_size=initial_train_size,
        initial_val_size=config.strategy.initial_val_size,
        n_iter=config.strategy.n_iter,
        
        random_seed=config.training.random_seed,
        num_epochs=config.training.num_epochs,
        checkpoint_every_n_epochs=config.training.checkpoint_every_n_epochs,
        config=OmegaConf.to_container(config),
    )
    trainer.run()

if __name__ == "__main__":
    run_exp()
