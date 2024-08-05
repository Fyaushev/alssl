import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import wandb
from dpipe.io import save
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from ..strategy.base import BaseStrategy


class ALTrainer:
    """
    Class for running an Active Learning training loop.
    """

    def __init__(
        self,
        exp_root_path: Path,
        exp_name: Path,
        al_strategy: BaseStrategy,
        al_datamodule: ALDataModule,
        al_model: BaseALModel,
        budget_size: int,
        initial_train_size: int,
        initial_val_size: int,
        n_iter: int,
        random_seed: int,
        *,
        finetune=True,
        project_name="alssl",
        checkpoint_every_n_epochs=50,
        num_epochs=101,
        check_val_every_n_epoch=1,
    ):
        """
        Args:
            exp_root_path: Path to the root directory for experiment logs.
            exp_name: Name of the experiment.
            al_strategy: The active learning strategy to use.
            al_datamodule: The data module for active learning.
            al_model: The model to be trained.
            budget_size: The number of new samples to select in each iteration.
            initial_train_size: Number of initial training samples.
            initial_val_size: Number of initial validation samples.
            n_iter: Number of active learning iterations.
            random_seed: Random seed for reproducibility.
            finetune: Whether to finetune the model or reinitialize in each iteration.
            project_name: Project name for logging (e.g., in Wandb).
            checkpoint_every_n_epochs: Frequency of saving model checkpoints.
            num_epochs: Number of epochs to train in each iteration.
            check_val_every_n_epoch: Frequency of validation checks.
        """
        self.exp_root_path = Path(exp_root_path)
        self.exp_name = exp_name
        self.exp_path = self.exp_root_path / self.exp_name
        # Create experiment directory
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name

        self.random_seed = random_seed
        self.al_strategy = al_strategy
        self.al_datamodule = al_datamodule
        self.al_model = al_model
        self.budget_size = budget_size
        self.n_iter = n_iter

        self.initial_train_size = initial_train_size
        self.initial_val_size = initial_val_size

        self.finetune = finetune
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.num_epochs = num_epochs

    def run(self):
        """
        Runs the active learning training loop.
        """

        # Set random seed for reproducibility
        L.seed_everything(seed=self.random_seed, workers=True)
        random.seed(self.random_seed)

        # Prepare training and validation sets
        # TODO: stratify?
        train_ids, val_ids = train_test_split(
            np.arange(len(self.al_datamodule.full_train_dataset)),
            train_size=self.initial_train_size,
            test_size=self.initial_val_size,
            random_state=self.random_seed,
        )

        self.al_datamodule.set_train_ids(list(train_ids))
        self.al_datamodule.set_val_ids(list(val_ids))
        module = self.al_model.get_lightning_module()

        for i in tqdm(range(self.n_iter), desc="AL iteration"):
            cur_exp_name = f"{self.exp_name}_iter_{i}"
            (self.exp_path / cur_exp_name).mkdir(parents=True)
            os.chdir(self.exp_path / cur_exp_name)  # FIXME

            save(self.al_datamodule.train_ids, "train_ids.json")
            save(self.al_datamodule.val_ids, "val_ids.json")

            wandb_run = wandb.init(
                dir=self.exp_path / cur_exp_name,
                project=self.project_name,
                group=self.exp_name,
                name=cur_exp_name,
            )

            # Setup Wandb logger and callbacks
            wandb_logger = WandbLogger(name=cur_exp_name, project=self.project_name)
            checkpoint_lr_monitor = LearningRateMonitor(logging_interval="epoch")
            checkpoint_callback = ModelCheckpoint(
                every_n_epochs=self.checkpoint_every_n_epochs,
                save_top_k=-1,
            )

            if i and not self.finetune:
                module = self.al_model.get_lightning_module()

            trainer = L.Trainer(
                max_epochs=self.num_epochs,
                check_val_every_n_epoch=self.check_val_every_n_epoch,
                logger=wandb_logger,
                callbacks=[checkpoint_lr_monitor, checkpoint_callback],
            )

            trainer.fit(module, datamodule=self.al_datamodule)
            trainer.validate(module, datamodule=self.al_datamodule)
            test_metrics = trainer.test(module, datamodule=self.al_datamodule)

            active_learning_id = self.al_strategy.select_ids(
                module, self.al_datamodule, self.budget_size
            )
            self.al_datamodule.update_train_ids(active_learning_id)
            wandb_run.finish()

            # Log summary metrics to Wandb
            if i == 0:
                wandb_run = wandb.init(
                    dir=self.exp_path,
                    project=self.project_name,
                    group=self.exp_name,
                    name=f"{self.exp_name}_summary",
                )
                run_id = wandb_run.id
            else:
                wandb_run = wandb.init(
                    dir=self.exp_path,
                    project=self.project_name,
                    group=self.exp_name,
                    name=f"{self.exp_name}_summary",
                    id=run_id,
                    resume="must",
                )

            wandb_run.log(test_metrics[0], step=i)  # FIXME
            wandb_run.finish()
