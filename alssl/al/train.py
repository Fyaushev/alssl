import logging
import os
from pathlib import Path

import lightning as L
import numpy as np
import wandb
from dpipe.io import load, save
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from ..strategy.base import BaseStrategy
from .utils import efficient_chdir, fix_seed, get_checkpoint, last_checkpoint

# clear output
os.environ['WANDB_SILENT']="true"
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

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
        config: dict,
        *,
        finetune=True,
        project_name="alssl",
        checkpoint_every_n_epochs=50,
        num_epochs=101,
        check_val_every_n_epoch=1,
        entitiy='dnogina'
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
        self.entitiy = entitiy
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
        self.config = config
        self.run_id = None

    def get_trainer(self, cur_exp_name):
        # Setup Wandb logger and callbacks
        wandb_logger = WandbLogger(name=cur_exp_name, project=self.project_name)
        checkpoint_lr_monitor = LearningRateMonitor(logging_interval="epoch")
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=self.checkpoint_every_n_epochs,
            save_weights_only=True,
            save_top_k=-1,
        )
        return L.Trainer(
            max_epochs=self.num_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            logger=wandb_logger,
            callbacks=[checkpoint_lr_monitor, checkpoint_callback],
        )
    
    def log_summary(self, i, test_metrics):
        # Log summary metrics to Wandb
        wandb_run = wandb.init(
            dir=self.exp_path,
            project=self.project_name,
            group=self.exp_name,
            name=f"{self.exp_name}_summary",
            id=self.run_id,
            resume="must" if self.run_id is not None else None,
            entity=self.entitiy,
            config=self.config
        )
        self.run_id = wandb_run.id

        wandb_run.log(test_metrics[0], step=i)  # FIXME
        wandb_run.finish()

    def train_model(self, i, curr_dir, module, is_fully_trained):
        cur_exp_name = f"iter_{i}"

        save(self.al_datamodule.train_ids, "train_ids.json")
        save(self.al_datamodule.val_ids, "val_ids.json")

        trainer = self.get_trainer(cur_exp_name)

        if is_fully_trained:
            print('is fully trained, do not retrain')
            return module

        wandb_run = wandb.init(
            dir=curr_dir,
            project=self.project_name,
            group=self.exp_name,
            name=cur_exp_name,
            entity=self.entitiy,
        )

        trainer.fit(module, datamodule=self.al_datamodule)
        trainer.validate(module, datamodule=self.al_datamodule, verbose=False)
        test_metrics = trainer.test(module, datamodule=self.al_datamodule, verbose=False)
        wandb_run.finish()
        self.log_summary(i, test_metrics)
        return module
    
    def load_model(self, curr_dir, prev_dir, iteration):
        """
        Each active learning iteration is a finetuning of the previous model.

        First, load the latest checkpoint either from current or previous iteration. 
        Checkpoint restored from the current iteration will be used only if the model was fully trained. Otherwise, the training will be done from the first epoch.
        """
        checkpoint_path = last_checkpoint(curr_dir)
        is_fully_trained = checkpoint_path is not None and checkpoint_path.stem.startswith(f'epoch={self.num_epochs - 1}')
        if iteration > 0 and checkpoint_path is None or not is_fully_trained:
            checkpoint_path = last_checkpoint(prev_dir)

        if iteration > 1:
            train_ids_path = curr_dir / "train_ids.json" if (curr_dir / "train_ids.json").exists() else prev_dir / "train_ids_after_update.json"
            print('set train ids, train_ids_path', train_ids_path)
            self.al_datamodule.set_train_ids(load(train_ids_path))

        module = self.al_model.get_lightning_module()
        if checkpoint_path is None:
            print('do NOT load checkpoint')
            model = module(**self.al_model.get_hyperparameters())
        else:
            print('DO load checkpoint')
            model = module.load_from_checkpoint(checkpoint_path, **self.al_model.get_hyperparameters())
        print(checkpoint_path)
        return model, is_fully_trained

    def run(self):
        """
        Runs the active learning training loop.
        """

        # Set random seed for reproducibility
        self.random_seed, rng = fix_seed(seed=self.random_seed)

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

        zero_iteration_dir = self.exp_path.parent.parent / 'zero_iteration'

        for i in tqdm(range(self.n_iter), desc="AL iteration", colour='green'):
            curr_dir = zero_iteration_dir if i == 0 else self.exp_path / f"iter_{i}"
            prev_dir = zero_iteration_dir if i == 1 else self.exp_path / f"iter_{i - 1}"

            # if we already saved the selected ids of this round, the iteration is complete
            if (curr_dir / "train_ids_after_update.json").exists():
                self.al_datamodule.set_train_ids(load(curr_dir / "train_ids_after_update.json"))
                continue
            # for the first iteration, we don't save the selected ids, because it is a shared folder
            if i == 0 and (self.exp_path / f"iter_{i+1}" / "train_ids.json").exists():
                self.al_datamodule.set_train_ids(load(self.exp_path / f"iter_{i+1}" / "train_ids.json"))
                continue

            efficient_chdir(curr_dir)
            
            model, is_fully_trained = self.load_model(curr_dir, prev_dir, i)
            self.train_model(i, curr_dir, model, is_fully_trained)

            active_learning_id = self.al_strategy.select_ids(
                model, self.al_datamodule, self.budget_size, self.al_model
            )
            self.al_datamodule.update_train_ids(active_learning_id)
            if i > 0:
                save(self.al_datamodule.train_ids, "train_ids_after_update.json")
