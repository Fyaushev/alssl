import argparse
import importlib
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from dpipe.io import load, save
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from alssl.al.train import ALTrainer
from alssl.al.utils import efficient_chdir, fix_seed
from alssl.coldstart import coldstarts
from alssl.data.base import ALDataModule
from alssl.model.base import BaseALModel
from alssl.model.clip import LightningCLIPClassifier
from alssl.model.dino import LightningDinoClassifier
from alssl.strategy import strategies
from alssl.strategy.utils import predict


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def run_iteration_pred(model, dataset: ALDataModule, almodel, i, num_neighbours=500, metric="cosine"):
    """
    Calculate nearest-neighbor-based scores and return neighbors for further operations.
    """
    # Load previous model if required
    prev_model = almodel.get_lightning_module()(**almodel.get_hyperparameters())

    all_dataset = MyDataset(dataset.full_train_dataset, transform=dataset.transform_test)
    all_dataloader = DataLoader(all_dataset, batch_size=1_000, shuffle=False, num_workers=5,)

    # Compute embeddings and neighbors for original and finetuned models
    if i == 0:
        y_gt_e0, y_pred_e0, features_e0 = predict(
            prev_model,
            all_dataloader, 
            scoring="none", desc="E0")
        
        save(features_e0, "e0.npy.gz", compression=1)

        neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric, n_jobs=-1)
        neigh.fit(X=features_e0)
        neighbors_e0 = neigh.kneighbors(X=features_e0, return_distance=False)[:, 1:]
        save(neighbors_e0, "neighbors_e0.npy.gz", compression=1)

    y_gt, y_pred_e1, features_e1 = predict(
        model,
        all_dataloader, 
        scoring="none", desc="E1")
    
    save(y_gt, "y_gt.npy.gz", compression=1)
    save(y_pred_e1, "y_pred_e1.npy.gz", compression=1)
    save(features_e1, "e1.npy.gz", compression=1)
    neigh = NearestNeighbors(n_neighbors=num_neighbours, metric=metric, n_jobs=-1)
    neigh.fit(X=features_e1)
    neighbors_e1 = neigh.kneighbors(X=features_e1, return_distance=False)[:, 1:]
    save(neighbors_e1, "neighbors_e1.npy.gz", compression=1)



class SavingALTrainer(ALTrainer):
    def run(self):
        """
        Runs the active learning training loop.
        """

        # Set random seed for reproducibility
        self.random_seed, rng = fix_seed(seed=self.random_seed)
        # TODO: it is run each time including coldstart, but further iteration can be already calculated. They are checked in the cycle
        self.setup_datamodule()

        # zero_iteration_dir = self.exp_path.parent.parent / 'zero_iteration'

        for i in tqdm(range(self.n_iter), desc="AL iteration", colour='green'):
            curr_dir = self.zero_iteration_dir if i == 0 else self.exp_path / f"iter_{i}"
            prev_dir = self.zero_iteration_dir if i == 1 else self.exp_path / f"iter_{i - 1}"
            
  
            # for the first iteration, we don't save the selected ids, because it is a shared folder
            if i == 0 and (self.exp_path / f"iter_1" / "train_ids.json").exists():
                self.al_datamodule.set_train_ids(load(self.exp_path / f"iter_1" / "train_ids.json"))
            # if we already saved the selected ids of this round, the iteration is complete
            elif (curr_dir / "train_ids_after_update.json").exists():
                self.al_datamodule.set_train_ids(load(curr_dir / "train_ids_after_update.json"))

            if (curr_dir / "neighbors_e1.npy.gz").exists():
                continue
            
            efficient_chdir(curr_dir)
            model, is_fully_trained = self.load_model(curr_dir, prev_dir, i, len(self.al_datamodule.train_dataloader()))
            print('is_fully_trained', is_fully_trained)
            assert is_fully_trained, f"Model is not fully trained on iteration {i}, directory: {curr_dir}"

            run_iteration_pred(model, self.al_datamodule, self.al_model, i)
            

@hydra.main(version_base=None, config_path=".", config_name="config_gen_embeddings")
def run_exp(config: DictConfig) -> None:
    # Load the dataset
    print("start loading dataset")
    ds_utils = importlib.import_module(f'alssl.data.img_classification.{config.experiment.dataset}')
    root_path = Path(config.experiment.exp_root_path) / config.experiment.dataset / config.training.backbone
    data_path = Path(config.experiment.data_path) / config.experiment.dataset
    print("start loading test_dataset")
    test_dataset, transform_test = ds_utils.get_dataset(subset="test", data_path=data_path)
    print("start loading train_dataset")
    train_dataset, transform_train = ds_utils.get_dataset(subset="train", data_path=data_path)
    

    # Initialize the Active Learning data module with the datasets and batch size
    data_module = ALDataModule(
        full_train_dataset=train_dataset,
        full_test_dataset=test_dataset,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=config.training.batch_size,
        batch_size_prediction=config.training.batch_size_prediction,
        num_workers = config.training.num_workers
    )

    # MODEL
    num_classes = ds_utils.get_num_classes()

    scheduler_kwargs = {
        "max_lr": config.training.learning_rate,
        "epochs": config.training.num_epochs,
        "steps_per_epoch": 10, # depends on the size of training loader and is set in train.py
    }
    
    class Model(BaseALModel):
        def get_lightning_module(self):
            if config.training.backbone == 'dino':
                return LightningDinoClassifier

        def get_hyperparameters(self):
            return {
                'learning_rate':config.training.learning_rate,
                'num_classes':num_classes,
                'optimizer_kwargs':config.training.optimizer_kwargs,
                'scheduler_kwargs':scheduler_kwargs,
                'include_param_loss':config.training.include_param_loss,
            }

    model = Model()

    # TRAIN
    N = len(train_dataset)
    if config.strategy.budget_per_class > 0:
        budget_size = config.strategy.budget_per_class * num_classes
    else:
        budget_size = int(config.strategy.budget_percent / 100 * N)
    initial_train_size = int(config.strategy.initial_train_percent / 100 * N)

    print('budget_size', budget_size)
    print('initial_train_size', initial_train_size)
    print('N', N)
    print('config.strategy.budget_percent', config.strategy.budget_percent)
    print('config.strategy.initial_train_percent', config.strategy.initial_train_percent)

    exp_name = config.strategy.strategy_name + '_' + '_'.join([f'{k[:4]}-{v}' for k, v in config.strategy.strategy_params.items()])
    coldstart_name = config.coldstart.coldstart_name + '_' + '_'.join([f'{k}-{v}' for k, v in config.coldstart.coldstart_params.items()])
    if config.training.include_param_loss:
        exp_name += '_ploss'
        coldstart_name += '_ploss'
    if config.training.blocks_to_retrain == 0:
        exp_name += '_frozen'
        coldstart_name += '_frozen'
    
    exp_root_path = root_path / (str(config.strategy.initial_train_percent) + '_' + coldstart_name) / str(config.training.random_seed) 
    
    if config.strategy.budget_per_class > 0:
        exp_root_path = exp_root_path / str(config.strategy.budget_per_class)
        # budget_size = config.strategy.budget_per_class * num_classes
    else:
        exp_root_path = exp_root_path / str(config.strategy.budget_percent)

    with open_dict(config):
        config.coldstart.coldstart_params.initial_train_size = initial_train_size
        config.coldstart.coldstart_params.random_seed = config.training.random_seed
        config.coldstart.coldstart_params.num_classes = num_classes
    
    print('exp_root_path', exp_root_path)

    if not config.experiment.exp_names:
        exp_names = [i.stem for i in exp_root_path.glob('*')]
    else:
        exp_names = config.experiment.exp_names


    for exp_name in exp_names:
        print(exp_root_path / exp_name)

        n_iter = max([int(i.stem.split('_')[-1]) for i in (exp_root_path / exp_name).glob('iter_*')]) + 1
    
        # Initialize the Active Learning trainer
        trainer = SavingALTrainer(
            exp_root_path=exp_root_path,
            exp_name=exp_name,
            al_strategy=exp_name, #partial(strategies[config.strategy.strategy_name], **config.strategy.strategy_params)(),
            al_coldstart=partial(coldstarts[config.coldstart.coldstart_name], **config.coldstart.coldstart_params)(),
            al_datamodule=data_module,
            al_model=model,
            
            budget_size=budget_size,
            initial_train_size=initial_train_size,
            initial_val_size=config.strategy.initial_val_size,
            n_iter=n_iter,
            
            finetune=config.training.finetune,
            optuna_trials=config.training.optuna_trials,
            random_seed=config.training.random_seed,
            num_epochs=config.training.num_epochs,
            checkpoint_every_n_epochs=config.training.num_epochs,
            check_val_every_n_epoch=config.training.check_val_every_n_epoch,
            config=OmegaConf.to_container(config),
            entitiy=config.experiment.wandb_entitiy
        )
        # trainer.num_neighbours = config.strategy.num_neighbours
        try:
            trainer.run()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    run_exp()
