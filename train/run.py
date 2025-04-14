import argparse
import importlib
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from alssl.al.train import ALTrainer
from alssl.coldstart import coldstarts
from alssl.data.base import ALDataModule
from alssl.model.base import BaseALModel
from alssl.model.dino import LightningDinoClassifier
from alssl.model.effnet_b0 import LightningEffNetB0Classifier
from alssl.model.lvm_med_resnet import LightningLMVMedResnetClassifier
# from alssl.model.lvm_med_vit import LightningLMVMedVitClassifier
from alssl.model.mae import LightningMAEClassifier
from alssl.model.regnet_x_400mf import LightningRegNetX400MFClassifier
from alssl.model.resnet18 import LightningResnet18Classifier
from alssl.strategy import strategies

ALTrainer, BaseALModel, ALDataModule


@hydra.main(version_base=None, config_path=".", config_name="config")
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
            elif config.training.backbone == 'effnet_b0':
                return LightningEffNetB0Classifier
            elif config.training.backbone == 'regnet_x_400mf':
                return LightningRegNetX400MFClassifier
            elif config.training.backbone == 'resnet18':
                return LightningResnet18Classifier
            elif config.training.backbone == 'lvm_med_resnet':
                return LightningLMVMedResnetClassifier
            # elif config.training.backbone == 'lvm_med_vit':
            #     return LightningLMVMedVitClassifier
            elif config.training.backbone == 'mae':
                return LightningMAEClassifier
        def get_hyperparameters(self):
            return {
                'learning_rate':config.training.learning_rate,
                'num_classes':num_classes,
                'blocks_to_retrain':config.training.blocks_to_retrain,
                'optimizer_kwargs':config.training.optimizer_kwargs,
                'scheduler_kwargs':scheduler_kwargs,
                'include_param_loss':config.training.include_param_loss,
                'root':config.experiment.data_path,
                'param_loss_beta': config.training.param_loss_beta,
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
    if config.training.finetune:
        exp_name += '_finetune'
        coldstart_name += '_finetune'
    
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
    
    # Initialize the Active Learning trainer
    trainer = ALTrainer(
        exp_root_path=exp_root_path,
        exp_name=exp_name,
        al_strategy=partial(strategies[config.strategy.strategy_name], **config.strategy.strategy_params)(),
        al_coldstart=partial(coldstarts[config.coldstart.coldstart_name], **config.coldstart.coldstart_params)(),
        al_datamodule=data_module,
        al_model=model,
        
        budget_size=budget_size,
        initial_train_size=initial_train_size,
        initial_val_size=config.strategy.initial_val_size,
        n_iter=config.strategy.n_iter,
        
        finetune=config.training.finetune,
        optuna_trials=config.training.optuna_trials,
        random_seed=config.training.random_seed,
        num_epochs=config.training.num_epochs,
        checkpoint_every_n_epochs=config.training.num_epochs,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        config=OmegaConf.to_container(config),
        entitiy=config.experiment.wandb_entitiy
    )
    trainer.run()

if __name__ == "__main__":
    run_exp()
