import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def get_trainer(trainer, trial):

    return L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=trainer.num_epochs,
        accelerator="auto",
        check_val_every_n_epoch=trainer.check_val_every_n_epoch,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

def run_optuna(trainer, len_train_dataloader, n_trials, pruning=True):
    '''
    adapted from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
    '''
    
    def objective(trial: optuna.trial.Trial, ) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)

        # model, datamodule, trainer
        module, hyperparams = trainer._get_lightning_module(len_train_dataloader)
        hyperparams['learning_rate'] = learning_rate
        model = module(**hyperparams)

        datamodule = trainer.al_datamodule

        trainer_ =  get_trainer(trainer, trial)
        
        hyperparameters = dict(learning_rate=learning_rate,)
        trainer_.logger.log_hyperparams(hyperparameters)
        trainer_.fit(model, datamodule=datamodule)

        return trainer_.callback_metrics["val_acc"].item()
    
    pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params["learning_rate"]