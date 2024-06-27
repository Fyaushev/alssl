import random

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from alssl.data.cifar10 import CIFAR10LightningDataModule
from alssl.model.dino import LightningDino

# Configuration
exp_name = "exp10"
batch_size = 100
num_epochs = 2000
learning_rate = 3e-4

random_state = 0
trains_size = 950
val_size = 50
num_workers = 16
checkpoint_every_n_epochs = 100
check_val_every_n_epoch = 1

L.seed_everything(seed=random_state, workers=True)
random.seed(random_state)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Training

data_module = CIFAR10LightningDataModule(
    batch_size=batch_size,
    trains_size=trains_size,
    val_size=val_size,
    random_state=random_state,
    num_workers=num_workers,
    train_transform=train_transform,
)

# logs
wandb_logger = WandbLogger(name=exp_name, project="alssl")
checkpoint_lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=checkpoint_every_n_epochs,
    save_top_k=-1,
)


trainer = Trainer(callbacks=[checkpoint_lr_monitor])

model = LightningDino(
    learning_rate=learning_rate,
    optimizer_kwargs={"weight_decay": 1e-1},
    scheduler_kwargs={
        # "max_lr": 1e-2,
        # "epochs": num_epochs,
        # "steps_per_epoch": len(data_module.train_dataloader()),
    },
)
trainer = L.Trainer(
    max_epochs=num_epochs,
    check_val_every_n_epoch=check_val_every_n_epoch,
    logger=wandb_logger,
    callbacks=[checkpoint_lr_monitor, checkpoint_callback],
)

trainer.fit(model, datamodule=data_module)
trainer.validate(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
