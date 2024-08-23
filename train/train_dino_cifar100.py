import random

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from alssl.data.cifar100 import CIFAR100LightningDataModule
from alssl.model.dino import LightningDinoClassifier

# Configuration
exp_name = "5k_exp0_rs0"
batch_size = 100
num_epochs = 101
learning_rate = 1e-5

num_classes = 100
random_state = 0
trains_size = 5000
val_size = 200
num_workers = 16
checkpoint_every_n_epochs = 50
check_val_every_n_epoch = 1

L.seed_everything(seed=random_state, workers=True)
random.seed(random_state)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Training

data_module = CIFAR100LightningDataModule(
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

model = LightningDinoClassifier(
    learning_rate=learning_rate,
    num_classes=num_classes,
    optimizer_kwargs={"weight_decay": 1e-1},
    scheduler_kwargs={"gamma": 0.1, "milestones": [50]},
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