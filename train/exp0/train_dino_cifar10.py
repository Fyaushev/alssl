import lightning as L

from alssl.data.cifar10 import CIFAR10LightningDataModule
from alssl.model.dino import LightningDino

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from torchvision import transforms
from lightning.pytorch.callbacks import LearningRateMonitor


# Configuration
exp_name = "exp0"
batch_size = 40
num_epochs = 2000
learning_rate = 1e-4

random_state = 0
trains_size = 950
val_size = 50
random_state = 0
num_workers = 8

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
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
wandb_logger = WandbLogger(log_model=exp_name, project="alssl")
lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = Trainer(callbacks=[lr_monitor])

print("len(data_module.train_dataloader): ", len(data_module.train_dataloader()))

L.seed_everything(seed=random_state, workers=True)
model = LightningDino(
    learning_rate=learning_rate,
    scheduler_kwargs={
        "max_lr": 1e-2,
        "epochs": num_epochs,
        "steps_per_epoch": len(data_module.train_dataloader()),
    },
)
trainer = L.Trainer(
    max_epochs=num_epochs,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    callbacks=[lr_monitor],
    accelerator="cuda",
)

trainer.fit(model, data_module)
trainer.validate(model, datamodule=data_module)
