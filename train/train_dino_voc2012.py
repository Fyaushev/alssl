import random

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, NeptuneLogger

# from clearml import Task

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from alssl.data.voc2012 import VOCLightningDataModule
from alssl.model.dino import LightningDinoSegmentation

# Configuration
exp_name = "segm_exp5"
batch_size = 13
num_epochs = 2000
learning_rate = 1e-5

random_state = 0
trains_size = 200
val_size = 10
num_workers = 24
checkpoint_every_n_epochs = 50
check_val_every_n_epoch = 1

L.seed_everything(seed=random_state, workers=True)
random.seed(random_state)

train_transform = A.Compose(
    [
        A.Resize(448, 448),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(448, 448),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# Training

data_module = VOCLightningDataModule(
    batch_size=batch_size,
    trains_size=trains_size,
    val_size=val_size,
    random_state=random_state,
    num_workers=num_workers,
    train_transform=train_transform,
    test_transform=test_transform,
)

# logs
# wandb_logger = WandbLogger(name=exp_name, project="alssl")
neptune_logger = NeptuneLogger(
    name=exp_name, project="fyaush/alssl", log_model_checkpoints=False
)

# task = Task.init(project_name="alssl", task_name=exp_name)
# clearml_logger = task.get_logger()

checkpoint_lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=checkpoint_every_n_epochs, save_top_k=-1, save_last=False
)


trainer = Trainer(callbacks=[checkpoint_lr_monitor])

model = LightningDinoSegmentation(
    learning_rate=learning_rate,
    optimizer_kwargs={"weight_decay": 1e-2},
    scheduler_kwargs={"factor": 0.1},
)

trainer = L.Trainer(
    max_epochs=num_epochs,
    check_val_every_n_epoch=check_val_every_n_epoch,
    logger=neptune_logger,
    # logger=wandb_logger,
    callbacks=[checkpoint_lr_monitor, checkpoint_callback],
)

trainer.fit(model, datamodule=data_module)
trainer.validate(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
