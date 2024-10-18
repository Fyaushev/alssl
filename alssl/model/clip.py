import lightning as L
import torch
import torch.nn as nn
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import accuracy
from transformers import AutoModel


class CLIPClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CLIPClassifier, self).__init__()
        self.num_classes = num_classes
        # Load CLIP model (image and text encoder), we'll use the image part
        self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Pass the image through the CLIP image encoder to get image embeddings
        image_embeddings = self.clip_model.get_image_features(pixel_values=x)
        # Pass the embeddings through the classification head
        logits = self.classifier(image_embeddings)
        return logits
    
class LightningCLIPClassifier(L.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        num_classes=10,
        scheduler_kwargs={},
        optimizer_kwargs={},
        include_param_loss: bool = True,
        param_loss_beta: float = 0.01,
    ):
        super().__init__()
        self.model = CLIPClassifier(num_classes=num_classes)
        self.include_param_loss = include_param_loss
        self.param_loss_beta = param_loss_beta
        self.source_weight = {}
        for name, param in self.model.named_parameters():
            self.source_weight[name] = param.detach()
        self.learning_rate = learning_rate
        self.validation_losses = []
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = MultiStepLR(optimizer, **self.scheduler_kwargs)

        # scheduler = ReduceLROnPlateau(
        #     optimizer, **self.scheduler_kwargs
        # )
        # "monitor": "train_loss"
        # scheduler = OneCycleLR(optimizer, **self.scheduler_kwargs)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        ]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        loss = self.criterion(logits, labels)
        self.log("train_loss_criterion", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # code from https://github.com/holyseven/TransferLearningClassification/blob/master/model/network_base.py
        if self.include_param_loss:
            param_loss = 0.0
            for name, param in self.model.named_parameters():
                param_loss += 0.5 * torch.norm(param - self.source_weight[name].to(param)) ** 2
            loss += param_loss * self.param_loss_beta
            self.log("param_loss", param_loss, prog_bar=True, on_epoch=True, on_step=False)

        acc = self._calculate_accuracy(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        loss = self.criterion(logits, labels)
        acc = self._calculate_accuracy(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        acc = self._calculate_accuracy(logits, labels)
        self.log("test_acc", acc, on_epoch=True, on_step=False)

    def _calculate_accuracy(self, logits, labels):
        return accuracy(
            torch.argmax(logits, dim=1),
            labels,
            task="multiclass",
            num_classes=self.num_classes,
        )