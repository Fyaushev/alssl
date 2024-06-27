import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics.functional import accuracy
from transformers import AutoModel


class DinoClassifier(nn.Module):
    def __init__(self):
        super(DinoClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        embeddings = self.transformer(x).pooler_output
        logits = self.classifier(embeddings)
        return logits


class LightningDino(L.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        num_classes=10,
        scheduler_kwargs={},
        optimizer_kwargs={},
    ):
        super().__init__()
        self.model = DinoClassifier()
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
        scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)  # "monitor": "train_loss"
        # scheduler = OneCycleLR(optimizer, **self.scheduler_kwargs)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}
        ]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        acc = self._calculate_accuracy(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        acc = self._calculate_accuracy(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        acc = self._calculate_accuracy(logits, labels)
        self.log("test_acc", acc, on_epoch=True, on_step=False)

    def _calculate_accuracy(self, logits, labels):
        return accuracy(
            torch.argmax(logits, dim=1),
            labels,
            task="multiclass",
            num_classes=self.num_classes,
        )
