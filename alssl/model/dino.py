import lightning as L
import torch
from torch import nn
from transformers import AutoModel
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DinoClassifier(nn.Module):
    def __init__(self):
        super(DinoClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
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
    ):
        super().__init__()
        self.model = DinoClassifier()
        self.learning_rate = learning_rate
        self.validation_losses = []
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch
        val_logits = self(val_images)

        val_loss = self.criterion(val_logits, val_labels)
        self.log("val_loss", val_loss, prog_bar=True)

        val_acc = accuracy(
            torch.argmax(val_logits, dim=1),
            val_labels,
            task="multiclass",
            num_classes=self.num_classes,
        )
        self.log("val_acc", val_acc, prog_bar=True)
        return val_loss
