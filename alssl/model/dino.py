import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, MultiStepLR
from torchmetrics.functional import accuracy
from torchmetrics.segmentation import MeanIoU
from transformers import AutoModel

from ..metric import mean_iou


class DinoClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DinoClassifier, self).__init__()
        self.num_classes = num_classes
        self.transformer = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        embeddings = self.transformer(x).pooler_output
        logits = self.classifier(embeddings)
        return logits


class LinearClassifierToken(torch.nn.Module):
    def __init__(self, in_channels=768, num_classes=1, token_w=32, token_h=32):
        super(LinearClassifierToken, self).__init__()
        self.in_channels = in_channels

        self.width = token_w
        self.height = token_h

        self.num_classes = num_classes

        self.classifier = torch.nn.Conv2d(in_channels, num_classes, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        return self.classifier(embeddings)


class ConvToken(nn.Module):
    def __init__(self, embedding_size=768, num_classes=21, token_w=32, token_h=32):
        super(ConvToken, self).__init__()
        self.width = token_w
        self.height = token_h
        self.embedding_size = embedding_size

        self.segmentation_conv = nn.Sequential(
            nn.Conv2d(embedding_size, 256, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = x.reshape(-1, self.height, self.width, self.embedding_size)
        x = x.permute(0, 3, 1, 2)
        x = self.segmentation_conv(x)
        return x


class DinoSemanticSegmentation(torch.nn.Module):
    def __init__(
        self,
        *,
        embedding_dim=768,
        num_classes=1,
        token_w=32,  # the size of the input image should be 448 by 448
        token_h=32,
        pretrained_model_name="facebook/dinov2-base",
    ):
        super(DinoSemanticSegmentation, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.linear_classifiers = ConvToken()
        # self.linear_classifiers = LinearClassifierToken(
        #     embedding_dim, num_classes, token_w, token_h
        # )

    def forward(self, image):
        outputs = self.transformer(image)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        logits = self.linear_classifiers(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=image.shape[2:], mode="bilinear", align_corners=False
        )

        return logits


class LightningDinoClassifier(L.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        num_classes=10,
        scheduler_kwargs={},
        optimizer_kwargs={},
    ):
        super().__init__()
        self.model = DinoClassifier(num_classes=num_classes)
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


class LightningDinoSegmentation(L.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        num_classes=21,
        scheduler_kwargs={},
        optimizer_kwargs={},
    ):
        super().__init__()
        self.model = DinoSemanticSegmentation()
        self.learning_rate = learning_rate
        self.validation_losses = []
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.miou = MeanIoU(num_classes=self.num_classes).to(self.device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}
        ]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        # raise ValueError(f"masks.shape: {masks.shape}; logits.shape: {logits.shape}")
        # print(masks.shape)
        # print(logits.shape)

        loss = self.criterion(logits, masks)
        miou = self._calculate_miou(logits, masks)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_miou", miou, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        loss = self.criterion(logits, masks)
        miou = self._calculate_miou(logits, masks)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_miou", miou, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        miou = self._calculate_miou(logits, masks)
        self.log("test_miou", miou, on_epoch=True, on_step=False)

    def _calculate_miou(self, logits, masks):
        return mean_iou(
            torch.argmax(logits, dim=1), masks, num_classes=self.num_classes
        )
