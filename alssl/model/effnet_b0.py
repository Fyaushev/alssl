import lightning as L
import timm
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics.functional import accuracy, average_precision
from torchmetrics.segmentation import MeanIoU
from torchvision.models import efficientnet_b0
from transformers import AutoModel

from ..metric import mean_iou


class EffNetB0Classifier(nn.Module):
    def __init__(self, num_classes=10, blocks_to_retrain=1):
        super(EffNetB0Classifier, self).__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(1280, num_classes, bias=True) # same as in the orig model

        for param in self.backbone.parameters():
            param.requires_grad_(False)
        
        # unfreeze blocks
        count = 0
        for name, block in self.backbone.blocks.named_children():
            if count >= (len(self.backbone.blocks) - blocks_to_retrain):
                print(f'Unfreeze block {name}')
                for pname, params in block.named_parameters():
                    if 'bn' not in pname:
                        params.requires_grad = True
            else:
                print(f'Keep block {name} frozen')
            count += 1
        
        if blocks_to_retrain > 0:
            # unfreeze last layers
            for name, child in self.backbone.named_children():
                if name in ['conv_head', 'act2', 'global_pool', 'classifier']:
                    print(f'Unfreeze block {name}')
                    for params in child.parameters():
                        params.requires_grad = True
            
        self.classifier.requires_grad = True

    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


class LightningEffNetB0Classifier(L.LightningModule):
    def __init__(
        self,
        root='',
        learning_rate=0.001,
        num_classes=10,
        blocks_to_retrain=1,
        scheduler_kwargs={},
        optimizer_kwargs={},
        include_param_loss: bool = True,
        param_loss_beta: float = 0.01,
        *args
    ):
        super().__init__()
        self.model = EffNetB0Classifier(num_classes=num_classes, blocks_to_retrain=blocks_to_retrain)
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

        # scheduler = MultiStepLR(optimizer, **self.scheduler_kwargs)

        # scheduler = ReduceLROnPlateau(
        #     optimizer, **self.scheduler_kwargs
        # )
        # "monitor": "train_loss"
        self.scheduler_kwargs["max_lr"] = self.learning_rate
        scheduler = OneCycleLR(optimizer, **self.scheduler_kwargs)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "train_loss"}
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
        auprc = self._calculate_average_precision(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_auprc", auprc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        loss = self.criterion(logits, labels)
        acc = self._calculate_accuracy(logits, labels)
        auprc = self._calculate_average_precision(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_auprc", auprc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        acc = self._calculate_accuracy(logits, labels)
        auprc = self._calculate_average_precision(logits, labels)
        self.log("test_acc", acc, on_epoch=True, on_step=False)
        self.log("test_auprc", auprc, on_epoch=True, on_step=False)

    def _calculate_accuracy(self, logits, labels):
        return accuracy(
            torch.argmax(logits, dim=1),
            labels,
            task="multiclass",
            num_classes=self.num_classes,
        )
    
    def _calculate_average_precision(self, logits, labels):
        return average_precision(
            logits,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
        )

