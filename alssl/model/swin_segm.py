import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import dice
from torchmetrics.segmentation import MeanIoU
from transformers import SwinModel


class BirefNetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BirefNetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)  
        return x

class SwinBirefNet(nn.Module):
    def __init__(self, num_classes=2, blocks_to_retrain=0):
        super(SwinBirefNet, self).__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")

        self.decoder = BirefNetDecoder(in_channels=1024, num_classes=num_classes)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        count = 0
        for name, block in self.backbone.encoder.layers.named_children():
            if count >= (len(self.backbone.encoder.layers) - blocks_to_retrain):
                print(f'Unfreeze block {name}')
                for pname, params in block.named_parameters():
                    if 'bn' not in pname:
                        params.requires_grad = True
            else:
                print(f'Keep block {name} frozen')
            count += 1
        
        for param in self.decoder.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state  
        B, L, C = features.shape
        H = W = int(L**0.5) 
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 1024, 7, 7)

        out = self.decoder(features) 
        return out, features
    

class LightningSwinSegmentation(L.LightningModule):
    def __init__(
        self,
        root='',
        learning_rate=0.001,
        num_classes=10,
        blocks_to_retrain=0,
        scheduler_kwargs={},
        optimizer_kwargs={},
        include_param_loss: bool = True,
        param_loss_beta: float = 1,
        *args
    ):
        super().__init__()
        self.model = SwinBirefNet(num_classes=num_classes, blocks_to_retrain=blocks_to_retrain)
        self.include_param_loss = include_param_loss
        self.param_loss_beta = param_loss_beta
        self.source_weight = {}
        for name, param in self.model.backbone.named_parameters():
            self.source_weight[name] = param.detach()
        self.learning_rate = learning_rate
        self.validation_losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dice_loss_fn = DiceLoss()
        self.miou = MeanIoU(num_classes=num_classes, per_class=True, include_background=False, input_format='index')
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
        scheduler = OneCycleLR(optimizer, **self.scheduler_kwargs)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "train_loss"}
        ]
    
    def resize_masks(self, masks, outputs):
        masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='nearest')
        masks = masks.squeeze(1).long()  # [B, 1, H, W] -> [B, H, W]
        return masks

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        labels = self.resize_masks(labels, logits)

        loss = self.criterion(logits, labels)
        pred_probs = nn.functional.softmax(logits, dim=1)
        dice_loss_val = self.dice_loss_fn(pred_probs[:, 1, :, :], labels.float())
        loss = dice_loss_val

        self.log("train_loss_criterion", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        # code from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/delta.py
        if self.include_param_loss:
            param_loss = 0.0
            for name, param in self.model.backbone.named_parameters():
                param_loss += 0.5 * torch.norm(param - self.source_weight[name].to(param)) ** 2
            loss += param_loss * self.param_loss_beta
            self.log("param_loss", param_loss, prog_bar=True, on_epoch=True, on_step=False)

        miou = self._calculate_miou(logits, labels)
        dice = self._calculate_dice(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_miou", miou, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_dice", dice, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        labels = self.resize_masks(labels, logits)

        loss = self.criterion(logits, labels)
        miou = self._calculate_miou(logits, labels)
        dice = self._calculate_dice(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_miou", miou, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_dice", dice, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, embeddings = self(images)

        labels = self.resize_masks(labels, logits)

        miou = self._calculate_miou(logits, labels)
        dice = self._calculate_dice(logits, labels)

        self.log("test_miou", miou, on_epoch=True, on_step=False)
        self.log("test_dice", dice, on_epoch=True, on_step=False)

    def _calculate_dice(self, logits, masks):
        return dice(
            logits,
            # torch.argmax(logits, dim=1), 
            masks, 
            num_classes=self.num_classes,
        )
    
    def _calculate_miou(self, logits, masks):
        return self.miou(
            torch.argmax(logits, dim=1),
            masks, 
        ).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        batch_size = logits.size(0)
        logits = logits.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        intersection = (logits * targets).sum(-1)
        dice = (2. * intersection + self.eps) / (logits.sum(-1) + targets.sum(-1) + self.eps)
        return 1 - dice.mean()