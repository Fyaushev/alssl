from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from dino_finetune import DINOV2EncoderLoRA
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import dice
from torchmetrics.segmentation import MeanIoU


class DinoLoRaSegmentation(nn.Module):
    def __init__(self, num_classes=10, ):
        super(DinoLoRaSegmentation, self).__init__()
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_lora = DINOV2EncoderLoRA(
            encoder=encoder,
            r=3, # These are the same settings used in training
            emb_dim=768, # The base ViT embedding dim
            img_dim=(280, 280), # For ease of use rescaling to a valid patch dimension 
            n_classes=num_classes, 
            use_fpn=True,
            use_lora=True,
        )
        self.inter_layers = 4 # same as in the orig code

    def forward(self, x):
        feature = self.dino_lora.encoder.get_intermediate_layers(
            x, n=self.inter_layers, reshape=True
        )
        logits = self.dino_lora.decoder(feature)

        # features = self.dino_lora.encoder.forward_features(x)
        # patch_embeddings = features["x_norm_patchtokens"]
        # logits = self.dino_lora.decoder(patch_embeddings)
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, feature[-1].mean(axis=(-2, -1))


class LightningDinoLoRaSegmentation(L.LightningModule):
    def __init__(
        self,
        root='',
        learning_rate=0.001,
        num_classes=10,
        blocks_to_retrain=0,
        scheduler_kwargs={},
        optimizer_kwargs={},
        include_param_loss: bool = False,
        param_loss_beta: float = 1,
        binary: bool = False,
        *args
    ):
        super().__init__()
        self.model = DinoLoRaSegmentation(num_classes=num_classes, )
        self.include_param_loss = include_param_loss
        self.param_loss_beta = param_loss_beta
        self.learning_rate = learning_rate
        self.validation_losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice_loss_fn = DiceLoss()
        # self.miou = MeanIoU(num_classes=num_classes, per_class=True, include_background=True, input_format='index')
        self.num_classes = num_classes
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.binary = binary

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
        labels = labels.long()
        logits, embeddings = self(images)

        # labels = self.resize_masks(labels, logits)
        if not self.binary:
            loss = self.criterion(logits, labels)
        else:
            pred_probs = nn.functional.softmax(logits, dim=1)
            loss = self.dice_loss_fn(pred_probs[:, 1, :, :], labels.float())

        self.log("train_loss_criterion", loss, prog_bar=True, on_epoch=True, on_step=False)

        miou = self._calculate_miou(logits, labels)
        if self.binary:
            dice = self._calculate_dice(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_miou", miou, prog_bar=True, on_epoch=True, on_step=False)
        if self.binary:
            self.log("train_dice", dice, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        logits, embeddings = self(images)

        labels = self.resize_masks(labels, logits)

        loss = self.criterion(logits, labels)
        miou = self._calculate_miou(logits, labels)
        if self.binary:
            dice = self._calculate_dice(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_miou", miou, prog_bar=True, on_epoch=True, on_step=False)
        if self.binary:
            self.log("val_dice", dice, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        logits, embeddings = self(images)

        labels = self.resize_masks(labels, logits)

        miou = self._calculate_miou(logits, labels)
        if self.binary:
            dice = self._calculate_dice(logits, labels)

        self.log("test_miou", miou, on_epoch=True, on_step=False)
        if self.binary:
            self.log("test_dice", dice, on_epoch=True, on_step=False)

    def _calculate_dice(self, logits, masks):
        return dice(
            logits,
            # torch.argmax(logits, dim=1), 
            masks, 
            num_classes=self.num_classes,
        )
    
    def _calculate_miou(self, logits, masks):
        return compute_iou_metric(logits, masks, ignore_index=-1)
        # return self.miou(
        #     torch.argmax(logits, dim=1),
        #     masks, 
        # ).mean()


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
    
def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int | None] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """

    y_hat = torch.argmax(y_hat, dim=1)
    y_hat = y_hat.int()
    y = y.int()

    if ignore_index is not None:
        mask = y != ignore_index
        y_hat = y_hat * mask
        y = y * mask

    intersection = (y_hat & y).float().sum((1, 2))
    union = (y_hat | y).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()