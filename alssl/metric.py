import numpy as np
import torch


def mean_iou(y_true, y_pred, num_classes=2):
    def iou(y_true, y_pred, num_classes):
        ious = []
        for c in range(1, num_classes):  # class 0 is often the background; skipping it
            TP = torch.sum((y_true == c) & (y_pred == c))
            FP = torch.sum((y_true != c) & (y_pred == c))
            FN = torch.sum((y_true == c) & (y_pred != c))

            numerator = TP
            denominator = TP + FP + FN

            # Skip empty classes
            if denominator == 0:
                continue

            iou = torch.divide(numerator, denominator + 1e-12)
            ious.append(iou)

        return torch.mean(torch.tensor(ious)) if ious else torch.tensor(0.0)

    batch = y_true.shape[0]
    y_true = y_true.view(batch, -1)
    y_pred = y_pred.view(batch, -1)

    score = torch.tensor(
        [iou(y_true[idx], y_pred[idx], num_classes) for idx in range(batch)]
    )
    return torch.mean(score).item()


def iou_numpy(y_true, y_pred, num_classes=2):
    ious = []
    for c in range(1, num_classes):  # class 0 is often the background; skipping it
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        numerator = TP
        denominator = TP + FP + FN

        # Skip empty classes
        if denominator == 0:
            continue

        iou = np.divide(numerator, denominator + 1e-12)
        ious.append(iou)

    return np.mean(ious) if ious else 0
