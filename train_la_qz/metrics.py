
import torch
import numpy as np


class SegMetric:
    def __init__(self, num_classes=7, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.hist = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target):
        pred = torch.argmax(logits, dim=1)
        pred = pred.detach().cpu()
        target = target.detach().cpu()

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        if target.numel() == 0:
            return

        inds = self.num_classes * target.long() + pred.long()
        hist = torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.hist += hist.double()

    def compute(self):
        hist = self.hist
        diag = torch.diag(hist)
        acc = diag.sum() / (hist.sum() + 1e-10)

        iou = diag / (hist.sum(1) + hist.sum(0) - diag + 1e-10)
        miou = torch.nanmean(iou)

        precision = diag / (hist.sum(0) + 1e-10)
        recall = diag / (hist.sum(1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        mf1 = torch.nanmean(f1)

        return {
            "OA": acc.item(),
            "mIoU": miou.item(),
            "mF1": mf1.item(),
            "class_IoU": iou.numpy().tolist(),
            "class_F1": f1.numpy().tolist(),
        }
