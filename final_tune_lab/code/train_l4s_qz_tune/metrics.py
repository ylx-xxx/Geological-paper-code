import torch


class BinarySegMetric:
    def __init__(self, num_classes=2, ignore_index=255):
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
        hist = torch.bincount(
            inds,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.hist += hist.double()

    def compute(self):
        hist = self.hist
        diag = torch.diag(hist)

        total = hist.sum()
        oa = diag.sum() / (total + 1e-10)

        iou = diag / (hist.sum(1) + hist.sum(0) - diag + 1e-10)
        miou = torch.nanmean(iou)

        precision = diag / (hist.sum(0) + 1e-10)
        recall = diag / (hist.sum(1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "OA": oa.item(),
            "mIoU": miou.item(),
            "NonLandslide_IoU": iou[0].item(),
            "Landslide_IoU": iou[1].item(),
            "NonLandslide_F1": f1[0].item(),
            "Landslide_F1": f1[1].item(),
            "NonLandslide_Precision": precision[0].item(),
            "Landslide_Precision": precision[1].item(),
            "NonLandslide_Recall": recall[0].item(),
            "Landslide_Recall": recall[1].item(),
            "hist": hist.numpy().tolist(),
        }
