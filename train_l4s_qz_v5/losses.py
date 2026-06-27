import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        class_weights=None,
        label_smoothing=0.02,
        topk_ratio=0.25,
        min_kept=4096,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.topk_ratio = topk_ratio
        self.min_kept = min_kept

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits, target):
        valid = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid] = 0

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        ce = F.cross_entropy(
            logits,
            target_safe,
            reduction="none",
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

        ce = ce[valid]

        if ce.numel() == 0:
            return logits.sum() * 0.0

        k = int(ce.numel() * self.topk_ratio)
        k = max(k, self.min_kept)
        k = min(k, ce.numel())

        hard_loss, _ = torch.topk(ce, k=k, largest=True)

        return hard_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=255, eps=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0

        onehot = F.one_hot(
            target_safe,
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        onehot = onehot * valid

        dims = (0, 2, 3)

        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2.0 * inter + self.eps) / (union + self.eps)
        loss_per_class = 1.0 - dice

        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            w = w / (w.mean() + self.eps)
            loss_per_class = loss_per_class * w

        return loss_per_class.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=None,
        label_smoothing=0.02,
        ohem_topk_ratio=0.25,
    ):
        super().__init__()

        self.ohem_ce = OhemCrossEntropyLoss(
            ignore_index=ignore_index,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            topk_ratio=ohem_topk_ratio,
            min_kept=4096,
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce_loss = self.ohem_ce(logits, target)
        dice_loss = self.dice(logits, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
