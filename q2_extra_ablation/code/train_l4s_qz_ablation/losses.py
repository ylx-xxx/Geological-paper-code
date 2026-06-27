import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=255, eps=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0

        onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        onehot = onehot * valid

        dims = (0, 2, 3)
        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2 * inter + self.eps) / (union + self.eps)
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
    ):
        super().__init__()

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights_tensor = None

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights_tensor,
            label_smoothing=label_smoothing,
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
