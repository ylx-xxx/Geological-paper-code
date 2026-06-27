
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

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
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
