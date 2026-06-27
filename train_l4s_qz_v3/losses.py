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


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)

    jaccard = 1.0 - intersection / (union + 1e-6)

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def lovasz_softmax_flat(probs, labels, classes="present"):
    if probs.numel() == 0:
        return probs * 0.0

    C = probs.size(1)
    losses = []

    class_to_sum = list(range(C))

    for c in class_to_sum:
        fg = (labels == c).float()

        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        return probs * 0.0

    return torch.mean(torch.stack(losses))


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        probs = probs.permute(0, 2, 3, 1).contiguous()
        probs = probs.view(-1, probs.size(-1))

        target = target.view(-1)
        valid = target != self.ignore_index

        probs = probs[valid]
        target = target[valid]

        return lovasz_softmax_flat(probs, target, classes="present")


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=0.8,
        lovasz_weight=0.6,
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

        self.lovasz = LovaszSoftmaxLoss(
            ignore_index=ignore_index,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        lovasz_loss = self.lovasz(logits, target)

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.lovasz_weight * lovasz_loss
        )
