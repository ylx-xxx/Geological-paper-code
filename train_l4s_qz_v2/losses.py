import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, ignore_index=255, gamma=2.0, class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

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

        ce = F.cross_entropy(
            logits,
            target_safe,
            reduction="none",
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
        )

        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        loss = loss[valid]

        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        ignore_index=255,
        alpha=0.35,
        beta=0.65,
        eps=1e-6,
        class_weights=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
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

        tp = torch.sum(probs * onehot, dims)
        fp = torch.sum(probs * (1.0 - onehot) * valid, dims)
        fn = torch.sum((1.0 - probs) * onehot, dims)

        tversky = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )

        loss_per_class = 1.0 - tversky

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
        ce_weight=0.4,
        focal_weight=1.0,
        tversky_weight=1.4,
        class_weights=None,
        label_smoothing=0.01,
        focal_gamma=2.0,
        tversky_alpha=0.35,
        tversky_beta=0.65,
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

        self.focal = FocalLoss(
            ignore_index=ignore_index,
            gamma=focal_gamma,
            class_weights=class_weights,
        )

        self.tversky = TverskyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=tversky_alpha,
            beta=tversky_beta,
            class_weights=class_weights,
        )

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        focal_loss = self.focal(logits, target)
        tversky_loss = self.tversky(logits, target)

        loss = (
            self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
        )

        return loss
