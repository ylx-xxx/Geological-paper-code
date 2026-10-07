"""Numerical checks of the manuscript equations against the actual loss module."""

import unittest
import torch
from train_l4s_qz.losses import CombinedLoss


class LossFormulaTests(unittest.TestCase):
    def test_smoothed_weighted_ce_uses_true_label_weight_denominator(self):
        logits = torch.tensor(
            [[[[0.2, -0.8], [1.3, 0.7]], [[1.1, 0.4], [-0.2, 0.1]]]],
            dtype=torch.float64,
        )
        target = torch.tensor([[[0, 1], [255, 1]]])
        alpha = torch.tensor([0.5, 1.7], dtype=torch.float64)
        loss = CombinedLoss(class_weights=alpha.tolist(), label_smoothing=0.02).double()
        valid = target != 255
        selected = logits.permute(0, 2, 3, 1)[valid]
        labels = target[valid]
        onehot = torch.nn.functional.one_hot(labels, 2).double()
        q = 0.98 * onehot + 0.02 / 2
        expected = -(q * alpha * selected.log_softmax(-1)).sum() / alpha[labels].sum()
        torch.testing.assert_close(
            loss.ce(logits, target), expected, rtol=1e-7, atol=1e-8
        )
        pixel_mean = -(q * alpha * selected.log_softmax(-1)).sum() / labels.numel()
        self.assertGreater(abs(float(expected - pixel_mean)), 0.01)

    def test_dice_weights_both_classes_and_excludes_ignored_pixels(self):
        logits = torch.tensor(
            [[[[0.2, -0.8], [1.3, 0.7]], [[1.1, 0.4], [-0.2, 0.1]]]],
            dtype=torch.float64,
        )
        target = torch.tensor([[[0, 1], [255, 1]]])
        alpha = torch.tensor([0.5, 1.7], dtype=torch.float64)
        loss = CombinedLoss(class_weights=alpha.tolist()).double()
        valid = target != 255
        prob = logits.softmax(1).permute(0, 2, 3, 1)[valid]
        y = torch.nn.functional.one_hot(target[valid], 2).double()
        dice = (2 * (prob * y).sum(0) + 1e-6) / (prob.sum(0) + y.sum(0) + 1e-6)
        expected = ((1 - dice) * alpha / (alpha.mean() + 1e-6)).mean()
        torch.testing.assert_close(
            loss.dice(logits, target), expected, rtol=1e-7, atol=1e-8
        )
        changed = logits.clone()
        changed[:, :, 1, 0] = torch.tensor([900.0, -900.0])
        torch.testing.assert_close(loss(logits, target), loss(changed, target))


if __name__ == "__main__":
    unittest.main()
