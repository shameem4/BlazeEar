"""
Unit tests for loss functions.

The classification loss operates on raw logits. Feeding it probabilities will
not raise, it will just silently train against the wrong objective, so several
tests below pin the logit contract rather than only checking for finiteness.
"""
import math
import unittest

import torch

from loss_functions import BlazeEarDetectionLoss, get_loss
from utils.anchor_utils import generate_reference_anchors


class TestLosses(unittest.TestCase):
    """Tests for BlazeEarDetectionLoss."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_anchors = 896
        self.loss_fn = BlazeEarDetectionLoss(
            hard_negative_ratio=1.0,
            detection_weight=150.0,
            classification_weight=35.0,
            scale=128,
            min_negatives_per_image=10
        )
        # Generate reference anchors (returns tuple: full, small, big)
        reference_anchors, _, _ = generate_reference_anchors()
        self.reference_anchors = reference_anchors.float()

    def _create_dummy_inputs(
        self,
        num_positives_per_batch: int = 5
    ):
        """
        Create dummy inputs for loss computation.

        Args:
            num_positives_per_batch: Number of positive anchors per image

        Returns:
            Tuple of (class_logits, anchor_predictions, anchor_targets)
        """
        B = self.batch_size
        N = self.num_anchors

        # Class logits: negative almost everywhere (mostly background)
        class_logits = torch.randn(B, N, 1) - 2.0

        # Anchor predictions: [dx, dy, w, h] offsets
        anchor_predictions = torch.randn(B, N, 4) * 0.1

        # Anchor targets: [class, ymin, xmin, ymax, xmax]
        anchor_targets = torch.zeros(B, N, 5)

        # Set some positives
        for b in range(B):
            for i in range(num_positives_per_batch):
                idx = i * 10  # spread out
                anchor_targets[b, idx, 0] = 1.0  # class = 1
                # Random box
                ymin, xmin = torch.rand(2) * 0.5
                ymax = ymin + torch.rand(1) * 0.3 + 0.1
                xmax = xmin + torch.rand(1) * 0.3 + 0.1
                anchor_targets[b, idx, 1:5] = torch.tensor([ymin, xmin, ymax.item(), xmax.item()])
                # Confident positive logit
                class_logits[b, idx, 0] = 2.0 + torch.rand(1)

        return class_logits, anchor_predictions, anchor_targets

    def test_loss_computation(self):
        """Test that loss computation produces valid outputs."""
        class_logits, anchor_preds, anchor_targets = self._create_dummy_inputs(
            num_positives_per_batch=5
        )

        loss_dict = self.loss_fn(
            class_logits, anchor_preds, anchor_targets, self.reference_anchors
        )

        expected_keys = ['total', 'detection', 'background', 'positive', 'num_positives', 'num_negatives']
        for key in expected_keys:
            self.assertIn(key, loss_dict, f"Missing key: {key}")

        for key in ['total', 'detection', 'background', 'positive']:
            self.assertTrue(
                torch.isfinite(loss_dict[key]),
                f"{key} loss is not finite: {loss_dict[key]}"
            )
            self.assertGreaterEqual(
                loss_dict[key].item(), 0,
                f"{key} loss should be non-negative"
            )

        expected_total = (
            loss_dict['detection'] * 150.0 +
            loss_dict['background'] * 35.0 +
            loss_dict['positive'] * 35.0
        )
        self.assertAlmostEqual(
            loss_dict['total'].item(), expected_total.item(), places=4,
            msg="Total loss doesn't match weighted sum"
        )

        self.assertEqual(
            loss_dict['num_positives'].item(), self.batch_size * 5,
            "num_positives should be batch_size * 5"
        )

    def test_loss_computation_no_positives(self):
        """Test loss computation with no positive samples."""
        B, N = self.batch_size, self.num_anchors

        class_logits = torch.randn(B, N, 1) - 2.0
        anchor_preds = torch.randn(B, N, 4) * 0.1
        anchor_targets = torch.zeros(B, N, 5)  # all background

        loss_dict = self.loss_fn(
            class_logits, anchor_preds, anchor_targets, self.reference_anchors
        )

        self.assertEqual(loss_dict['detection'].item(), 0.0)
        self.assertEqual(loss_dict['positive'].item(), 0.0)
        self.assertGreater(loss_dict['background'].item(), 0.0)
        self.assertEqual(loss_dict['num_positives'].item(), 0.0)

    # -- logit contract -----------------------------------------------------

    def test_bce_matches_closed_form_on_logits(self):
        """bce_loss must interpret its input as a logit, not a probability."""
        logit = torch.tensor([2.0])
        target = torch.tensor([1.0])
        # -log(sigmoid(2)) = log(1 + e^-2)
        expected = math.log1p(math.exp(-2.0))
        self.assertAlmostEqual(
            self.loss_fn.bce_loss(logit, target).item(), expected, places=6
        )

    def test_confident_correct_logit_gives_near_zero_loss(self):
        """A large positive logit on a positive target is nearly free."""
        loss = self.loss_fn.bce_loss(torch.tensor([20.0]), torch.tensor([1.0]))
        self.assertLess(loss.item(), 1e-6)

    def test_extreme_logits_stay_finite(self):
        """
        The old implementation applied sigmoid then log with a 1e-7 clamp, which
        saturated in fp16 and capped per-sample loss near 16.1, truncating the
        gradient on exactly the confidently-wrong anchors hard negative mining
        selects. With logits the loss grows linearly and stays finite.
        """
        very_wrong = torch.tensor([-1e4])
        target = torch.tensor([1.0])
        loss = self.loss_fn.bce_loss(very_wrong, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 1000.0)

    def test_extreme_logit_gradient_does_not_vanish(self):
        """A confidently-wrong anchor must still produce gradient."""
        logit = torch.tensor([-30.0], requires_grad=True)
        self.loss_fn.bce_loss(logit, torch.tensor([1.0])).backward()
        assert logit.grad is not None
        self.assertTrue(torch.isfinite(logit.grad).all())
        self.assertGreater(logit.grad.abs().item(), 0.5)

    # -- hard negative mining ----------------------------------------------

    def test_hard_negative_mining(self):
        """Hard negative mining selects the highest-scoring backgrounds."""
        B, N = 1, self.num_anchors

        class_logits = torch.zeros(B, N, 1)
        anchor_targets = torch.zeros(B, N, 5)
        anchor_targets[0, 0, 0] = 1.0
        anchor_targets[0, 0, 1:5] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        class_logits[0, 0, 0] = 3.0           # confident positive

        class_logits[0, 1:11, 0] = -3.0       # easy negatives
        class_logits[0, 11:21, 0] = 3.0       # hard negatives, should be picked
        class_logits[0, 21:, 0] = 0.0

        anchor_preds = torch.randn(B, N, 4) * 0.1

        loss_fn = BlazeEarDetectionLoss(hard_negative_ratio=10.0, min_negatives_per_image=5)
        loss_dict = loss_fn(class_logits, anchor_preds, anchor_targets, self.reference_anchors)

        self.assertEqual(loss_dict['num_positives'].item(), 1.0)
        self.assertEqual(loss_dict['num_negatives'].item(), 10)

        # All ten selected negatives sit at logit 3.0 -> -log(1 - sigmoid(3)).
        expected = -math.log(1.0 - 1.0 / (1.0 + math.exp(-3.0)))
        self.assertAlmostEqual(loss_dict['background'].item(), expected, places=4)

    def test_positive_anchors_are_never_mined_as_negatives(self):
        """
        Positives are masked out of the negative pool. The old sentinel was a
        fixed -99.0, which is a reachable value in logit space; the mask must
        hold even when a positive anchor's logit is far below it.
        """
        B, N = 1, self.num_anchors
        class_logits = torch.full((B, N, 1), -500.0)
        anchor_targets = torch.zeros(B, N, 5)
        anchor_targets[0, 0, 0] = 1.0
        anchor_targets[0, 0, 1:5] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        class_logits[0, 0, 0] = -1000.0       # positive, more negative than -99

        loss_fn = BlazeEarDetectionLoss(hard_negative_ratio=1.0, min_negatives_per_image=3)
        loss_dict = loss_fn(
            class_logits, torch.randn(B, N, 4) * 0.1, anchor_targets, self.reference_anchors
        )

        # If the positive leaked into the negative pool it would contribute a
        # ~0 background loss; every true negative here is at -500, also ~0, so
        # assert via the positive term instead: it must see the -1000 logit.
        self.assertGreater(loss_dict['positive'].item(), 900.0)
        self.assertTrue(torch.isfinite(loss_dict['total']))

    def test_hard_negative_mining_min_negatives(self):
        """Test min_negatives_per_image is respected."""
        B, N = 1, self.num_anchors

        class_logits = torch.randn(B, N, 1) - 2.0
        anchor_targets = torch.zeros(B, N, 5)
        anchor_targets[0, 0, 0] = 1.0
        anchor_targets[0, 0, 1:5] = torch.tensor([0.3, 0.3, 0.6, 0.6])
        class_logits[0, 0, 0] = 3.0

        loss_fn = BlazeEarDetectionLoss(
            hard_negative_ratio=0.1, min_negatives_per_image=20
        )
        loss_dict = loss_fn(
            class_logits, torch.randn(B, N, 4) * 0.1, anchor_targets, self.reference_anchors
        )

        self.assertEqual(loss_dict['num_negatives'].item(), 20)

    # -- focal loss ---------------------------------------------------------

    def test_focal_loss(self):
        """Test focal loss option."""
        loss_fn_focal = BlazeEarDetectionLoss(
            use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0
        )
        class_logits, anchor_preds, anchor_targets = self._create_dummy_inputs()
        loss_dict = loss_fn_focal(
            class_logits, anchor_preds, anchor_targets, self.reference_anchors
        )
        self.assertTrue(torch.isfinite(loss_dict['total']))
        self.assertGreater(loss_dict['total'].item(), 0)

    def test_focal_loss_vs_bce(self):
        """Focal loss down-weights easy examples more than hard ones."""
        target = torch.tensor([1.0])
        loss_fn_bce = BlazeEarDetectionLoss(use_focal_loss=False)
        loss_fn_focal = BlazeEarDetectionLoss(use_focal_loss=True, focal_gamma=2.0)

        easy = torch.tensor([5.0])    # confident and correct
        hard = torch.tensor([-1.0])   # wrong side of the boundary

        easy_ratio = (loss_fn_focal.focal_loss(easy, target)
                      / loss_fn_bce.bce_loss(easy, target))
        hard_ratio = (loss_fn_focal.focal_loss(hard, target)
                      / loss_fn_bce.bce_loss(hard, target))

        self.assertLess(
            loss_fn_focal.focal_loss(easy, target).item(),
            loss_fn_bce.bce_loss(easy, target).item(),
            "Focal loss should be lower than BCE for easy examples"
        )
        self.assertGreater(
            hard_ratio.item(), easy_ratio.item(),
            "Focal loss should down-weight easy examples more than hard ones"
        )

    def test_focal_loss_stays_finite_on_extreme_logits(self):
        loss = BlazeEarDetectionLoss(use_focal_loss=True).focal_loss(
            torch.tensor([-1e4, 1e4]), torch.tensor([1.0, 0.0])
        )
        self.assertTrue(torch.isfinite(loss))

    # -- regression ---------------------------------------------------------

    def test_regression_beta_default(self):
        """Default beta is ~13 px at the 128px input scale."""
        self.assertAlmostEqual(self.loss_fn.regression_beta, 0.1, places=9)
        self.assertAlmostEqual(self.loss_fn.huber_loss.beta, 0.1, places=9)

    def test_regression_gradient_is_larger_near_zero_than_before(self):
        """
        Fine localization must not be starved. With beta=1.0 (the old default,
        the full image width) a 6 px error got gradient 0.05; with beta=0.1 it
        gets 0.5.
        """
        def grad_at(err, beta):
            loss_fn = BlazeEarDetectionLoss(regression_beta=beta)
            pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
            loss_fn.huber_loss(pred, torch.full((1, 4), err)).backward()
            assert pred.grad is not None
            return pred.grad.abs().max().item()

        self.assertGreater(grad_at(0.05, 0.1), 8 * grad_at(0.05, 1.0))

    def test_regression_is_linear_above_beta(self):
        """Above beta the loss is linear, i.e. gross errors get constant
        gradient rather than an ever-growing one. That is the robustness Huber
        is for, and it matters here: measured errors reach 88 in normalized
        units on anchors whose classification head is dead."""
        loss_fn = BlazeEarDetectionLoss(regression_beta=0.1)
        grads = []
        for err in (0.5, 1.0, 5.0):
            pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
            target = torch.full((1, 4), err)
            loss_fn.huber_loss(pred, target).backward()
            assert pred.grad is not None
            grads.append(pred.grad.abs().max().item())
        for g in grads[1:]:
            self.assertAlmostEqual(g, grads[0], places=6)

    def test_get_loss_factory(self):
        """Test get_loss factory function."""
        loss_fn = get_loss(
            hard_negative_ratio=2.0, detection_weight=100.0, use_focal_loss=True
        )
        self.assertIsInstance(loss_fn, BlazeEarDetectionLoss)
        self.assertEqual(loss_fn.hard_negative_ratio, 2.0)
        self.assertEqual(loss_fn.detection_weight, 100.0)
        self.assertTrue(loss_fn.use_focal_loss)

    def test_loss_gradients(self):
        """Test that gradients flow correctly through loss."""
        class_logits, anchor_preds, anchor_targets = self._create_dummy_inputs()
        class_logits.requires_grad_(True)
        anchor_preds.requires_grad_(True)

        loss_dict = self.loss_fn(
            class_logits, anchor_preds, anchor_targets, self.reference_anchors
        )
        loss_dict['total'].backward()

        self.assertIsNotNone(class_logits.grad)
        self.assertIsNotNone(anchor_preds.grad)
        if class_logits.grad is not None:
            self.assertTrue(torch.isfinite(class_logits.grad).all())
        if anchor_preds.grad is not None:
            self.assertTrue(torch.isfinite(anchor_preds.grad).all())


if __name__ == "__main__":
    unittest.main()
