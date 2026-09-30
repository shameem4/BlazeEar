"""Tests for the hard-negative budget.

The default ratio was 1.5. With ANCHOR_TOP_K = 3 that is about 2.3 positive
anchors per image, so the budget came out as max(3, 10) = 10 negatives out of
the 894 available, and the model settled at 0.34 background accuracy against
0.42 positive. Every run since passed --hard-negative-ratio 50 by hand, which
made the default a bug with a workaround rather than a fixed bug.
"""
import torch

from loss_functions import BlazeEarDetectionLoss
from utils.anchor_utils import get_anchors
from utils.config import ANCHOR_TOP_K, HARD_NEGATIVE_RATIO, TOTAL_ANCHORS


def make_batch(batch_size=32, positives_per_image=ANCHOR_TOP_K, seed=0):
    """A batch shaped like real training data: a couple of ears per image."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, TOTAL_ANCHORS, 1, generator=generator)
    predictions = torch.randn(batch_size, TOTAL_ANCHORS, 4, generator=generator) * 0.1
    targets = torch.zeros(batch_size, TOTAL_ANCHORS, 5)
    for b in range(batch_size):
        chosen = torch.randperm(TOTAL_ANCHORS, generator=generator)[:positives_per_image]
        targets[b, chosen, 0] = 1.0
        targets[b, chosen, 1:] = torch.tensor([0.2, 0.2, 0.5, 0.4])
    return logits, predictions, targets, get_anchors()


def negatives_mined(ratio, **kwargs):
    loss = BlazeEarDetectionLoss(hard_negative_ratio=ratio)
    logits, predictions, targets, anchors = make_batch(**kwargs)
    out = loss(logits, predictions, targets, anchors)
    return int(out['num_negatives'].item()), int(out['num_positives'].item())


class TestBudget:
    def test_the_default_mines_far_more_than_the_floor(self):
        """
        The failure this guards is quiet: the floor of 10 per image is a
        plausible-looking number, and nothing reports that mining is starved.
        """
        negatives, positives = negatives_mined(HARD_NEGATIVE_RATIO)
        per_image = negatives / 32
        assert per_image > 50, f'only {per_image:.0f} negatives per image'

    def test_the_old_default_would_have_hit_the_floor(self):
        """Documents the bug, so the regression is recognisable if it returns."""
        negatives, _ = negatives_mined(1.5)
        assert negatives / 32 == 10, 'expected the min_negatives floor'

    def test_the_budget_scales_with_the_ratio(self):
        low, _ = negatives_mined(20.0)
        high, _ = negatives_mined(60.0)
        assert high > low

    def test_the_budget_never_exceeds_the_available_negatives(self):
        negatives, positives = negatives_mined(10000.0)
        assert negatives / 32 <= TOTAL_ANCHORS - positives / 32

    def test_config_is_the_single_definition(self):
        import inspect
        signature = inspect.signature(BlazeEarDetectionLoss.__init__)
        assert signature.parameters['hard_negative_ratio'].default == HARD_NEGATIVE_RATIO


class TestBackgroundImagesCoupling:
    """
    The budget comes from the batch's MEAN positive count, so adding
    all-background face crops -- about 30% of them -- also lowers the number
    of negatives mined on the images that do have ears.
    """

    def test_all_background_images_reduce_the_budget_for_every_image(self):
        loss = BlazeEarDetectionLoss(hard_negative_ratio=HARD_NEGATIVE_RATIO)
        logits, predictions, targets, anchors = make_batch()
        full = int(loss(logits, predictions, targets, anchors)['num_negatives'].item())

        # Blank the positives on a third of the images, as a background crop.
        targets_mixed = targets.clone()
        targets_mixed[:11, :, 0] = 0.0
        mixed = int(loss(logits, predictions, targets_mixed, anchors)['num_negatives'].item())
        assert mixed < full

    def test_a_batch_of_pure_background_still_mines_negatives(self):
        """A background image is ground the model must learn to stay silent on."""
        loss = BlazeEarDetectionLoss(hard_negative_ratio=HARD_NEGATIVE_RATIO)
        logits, predictions, targets, anchors = make_batch()
        targets[:, :, 0] = 0.0
        out = loss(logits, predictions, targets, anchors)
        assert int(out['num_positives'].item()) == 0
        assert int(out['num_negatives'].item()) > 0

    def test_a_pure_background_batch_produces_a_finite_loss(self):
        loss = BlazeEarDetectionLoss(hard_negative_ratio=HARD_NEGATIVE_RATIO)
        logits, predictions, targets, anchors = make_batch()
        targets[:, :, 0] = 0.0
        out = loss(logits, predictions, targets, anchors)
        assert torch.isfinite(out['total'])
