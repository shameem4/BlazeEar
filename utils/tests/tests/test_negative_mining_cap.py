"""The per-prior ceiling on hard-negative mining.

Mining takes a global top-k of background logits, so the budget follows how
confusing a prior is rather than how many anchors it owns. Measured on the
shipped weights, one prior holding 7.1% of the anchors and receiving 1.0% of
the positives took 53.5% of the whole negative budget. These pin the fix.
"""
import collections

import torch

from loss_functions import BlazeEarDetectionLoss
from utils.anchor_utils import get_anchors
from utils.config import DEFAULT_INPUT_SIZE

BUDGET = 112


def _scores(batch=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, get_anchors(input_size=DEFAULT_INPUT_SIZE).shape[0]) * 3


def _shares(loss, scores, selected, anchors):
    """Per-prior share of the selected negatives, recovered by value."""
    groups = loss._groups_for(anchors)
    owner = torch.empty(anchors.shape[0], dtype=torch.long)
    for index, members in enumerate(groups):
        owner[members] = index
    counts = collections.Counter()
    for row in range(scores.shape[0]):
        chosen = torch.isin(scores[row], selected[row])
        for group in owner[chosen].tolist():
            counts[group] += 1
    total = max(sum(counts.values()), 1)
    return {group: count / total for group, count in counts.items()}


def test_cap_disabled_is_the_previous_global_topk():
    """1.0 must restore the exact prior behaviour, so the cap is reviewable
    as a change of policy rather than a change of arithmetic."""
    anchors = get_anchors(input_size=DEFAULT_INPUT_SIZE)
    scores = _scores()
    expected = torch.sort(scores, dim=-1, descending=True).values[:, :BUDGET]
    loss = BlazeEarDetectionLoss(hard_negative_prior_cap=1.0)
    assert torch.equal(expected, loss._mine_negatives(scores, BUDGET, anchors))


def test_cap_binds_without_changing_the_budget():
    """The ceiling holds, and the TOTAL is untouched -- otherwise the
    background loss would change scale and the cap could not be attributed."""
    anchors = get_anchors(input_size=DEFAULT_INPUT_SIZE)
    scores = _scores()
    loss = BlazeEarDetectionLoss(hard_negative_prior_cap=0.25)
    selected = loss._mine_negatives(scores, BUDGET, anchors)
    assert selected.shape == (scores.shape[0], BUDGET)
    worst = max(_shares(loss, scores, selected, anchors).values())
    assert worst <= 0.27, f'a prior took {100 * worst:.1f}% of the budget'


def test_cap_lowers_the_worst_share():
    anchors = get_anchors(input_size=DEFAULT_INPUT_SIZE)
    scores = _scores()
    uncapped = BlazeEarDetectionLoss(hard_negative_prior_cap=1.0)
    capped = BlazeEarDetectionLoss(hard_negative_prior_cap=0.25)
    before = max(_shares(uncapped, scores,
                         uncapped._mine_negatives(scores, BUDGET, anchors),
                         anchors).values())
    after = max(_shares(capped, scores,
                        capped._mine_negatives(scores, BUDGET, anchors),
                        anchors).values())
    assert after <= before


def test_cap_imposes_no_floor():
    """A ceiling, not a quota. A prior whose logits are not competitive must
    still contribute nothing -- a uniform quota would hand the priors carrying
    most of the positives around a hundred times their current background
    pressure, which is how a rebalance becomes a regression.
    """
    anchors = get_anchors(input_size=DEFAULT_INPUT_SIZE)
    scores = _scores()
    loss = BlazeEarDetectionLoss(hard_negative_prior_cap=0.25)
    # Make one prior's anchors unambiguously the softest in the batch.
    quiet = loss._groups_for(anchors)[4]
    scores = scores.clone()
    scores[:, quiet] = torch.finfo(scores.dtype).min / 2
    selected = loss._mine_negatives(scores, BUDGET, anchors)
    shares = _shares(loss, scores, selected, anchors)
    assert shares.get(4, 0.0) == 0.0, 'the cap manufactured negatives'


def test_small_budget_still_fills():
    """min_negatives_per_image is 10, well under one prior's cap share, so the
    pool has to be floored at budget/groups or the budget comes back short."""
    anchors = get_anchors(input_size=DEFAULT_INPUT_SIZE)
    scores = _scores()
    loss = BlazeEarDetectionLoss(hard_negative_prior_cap=0.05)
    for budget in (1, 10, 37):
        selected = loss._mine_negatives(scores, budget, anchors)
        assert selected.shape[1] == budget, f'budget {budget} came back short'
