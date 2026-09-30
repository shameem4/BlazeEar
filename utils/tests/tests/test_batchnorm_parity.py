"""
BatchNorm backbone: weight loading and parity with the folded MediaPipe model.

The project's premise is that MediaPipe's inference-only weights, which have
BatchNorm folded into the conv weights, can be unfolded into trainable
BatchNorm layers without changing what the network computes. These tests pin
that, and pin the loading bug that made it untrue in practice.
"""
import math
from pathlib import Path

import pytest
import torch

from blazebase import load_mediapipe_weights
from blazeear import BlazeEar

WEIGHTS = Path('model_weights/blazeface.pth')
requires_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason='MediaPipe weights not available'
)


@pytest.fixture(scope='module')
def sample_input():
    torch.manual_seed(0)
    return torch.randn(4, 3, 128, 128) * 50 + 128


class TestBackboneConstruction:
    def test_batchnorm_is_the_default(self):
        model = BlazeEar()
        norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(norms) == 32, 'default backbone must carry trainable BatchNorm'

    def test_folded_variant_has_no_normalization(self):
        model = BlazeEar(use_batchnorm=False)
        norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert norms == []

    def test_both_variants_share_the_output_contract(self, sample_input):
        for flag in (True, False):
            boxes, scores = BlazeEar(use_batchnorm=flag)(sample_input)
            assert boxes.shape == (4, 896, 16)
            assert scores.shape == (4, 896, 1)


@requires_weights
class TestWeightLoading:
    def test_folded_model_loads_completely(self):
        missing, unexpected = load_mediapipe_weights(
            BlazeEar(use_batchnorm=False), str(WEIGHTS), strict=False
        )
        assert missing == [] and unexpected == []

    def test_batchnorm_model_loads_completely(self):
        """
        Regression test. load_state_dict(strict=False) does not raise on missing
        keys, so the previous `try/except RuntimeError` fallback never ran and
        the BatchNorm backbone silently kept its random initialization while
        reporting success: 160 missing and 64 unexpected keys, unnoticed.
        """
        missing, unexpected = load_mediapipe_weights(
            BlazeEar(use_batchnorm=True), str(WEIGHTS), strict=False
        )
        assert missing == [], f'{len(missing)} weights not loaded, e.g. {missing[:3]}'
        assert unexpected == []

    def test_unloaded_backbone_is_an_error_not_a_warning(self, tmp_path):
        """A checkpoint that cannot populate the backbone must fail loudly."""
        junk = tmp_path / 'junk.pth'
        torch.save({'classifier_8.weight': torch.zeros(2, 88, 1, 1)}, junk)
        with pytest.raises(RuntimeError, match='random initialization'):
            load_mediapipe_weights(BlazeEar(), str(junk), strict=False)


@requires_weights
class TestParity:
    def test_eval_mode_matches_the_folded_model(self, sample_input):
        """
        Unfolding sets BatchNorm to an identity transform (gamma=1, beta=bias,
        mean=0, var=1), so in eval mode the BatchNorm model must reproduce the
        folded model. The residual comes from BatchNorm's eps compounding over
        32 layers, so it is checked relatively.
        """
        folded = BlazeEar(use_batchnorm=False)
        batchnorm = BlazeEar(use_batchnorm=True)
        load_mediapipe_weights(folded, str(WEIGHTS), strict=False)
        load_mediapipe_weights(batchnorm, str(WEIGHTS), strict=False)
        folded.eval()
        batchnorm.eval()

        with torch.no_grad():
            boxes_f, scores_f = folded(sample_input)
            boxes_b, scores_b = batchnorm(sample_input)

        box_rel = (boxes_f - boxes_b).abs().max() / boxes_f.abs().max()
        score_rel = (scores_f - scores_b).abs().max() / scores_f.abs().max()
        assert box_rel < 1e-3, f'box relative error {box_rel:.2e}'
        assert score_rel < 1e-3, f'score relative error {score_rel:.2e}'

    def test_train_mode_uses_batch_statistics(self, sample_input):
        """
        The counterpart to the parity guarantee: in train mode BatchNorm
        normalizes by batch statistics, so outputs necessarily move away from
        the folded model. That is the point of the change, and it is why the
        parity claim only ever held in eval mode.
        """
        batchnorm = BlazeEar(use_batchnorm=True)
        load_mediapipe_weights(batchnorm, str(WEIGHTS), strict=False)

        batchnorm.eval()
        with torch.no_grad():
            _, eval_scores = batchnorm(sample_input)
        batchnorm.train()
        with torch.no_grad():
            _, train_scores = batchnorm(sample_input)

        assert not torch.allclose(eval_scores, train_scores)

    def test_gradients_reach_batchnorm_parameters(self, sample_input):
        model = BlazeEar(use_batchnorm=True)
        load_mediapipe_weights(model, str(WEIGHTS), strict=False)
        model.train()
        boxes, scores = model(sample_input)
        (boxes.mean() + scores.mean()).backward()

        bn = model.backbone1[2].bn1
        assert bn.weight.grad is not None and torch.isfinite(bn.weight.grad).all()
        assert bn.weight.grad.abs().sum() > 0


class TestHeadInitialization:
    def test_prior_bias_starts_near_the_background_prior(self):
        """
        Every anchor should start at roughly p = 0.01, so the classifier begins
        biased toward "no object" instead of wherever the inherited head sat.
        """
        model = BlazeEar()
        model.init_detection_heads(prior_probability=0.01)
        expected = -math.log((1 - 0.01) / 0.01)
        for layer in (model.classifier_8, model.classifier_16):
            assert torch.allclose(layer.bias, torch.full_like(layer.bias, expected), atol=1e-5)

    def test_prior_probability_is_honoured(self):
        model = BlazeEar()
        model.init_detection_heads(prior_probability=0.5)
        assert torch.allclose(model.classifier_8.bias,
                              torch.zeros_like(model.classifier_8.bias), atol=1e-6)

    def test_regressor_bias_is_zero(self):
        model = BlazeEar()
        model.init_detection_heads()
        for layer in (model.regressor_8_box, model.regressor_16_box):
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    @requires_weights
    def test_backbone_only_load_leaves_heads_reinitialisable(self):
        """
        The inherited MediaPipe head is calibrated for folded-BN activation
        scales; under trainable BatchNorm it produced a median logit of -339 on
        ear anchors. Loading the backbone alone must not repopulate the heads.
        """
        model = BlazeEar()
        load_mediapipe_weights(model, str(WEIGHTS), strict=False,
                               load_detection_heads=False)
        model.init_detection_heads()
        model.train()
        with torch.no_grad():
            _, scores = model(torch.randn(4, 3, 128, 128) * 50 + 128)
        assert abs(float(scores.median())) < 20, 'logits should start near the prior'
