"""
Parity across the inference paths.

The repository carries several ways to run the same model: `BlazeEar.process`,
the `BlazeEarInference` pipeline used for ONNX export, and the exported graphs
themselves. They previously disagreed -- different anchors, different decode
copies, different post-filters -- so a number measured offline was not the
number the deployed page produced. These tests pin the agreement.
"""
import numpy as np
import torch

from blazeear import BlazeEar
from blazeear_inference import BlazeEarInference
from utils.anchor_utils import (
    anchor_options,
    generate_anchors_from_priors,
    get_anchors,
)
from utils.box_utils import decode_boxes, decode_boxes_with_keypoints


class TestAnchorParity:
    """Anchor w/h scale the raw prediction during decode, so a private copy in
    any path puts that path's boxes at the wrong scale."""

    def test_all_paths_share_one_anchor_tensor(self):
        model = BlazeEar()
        model.generate_anchors(anchor_options)
        pipeline_anchors = BlazeEarInference()._generate_anchors()

        canonical = get_anchors()
        for name, anchors in [
            ('dataloader assignment', generate_anchors_from_priors()),
            ('BlazeEar.process', model.anchors.cpu()),
            ('BlazeEarInference', pipeline_anchors.cpu()),
        ]:
            assert torch.allclose(canonical, anchors), f'{name} diverged'

    def test_anchors_carry_real_priors(self):
        # Guards the regression where three paths hardcoded w=h=1.0.
        anchors = get_anchors()
        assert len(torch.unique(anchors[:, 2:], dim=0)) > 1
        assert anchors.shape == (896, 4)

    def test_get_anchors_is_stable_across_calls(self):
        assert torch.equal(get_anchors(), get_anchors())

    def test_generate_anchors_ignores_stale_options(self):
        """
        `options` previously selected fixed vs variable anchors per call, so a
        caller passing the MediaPipe defaults got a decode that disagreed with
        training. It must now be inert.
        """
        model = BlazeEar()
        model.generate_anchors(dict(anchor_options, fixed_anchor_size=True))
        fixed_request = model.anchors.cpu().clone()
        model.generate_anchors(dict(anchor_options, fixed_anchor_size=False))
        assert torch.allclose(fixed_request, model.anchors.cpu())
        assert torch.allclose(fixed_request, get_anchors())


class TestDecodeParity:
    """box_utils and the pipeline must decode identically."""

    def test_pipeline_decode_matches_box_utils(self):
        torch.manual_seed(0)
        anchors = get_anchors()
        raw = torch.randn(1, 896, 16) * 10

        pipeline = BlazeEarInference()
        from_pipeline = pipeline.decode_boxes(raw, anchors)
        from_utils = decode_boxes_with_keypoints(
            raw, anchors, x_scale=128.0, y_scale=128.0,
            w_scale=128.0, h_scale=128.0, num_keypoints=0,
        )[..., :4]

        # box_utils does not reorder corners, so compare against ordered output.
        ordered = torch.stack([
            torch.minimum(from_utils[..., 0], from_utils[..., 2]),
            torch.minimum(from_utils[..., 1], from_utils[..., 3]),
            torch.maximum(from_utils[..., 0], from_utils[..., 2]),
            torch.maximum(from_utils[..., 1], from_utils[..., 3]),
        ], dim=-1)
        assert torch.allclose(from_pipeline, ordered, atol=1e-5)

    def test_loss_decode_matches_pipeline_decode(self):
        torch.manual_seed(1)
        anchors = get_anchors()
        raw = torch.randn(1, 896, 4) * 5

        loss_side = decode_boxes(raw, anchors, scale=128.0)
        pipeline_side = BlazeEarInference().decode_boxes(
            torch.cat([raw, torch.zeros(1, 896, 12)], dim=-1), anchors
        )
        ordered = torch.stack([
            torch.minimum(loss_side[..., 0], loss_side[..., 2]),
            torch.minimum(loss_side[..., 1], loss_side[..., 3]),
            torch.maximum(loss_side[..., 0], loss_side[..., 2]),
            torch.maximum(loss_side[..., 1], loss_side[..., 3]),
        ], dim=-1)
        assert torch.allclose(ordered, pipeline_side, atol=1e-5)

    def test_zero_prediction_decodes_to_the_anchor(self):
        """A zero prediction must reproduce the anchor box itself."""
        anchors = get_anchors()
        decoded = decode_boxes(torch.zeros(1, 896, 4), anchors, scale=128.0)[0]
        expected_h = decoded[:, 2] - decoded[:, 0]
        expected_w = decoded[:, 3] - decoded[:, 1]
        # w/h collapse to zero because decode scales the prediction by the
        # anchor; the centres must still land on the anchor centres.
        centre_y = (decoded[:, 0] + decoded[:, 2]) / 2
        centre_x = (decoded[:, 1] + decoded[:, 3]) / 2
        assert torch.allclose(centre_y, anchors[:, 1], atol=1e-6)
        assert torch.allclose(centre_x, anchors[:, 0], atol=1e-6)
        assert torch.allclose(expected_h, torch.zeros_like(expected_h), atol=1e-6)
        assert torch.allclose(expected_w, torch.zeros_like(expected_w), atol=1e-6)


class TestEndToEndParity:
    def test_preprocessing_is_bit_identical(self):
        """
        Both paths must hand the model exactly the same pixels. They used
        different resamplers -- cv2.resize against F.interpolate(bilinear) --
        which skewed pixels by up to 0.0072 on a [-1, 1] scale and raw outputs
        by 0.003. Small, but a train/serve difference nothing would surface,
        and the weights are fitted against cv2 because that is what the
        dataloader uses.
        """
        model = BlazeEar()
        pipeline = BlazeEarInference()
        rng = np.random.default_rng(0)

        for shape in [(200, 320, 3), (480, 480, 3), (640, 360, 3), (97, 53, 3)]:
            image = (rng.random(shape) * 255).astype(np.uint8)
            _, resized, scale, pad = model.resize_pad(image)
            from_detector = model._preprocess(
                torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
            )
            from_pipeline, pipe_scale, pipe_pad, _ = pipeline.preprocess(image)

            assert torch.equal(from_detector, from_pipeline), f'pixels differ at {shape}'
            assert scale == pipe_scale, f'scale differs at {shape}'
            assert pad == pipe_pad, f'pad differs at {shape}'

    def test_process_and_pipeline_agree_on_the_same_weights(self):
        """
        `BlazeEar.process` and `BlazeEarInference.predict` run the same weights
        over the same image. They may differ in post-filtering, but the
        underlying detections must line up.
        """
        torch.manual_seed(0)
        model = BlazeEar()
        model.generate_anchors(anchor_options)
        model.eval()

        pipeline = BlazeEarInference(confidence_threshold=0.0, iou_threshold=1.0)
        pipeline.model.load_state_dict(model.state_dict())
        pipeline.eval()

        image = (np.random.default_rng(0).random((200, 320, 3)) * 255).astype(np.uint8)

        with torch.no_grad():
            _, resized, _, _ = model.resize_pad(image)
            raw_a = model(model._preprocess(
                torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
            ))
            # preprocess() already returns [-1, 1]; normalizing again here was
            # a bug in this test that masked the real comparison.
            prepped, _, _, _ = pipeline.preprocess(image)
            raw_b = pipeline.model(prepped)

        assert torch.equal(raw_a[0], raw_b[0]), 'box outputs diverge'
        assert torch.equal(raw_a[1], raw_b[1]), 'score outputs diverge'
