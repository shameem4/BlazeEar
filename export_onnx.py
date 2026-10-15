"""
Export a trained BlazeEar `.pth` checkpoint to ONNX.

Default input/output expects the training checkpoint format produced by
`train_blazeear.py` (a dict containing `model_state_dict`), but it also
supports raw `state_dict` checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Tuple, cast
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision

from blazebase import checkpoint_is_folded, load_checkpoint_state
from blazeear import BlazeEar
from utils.box_utils import sort_corners, usable_box_mask, yxyx_to_xyxy
from utils.config import (DEFAULT_INPUT_SIZE, DETECTION_SCORE_THRESHOLD,
                          NMS_IOU_THRESHOLD)


# The torch.export-based exporter implements opset 18 upwards, and when it
# cannot down-convert a node it keeps 18 rather than failing -- this model's
# `Pad` has no 17 adapter, so asking for 17 used to yield an 18 graph labelled
# as a 17 request. Ask for what we actually get.
DEFAULT_OPSET = 18

# Tolerances for the check below, per quantity rather than one global number.
# Raw boxes are offsets in the input's pixel scale (hundreds), so a hundredth
# of a pixel is already far below anything that could move a detection; raw
# scores are logits, and a thousandth of a logit cannot cross a threshold.
# Postprocessed boxes are normalized coordinates and the scores probabilities,
# so both get the tighter bound.
RAW_BOX_TOLERANCE = 1e-2
RAW_SCORE_TOLERANCE = 1e-3
DETECTION_TOLERANCE = 1e-4


def build_probe(input_size: int, image_path: Path | None) -> torch.Tensor:
    """The tensor the graph is exported with and checked against.

    The default is random noise, and for `--postprocess` that is close to
    useless: the wrapper maps input through `x / 127.5 - 1.0`, so a `randn`
    probe lands in roughly [-1.03, -0.97] -- near-uniform grey, which scores
    nowhere near the 0.8 gate. Decode and NMS then see no box and the check
    reports that it verified nothing. `--check-image` is the way to make the
    check mean something: a real image exercises decode, NMS and the gate on
    the actual graph.
    """
    if image_path is None:
        return torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"--check-image: cannot read {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    # The wrapper normalizes internally, so hand it [0, 255].
    return (torch.from_numpy(resized.copy()).permute(2, 0, 1)
            .unsqueeze(0).float())


def check_against_torch(
    wrapper: nn.Module,
    output_path: Path,
    dummy: torch.Tensor,
    postprocess: bool,
) -> bool:
    """A graph that exported without raising can still disagree with its model.

    This is what replaced the old `--no-fallback`: rather than choosing an
    exporter and hoping, export and then prove the graph matches.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed, so the exported graph was not "
              "checked against the model.\n"
              "  Install it with: pip install onnxruntime  "
              "(or pass --no-check to skip this deliberately).")
        return True

    with torch.no_grad():
        expected = wrapper(dummy)
    want = [expected] if isinstance(expected, torch.Tensor) else list(expected)

    session = ort.InferenceSession(str(output_path),
                                   providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": dummy.numpy()})

    if postprocess:
        names = ["detections"]
        tolerances = [DETECTION_TOLERANCE]
        if not len(want[0]):
            # The probe is random noise, which usually scores below threshold.
            # The network itself is still covered -- it is the same weights the
            # raw-head export checks -- but decode and NMS saw no box, so say
            # so instead of reporting an empty comparison as a pass.
            print("  detections: the probe produced no detection above "
                  f"{wrapper.score_thresh:g}, so decode, NMS and the gate "
                  "were NOT verified -- this is not a pass, it is an "
                  "unverified graph.\n"
                  "    Re-run with --check-image <a real image> to check "
                  "them, or without --postprocess to check the network.")
            return True
    else:
        names = ["raw_boxes", "raw_scores"]
        tolerances = [RAW_BOX_TOLERANCE, RAW_SCORE_TOLERANCE]

    ok = True
    for name, tol, g, w in zip(names, tolerances, got, want):
        w_np = w.numpy()
        if g.shape != w_np.shape:
            print(f"  {name}: shape {g.shape} != torch {w_np.shape}  "
                  f"[MISMATCH]")
            ok = False
            continue
        err = float(np.abs(g - w_np).max()) if g.size else 0.0
        label = "ok" if err < tol else "MISMATCH"
        print(f"  {name}: max|d| {err:.2e} (tol {tol:.0e})  [{label}]")
        ok &= err < tol
    return ok


class BlazeEarONNXWrapper(nn.Module):
    def __init__(self, model: BlazeEar):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_boxes, raw_scores = self.model(x)
        return raw_boxes, raw_scores


class BlazeEarDetectionsWrapper(nn.Module):
    def __init__(
        self,
        model: BlazeEar,
        anchors: torch.Tensor,
        score_thresh: float,
        iou_thresh: float,
        topk: int,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", anchors)
        self.score_thresh = float(score_thresh)
        self.iou_thresh = float(iou_thresh)
        self.topk = int(topk)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.float32:
            image = image.float()

        x = image / 127.5 - 1.0
        raw_boxes, raw_scores = self.model(x)

        anchors = cast(torch.Tensor, self.anchors)
        x_scale = float(getattr(self.model, "x_scale", 128.0))
        y_scale = float(getattr(self.model, "y_scale", 128.0))
        w_scale = float(getattr(self.model, "w_scale", 128.0))
        h_scale = float(getattr(self.model, "h_scale", 128.0))

        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

        y_min = y_center - h / 2.0
        x_min = x_center - w / 2.0
        y_max = y_center + h / 2.0
        x_max = x_center + w / 2.0
        boxes_yxyx = torch.stack((y_min, x_min, y_max, x_max), dim=-1)

        boxes_yxyx = sort_corners(boxes_yxyx)

        thresh = float(getattr(self.model, "score_clipping_thresh", 100.0))
        scores = raw_scores.clamp(-thresh, thresh).sigmoid().squeeze(-1)

        # Export assumes batch=1 for postprocessed detections.
        scores = scores[0]
        boxes_yxyx = boxes_yxyx[0]

        # Zero the candidates that fail the gate, and the ones with no area,
        # rather than masking rows: topk below needs a fixed length, and the
        # eager paths drop degenerate boxes at this same point.
        keep = (scores >= self.score_thresh) & usable_box_mask(boxes_yxyx, scores)
        scores = torch.where(keep, scores, torch.zeros_like(scores))

        topk_scores, topk_idx = torch.topk(scores, k=self.topk, dim=0)
        boxes_topk = boxes_yxyx.index_select(0, topk_idx)

        boxes_xyxy = yxyx_to_xyxy(boxes_topk)
        keep_idx = torchvision.ops.nms(boxes_xyxy, topk_scores, self.iou_thresh)

        final_boxes = boxes_topk.index_select(0, keep_idx)
        final_scores = topk_scores.index_select(0, keep_idx)

        # Drop the zeroed tail, which used to be returned. On a real crop the
        # graph emitted 56 to 72 rows where only 1 or 2 cleared the gate, the
        # rest at score exactly 0.0, so `detections` did not mean what the
        # README says it means and a caller that did not re-filter got dozens
        # of phantom boxes per image. The check below knew about the tail and
        # filtered it, so the graph was never WRONG where it was checked --
        # the tail was a contract defect, not a numerical one.
        #
        # Removing it also makes the graph checkable against this wrapper at
        # all: the tail is NMS tie-break order among equal zero scores, which
        # torch and onnxruntime resolve differently, so the row counts
        # disagreed (165 against 56) while every real detection matched.
        final_mask = final_scores > 0.0
        final_boxes = final_boxes[final_mask]
        final_scores = final_scores[final_mask].unsqueeze(-1)
        return torch.cat((final_boxes, final_scores), dim=-1)  # (N, 5) yxyx + score


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: int = DEFAULT_INPUT_SIZE,
    opset: int = DEFAULT_OPSET,
    constant_folding: bool = False,
    verbose: bool = False,
    check: bool = True,
    postprocess: bool = False,
    score_thresh: float = DETECTION_SCORE_THRESHOLD,
    iou_thresh: float = NMS_IOU_THRESHOLD,
    topk: int = 200,
    check_image: Path | None = None,
) -> None:
    if importlib.util.find_spec("onnxscript") is None:
        raise RuntimeError(
            "The torch.export-based ONNX exporter requires `onnxscript`.\n"
            "Install it with: pip install onnx onnxscript"
        )

    # Build whichever architecture the checkpoint was written from. Without
    # this, a pre-BatchNorm checkpoint raises on load and the current one
    # exports through a path that was never checked against it.
    state_dict = load_checkpoint_state(str(checkpoint_path))
    folded = checkpoint_is_folded(state_dict)
    if folded:
        print("Checkpoint predates the BatchNorm switch; exporting the folded model.")
    model = BlazeEar(use_batchnorm=not folded)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if postprocess:
        from blazebase import anchor_options

        model.generate_anchors(anchor_options)
        anchors = model.anchors.detach().cpu()
        wrapper: nn.Module = BlazeEarDetectionsWrapper(
            model=model,
            anchors=anchors,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            topk=topk,
        )
        output_names = ["detections"]
        dynamic_axes = {
            "detections": {0: "num_detections"},
        }
    else:
        wrapper = BlazeEarONNXWrapper(model)
        output_names = ["raw_boxes", "raw_scores"]
        dynamic_axes = {
            "image": {0: "batch"},
            "raw_boxes": {0: "batch"},
            "raw_scores": {0: "batch"},
        }
    wrapper.eval()

    dummy = build_probe(input_size, check_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=constant_folding,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    if verbose:
        export_kwargs["verbose"] = True

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        dynamo=True,
        **export_kwargs,
    )

    if check:
        print("Checking the exported graph against the PyTorch model:")
        if not check_against_torch(wrapper, output_path, dummy, postprocess):
            raise SystemExit(
                f"{output_path} disagrees with the model it came from; "
                "do not use it."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BlazeEar checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/checkpoints/BlazeEar_best.pth"),
        help="Path to `.pth` checkpoint (training checkpoint or raw state_dict)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/checkpoints/BlazeEar_best.onnx"),
        help="Output `.onnx` path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Model input resolution (expects square, default "
             f"{DEFAULT_INPUT_SIZE})",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default {DEFAULT_OPSET}; the exporter "
             f"implements 18 upwards and keeps 18 when it cannot "
             f"down-convert)",
    )
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable constant folding during export (can emit warnings for some ops).",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Export a model that outputs final detections (decode + NMS) instead of raw heads.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=DETECTION_SCORE_THRESHOLD,
        help=f"Score threshold used during ONNX postprocessing export "
             f"(default {DETECTION_SCORE_THRESHOLD}, the fitted deployed "
             f"threshold).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=NMS_IOU_THRESHOLD,
        help=f"IoU threshold used for NMS during ONNX postprocessing export "
             f"(default {NMS_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--check-image",
        type=Path,
        default=None,
        help="Check the exported graph against the model on this image "
             "instead of random noise. Strongly recommended with "
             "--postprocess: a noise probe scores below the gate, so decode, "
             "NMS and the threshold go unverified.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Top-K candidates before NMS during ONNX postprocessing export.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip comparing the exported graph against the PyTorch model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable exporter verbosity (can be noisy).",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show PyTorch ONNX exporter warnings.",
    )
    args = parser.parse_args()

    if not args.show_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"# 'dynamic_axes' is not recommended when dynamo=True.*",
            category=UserWarning,
        )

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=args.input_size,
        opset=args.opset,
        constant_folding=args.constant_folding,
        verbose=args.verbose,
        check=not args.no_check,
        postprocess=args.postprocess,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        topk=args.topk,
        check_image=args.check_image,
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
