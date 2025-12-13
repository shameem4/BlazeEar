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
from typing import Any, Dict, Tuple, cast
import warnings

import torch
import torch.nn as nn
import torchvision

from blazeear import BlazeEar
from utils.box_utils import yxyx_to_xyxy


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

        ymin = torch.minimum(boxes_yxyx[..., 0], boxes_yxyx[..., 2])
        ymax = torch.maximum(boxes_yxyx[..., 0], boxes_yxyx[..., 2])
        xmin = torch.minimum(boxes_yxyx[..., 1], boxes_yxyx[..., 3])
        xmax = torch.maximum(boxes_yxyx[..., 1], boxes_yxyx[..., 3])
        boxes_yxyx = torch.stack((ymin, xmin, ymax, xmax), dim=-1)

        thresh = float(getattr(self.model, "score_clipping_thresh", 100.0))
        scores = raw_scores.clamp(-thresh, thresh).sigmoid().squeeze(-1)

        # Export assumes batch=1 for postprocessed detections.
        scores = scores[0]
        boxes_yxyx = boxes_yxyx[0]

        scores = torch.where(scores >= self.score_thresh, scores, torch.zeros_like(scores))
        topk_scores, topk_idx = torch.topk(scores, k=self.topk, dim=0)
        boxes_topk = boxes_yxyx.index_select(0, topk_idx)

        boxes_xyxy = yxyx_to_xyxy(boxes_topk)
        keep_idx = torchvision.ops.nms(boxes_xyxy, topk_scores, self.iou_thresh)

        final_boxes = boxes_topk.index_select(0, keep_idx)
        final_scores = topk_scores.index_select(0, keep_idx).unsqueeze(-1)
        return torch.cat((final_boxes, final_scores), dim=-1)  # (N, 5) yxyx + score


def _unwrap_checkpoint(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        state_dict = obj

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint does not look like a PyTorch state_dict.")

    first_key = next(iter(state_dict.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    return state_dict


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: int = 128,
    opset: int = 17,
    constant_folding: bool = False,
    fallback: bool = True,
    verbose: bool = False,
    postprocess: bool = False,
    score_thresh: float = 0.75,
    iou_thresh: float = 0.3,
    topk: int = 200,
) -> None:
    if not fallback and importlib.util.find_spec("onnxscript") is None:
        raise RuntimeError(
            "The torch.export-based ONNX exporter requires `onnxscript`.\n"
            "Install it with: pip install onnx onnxscript\n"
            "(Or run without `--no-fallback` to allow a legacy fallback.)"
        )

    model = BlazeEar()
    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_checkpoint(checkpoint_obj)
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

    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

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
        fallback=fallback,
        **export_kwargs,
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
        default=128,
        help="Model input resolution (expects square, default 128)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
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
        default=0.75,
        help="Score threshold used during ONNX postprocessing export.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="IoU threshold used for NMS during ONNX postprocessing export.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Top-K candidates before NMS during ONNX postprocessing export.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable the legacy fallback (requires a fully-supported torch.export -> ONNX path).",
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
        fallback=not args.no_fallback,
        verbose=args.verbose,
        postprocess=args.postprocess,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        topk=args.topk,
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
