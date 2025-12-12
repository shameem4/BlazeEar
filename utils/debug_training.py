import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Allow running as `python utils/debug_training.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataloader import CSVDetectorDataset, encode_boxes_to_anchors, flatten_anchor_targets
from blazeear import BlazeEar
from blazedetector import BlazeDetector
from blazebase import generate_reference_anchors
from loss_functions import BlazeEarDetectionLoss, compute_mean_iou
from utils import model_utils
from utils.detection_filters import filter_duplicate_boxes
from utils.config import (
    DEFAULT_COMPARE_LABEL,
    DEFAULT_COMPARE_THRESHOLD,
    DEFAULT_DATA_ROOT,
    DEFAULT_DEBUG_WEIGHTS,
    DEFAULT_DETECTOR_THRESHOLD_DEBUG,
    DEFAULT_EVAL_IOU_THRESHOLD,
    DEFAULT_EVAL_MAX_IMAGES,
    DEFAULT_EVAL_SCORE_THRESHOLD,
    DEFAULT_SCREENSHOT_CANDIDATES,
    DEFAULT_SCREENSHOT_COUNT,
    DEFAULT_SCREENSHOT_MIN_FACES,
    DEFAULT_SCREENSHOT_OUTPUT,
    DEFAULT_TRAIN_CSV
)
from utils.visualization_utils import (
    compute_resize_metadata,
    convert_ymin_xmin_to_xyxy,
    draw_box,
    map_preprocessed_boxes_to_original,
)

LOSS_DEBUG_KWARGS = {
    "use_focal_loss": True,
    "positive_classification_weight": 70.0
}

LOG_LINE = "=" * 70


def _log_heading(title: str, char: str = "=") -> None:
    line = (char * len(title)) if len(title) >= len(LOG_LINE) else LOG_LINE


def run_decode_unit_test(loss_fn: BlazeEarDetectionLoss) -> None:
    _log_subheading("Decode sanity check")
    reference_anchors, _, _ = generate_reference_anchors()
    gt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    preds = torch.zeros((1, 1, 4))
    preds[..., 0] = (gt[0, 1] + gt[0, 3]) / 2 * loss_fn.scale - reference_anchors[0, 0]
    preds[..., 1] = (gt[0, 0] + gt[0, 2]) / 2 * loss_fn.scale - reference_anchors[0, 1]
    preds[..., 2] = (gt[0, 3] - gt[0, 1]) * loss_fn.scale
    preds[..., 3] = (gt[0, 2] - gt[0, 0]) * loss_fn.scale
    decoded = loss_fn.decode_boxes(preds, reference_anchors).squeeze(0)
    torch.testing.assert_close(decoded[0], gt[0], atol=1e-3)
    decoded_xyxy = convert_ymin_xmin_to_xyxy(decoded[0].cpu().numpy())
    expected_xyxy = convert_ymin_xmin_to_xyxy(gt[0].cpu().numpy())
    torch.testing.assert_close(torch.from_numpy(decoded_xyxy), torch.from_numpy(expected_xyxy), atol=1e-3)
    _log_kv("Decode", "PASS synthetic box round-trip")


def run_csv_encode_decode_test(
    dataset: CSVDetectorDataset,
    max_samples: int = 3
) -> None:
    """Ensure encode/decode math is consistent with CSV-derived GT boxes."""
    _log_subheading("CSV encode/decode consistency test")
    reference_anchors, _, _ = generate_reference_anchors()
    loss_fn = BlazeEarDetectionLoss(**LOSS_DEBUG_KWARGS)
    sample_indices = list(range(min(max_samples, len(dataset))))

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        anchor_targets = sample["anchor_targets"]
        positive_mask = anchor_targets[:, 0] == 1
        if not bool(positive_mask.any()):
            _log_kv(f"Sample {sample_idx}", "no positives – skipping")
            continue

        pos_indices = torch.nonzero(positive_mask).squeeze(1)
        true_boxes = anchor_targets[pos_indices, 1:]
        anchor_predictions = torch.zeros((reference_anchors.shape[0], 4), dtype=torch.float32)

        for anchor_idx, true_box in zip(pos_indices.tolist(), true_boxes):
            anchor = reference_anchors[anchor_idx]
            y_min, x_min, y_max, x_max = true_box.tolist()
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            anchor_w = anchor[2].item()
            anchor_h = anchor[3].item()

            anchor_predictions[anchor_idx, 0] = ((x_center - anchor[0].item()) / anchor_w) * loss_fn.scale
            anchor_predictions[anchor_idx, 1] = ((y_center - anchor[1].item()) / anchor_h) * loss_fn.scale
            anchor_predictions[anchor_idx, 2] = (width / anchor_w) * loss_fn.scale
            anchor_predictions[anchor_idx, 3] = (height / anchor_h) * loss_fn.scale

        decoded = loss_fn.decode_boxes(anchor_predictions.unsqueeze(0), reference_anchors).squeeze(0)
        decoded_pos = decoded[pos_indices]
        max_error = (decoded_pos - true_boxes).abs().max().item()
        mean_iou = compute_mean_iou(decoded_pos, true_boxes).item()
        _log_kv(
            f"Sample {sample_idx}",
            f"positives={len(pos_indices)} max_err={max_error:.6f} mean_iou={mean_iou:.4f}"
        )




def _select_top_indices(
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    top_k: int
) -> tuple[list[int], list[float]]:
    """Return up to top_k anchor indices prioritizing positives."""
    selected_indices: list[int] = []
    selected_scores: list[float] = []

    for anchor_idx_np in top_indices.detach().cpu().numpy():
        anchor_idx = int(anchor_idx_np)
        if anchor_targets[anchor_idx, 0] == 1:
            selected_indices.append(anchor_idx)
            selected_scores.append(float(class_predictions[anchor_idx].item()))

    if not selected_indices:
        selected_indices = [int(idx) for idx in top_indices.detach().cpu().numpy().tolist()]
        selected_scores = [float(score) for score in top_scores.detach().cpu().numpy().tolist()]

    selected_indices = selected_indices[:top_k]
    selected_scores = selected_scores[:top_k]
    return selected_indices, selected_scores


def analyze_scoring_process(
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    decoded_boxes: torch.Tensor,
    top_k: Tuple[int, int] = (10, 50)
) -> None:
    """Trace how classification scores align with GT assignments."""
    scores = class_predictions.detach().cpu().squeeze(-1)
    decoded = decoded_boxes.detach().cpu()
    targets = anchor_targets.detach().cpu()

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    highest_idx = int(sorted_indices[0].item())

    pos_mask = targets[:, 0] == 1
    positive_indices = torch.nonzero(pos_mask).squeeze(1)
    pos_indices = positive_indices.to(dtype=torch.long)
    pos_count = int(pos_indices.numel())

    _log_subheading("Scoring diagnostics")
    if pos_count == 0:
        _log_kv("Status", "No positive anchors available")
        return

    pos_scores = scores.index_select(0, pos_indices)
    pos_boxes = targets.index_select(0, pos_indices)[:, 1:]
    decoded_pos = decoded.index_select(0, pos_indices)
    pos_iou = aligned_iou(decoded_pos, pos_boxes)

    mean_score = pos_scores.mean().item()
    mean_iou = pos_iou.mean().item()
    corr = float("nan")
    if pos_scores.numel() > 1:
        stacked = torch.stack([pos_scores, pos_iou])
        corr = torch.corrcoef(stacked)[0, 1].item()

    best_iou_idx = int(torch.argmax(pos_iou).item())
    best_iou_anchor = int(pos_indices[best_iou_idx].item())
    best_iou_score = pos_scores[best_iou_idx].item()
    best_iou_value = pos_iou[best_iou_idx].item()
    best_iou_rank = (sorted_indices == best_iou_anchor).nonzero(as_tuple=False)[0].item() + 1

    best_score_idx = int(torch.argmax(pos_scores).item())
    best_score_anchor = int(pos_indices[best_score_idx].item())
    best_score_value = pos_scores[best_score_idx].item()
    best_score_iou = pos_iou[best_score_idx].item()
    best_score_rank = (sorted_indices == best_score_anchor).nonzero(as_tuple=False)[0].item() + 1

    highest_iou = aligned_iou(decoded[highest_idx].unsqueeze(0), targets[highest_idx, 1:].unsqueeze(0)).item() \
        if targets[highest_idx, 0] == 1 else 0.0

    _log_kv(
        "Positive anchors",
        f"{pos_count} | mean_score={mean_score:.3f} mean_iou={mean_iou:.3f} corr={corr:.3f}"
    )
    _log_kv(
        "Best IoU",
        f"anchor #{best_iou_anchor} IoU={best_iou_value:.3f} score={best_iou_score:.3f} rank={best_iou_rank}"
    )
    _log_kv(
        "Best score",
        f"anchor #{best_score_anchor} score={best_score_value:.3f} IoU={best_score_iou:.3f} rank={best_score_rank}"
    )
    _log_kv(
        "Global top score",
        f"anchor #{highest_idx} {'(pos)' if targets[highest_idx,0]==1 else '(bg)'} "
        f"IoU={highest_iou:.3f} score={sorted_scores[0].item():.3f}"
    )

    for k in top_k:
        window = min(k, len(sorted_indices))
        selected = sorted_indices[:window]
        selected_targets = targets.index_select(0, selected.long())
        positive_hits = int(selected_targets[:, 0].sum().item())
        _log_kv(f"Positives top-{window}", f"{positive_hits}/{window}")


def create_debug_visualization(
    dataset: CSVDetectorDataset,
    sample_idx: int,
    decoded_boxes: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    top_k: int = 5,
    comparison_detector: Optional[BlazeEar] = None,
    comparison_label: str = "Mediapipe",
    primary_label: str = "Finetuned",
    filename: Optional[str] = None,
    averaged_detector: Optional[BlazeEar] = None,
    averaged_label: str = "Finetuned",
    averaged_threshold: float = 0.5,
    averaged_color: Tuple[int, int, int] = (255, 0, 255)
) -> Path:
    """Create a debug overlay showing GT/padded GT/model predictions."""
    image_path, gt_boxes_xywh = dataset.get_sample_annotations(sample_idx)
    orig_image = dataset._load_image(image_path)
    orig_h, orig_w = orig_image.shape[:2]

    if len(gt_boxes_xywh) > 0:
        x1 = gt_boxes_xywh[:, 0]
        y1 = gt_boxes_xywh[:, 1]
        w = gt_boxes_xywh[:, 2]
        h = gt_boxes_xywh[:, 3]

        gt_box_orig = np.stack([x1, y1, x1 + w, y1 + h], axis=1).astype(np.float32)

        ymin = y1 / orig_h
        xmin = x1 / orig_w
        ymax = (y1 + h) / orig_h
        xmax = (x1 + w) / orig_w
        gt_norm = np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)
    else:
        gt_box_orig = np.zeros((0, 4), dtype=np.float32)
        gt_norm = np.zeros((0, 4), dtype=np.float32)

    _, resized_boxes_norm = dataset._resize_and_pad(orig_image, gt_norm.copy())
    gt_resized_xy = convert_ymin_xmin_to_xyxy(resized_boxes_norm)

    scale, pad_top, pad_left = compute_resize_metadata(orig_h, orig_w, dataset.target_size)
    gt_resized_on_orig = map_preprocessed_boxes_to_original(
        gt_resized_xy,
        (orig_h, orig_w),
        dataset.target_size,
        scale,
        pad_top,
        pad_left
    )

    debug_image = cv2.cvtColor(orig_image.copy(), cv2.COLOR_RGB2BGR)

    comparison_np: Optional[np.ndarray] = None
    mediapipe_count = 0
    if comparison_detector is not None:
        comparison_input = np.ascontiguousarray(orig_image)
        try:
            detections = comparison_detector.process(comparison_input)
            if detections is not None and detections.numel() > 0:
                comparison_np = detections.detach().cpu().numpy()
                mediapipe_count = len(comparison_np)
                _log_info(f"{comparison_label}: {len(comparison_np)} detections")
        except Exception as exc:  # pragma: no cover - debug helper
            _log_warn(f"Secondary detector failed on sample {sample_idx}: {exc}")

    for box in gt_box_orig:
        draw_box(debug_image, box, (0, 255, 0), "GT original")

    decoded_np = decoded_boxes.detach().cpu().numpy()
    selected_indices, selected_scores = _select_top_indices(
        anchor_targets, class_predictions, top_indices, top_scores, top_k
    )
    pred_boxes = convert_ymin_xmin_to_xyxy(decoded_np[selected_indices])
    pred_boxes_on_orig = map_preprocessed_boxes_to_original(
        pred_boxes,
        (orig_h, orig_w),
        dataset.target_size,
        scale,
        pad_top,
        pad_left
    )

    for rank, (box, score, anchor_idx) in enumerate(
        zip(pred_boxes_on_orig, selected_scores, selected_indices)
    ):
        label = f"{primary_label} {rank} #{anchor_idx} {score:.2f}"
        # draw_box(debug_image, box, (0, 0, 255), label)

    if comparison_np is not None and comparison_np.size > 0:
        for det_idx, det in enumerate(comparison_np):
            comp_box = np.array([det[1], det[0], det[3], det[2]], dtype=np.float32)
            score = float(det[-1]) if det.shape[0] > 4 else 0.0
            draw_box(
                debug_image,
                comp_box,
                color=(0, 165, 255),
                label=f"{comparison_label} {det_idx} {score:.2f}"
            )

    averaged_count = 0
    if averaged_detector is not None:
        avg_boxes, avg_scores = _run_detector_on_image(
            detector=averaged_detector,
            image_rgb=orig_image,
            device=device,
            target_size=dataset.target_size
        )
        if avg_scores.size > 0:
            mask = avg_scores >= averaged_threshold
            avg_boxes = avg_boxes[mask]
            avg_scores = avg_scores[mask]
        averaged_count = len(avg_boxes)
        for det_idx, (box, score) in enumerate(zip(avg_boxes, avg_scores)):
            draw_box(
                debug_image,
                box,
                color=averaged_color,
                label=f"{averaged_label} {det_idx} {score:.2f}"
            )

    summary_lines = [
        f"GT: {gt_box_orig.shape[0]}",
        # f"{comparison_label}: {mediapipe_count}",
        f"{averaged_label}: {averaged_count}"
    ]
    for idx, line in enumerate(summary_lines):
        y = 25 + idx * 25
        cv2.putText(
            debug_image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    target_name = filename or f"sample_{sample_idx:04d}_{image_stem}.png"
    debug_path = output_dir / target_name
    cv2.imwrite(str(debug_path), debug_image)
    return debug_path


def _boxes_are_near_xyxy_np(
    candidate: np.ndarray,
    boxes: np.ndarray,
    max_center_distance_frac: float = 0.55,
    min_area_ratio: float = 0.35
) -> np.ndarray:
    """Return mask of boxes whose centers/areas closely match the candidate."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)

    candidate = candidate.astype(np.float32, copy=False)
    boxes = boxes.astype(np.float32, copy=False)

    cand_h = max(1e-6, candidate[2] - candidate[0])
    cand_w = max(1e-6, candidate[3] - candidate[1])
    cand_area = cand_h * cand_w
    cand_center_y = (candidate[0] + candidate[2]) / 2.0
    cand_center_x = (candidate[1] + candidate[3]) / 2.0

    box_h = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None)
    box_w = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
    box_area = box_h * box_w
    box_center_y = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_center_x = (boxes[:, 1] + boxes[:, 3]) / 2.0

    center_distance = np.sqrt(
        (cand_center_y - box_center_y) ** 2 + (cand_center_x - box_center_x) ** 2
    )
    other_max_dim = np.maximum(box_h, box_w)
    max_dim = np.maximum(np.maximum(cand_h, cand_w), other_max_dim)
    center_close = center_distance <= (max_center_distance_frac * max_dim)

    area_ratio = np.minimum(cand_area, box_area) / np.maximum(cand_area, box_area)
    return center_close & (area_ratio >= min_area_ratio)


def _box_covers_target_xyxy_np(
    candidate: np.ndarray,
    boxes: np.ndarray,
    min_coverage: float = 0.8
) -> np.ndarray:
    """Return mask for boxes that are mostly covered by candidate."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)

    candidate = candidate.astype(np.float32, copy=False)
    boxes = boxes.astype(np.float32, copy=False)

    y_min = np.maximum(candidate[0], boxes[:, 0])
    x_min = np.maximum(candidate[1], boxes[:, 1])
    y_max = np.minimum(candidate[2], boxes[:, 2])
    x_max = np.minimum(candidate[3], boxes[:, 3])

    inter_h = np.clip(y_max - y_min, 0.0, None)
    inter_w = np.clip(x_max - x_min, 0.0, None)
    inter_area = inter_h * inter_w
    target_area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
    coverage = inter_area / target_area
    return coverage >= min_coverage


def _filter_duplicate_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    center_distance_frac: float = 0.55,
    min_area_ratio: float = 0.35,
    min_coverage: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress duplicate predictions that differ mainly by scale."""
    if boxes.shape[0] <= 1:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    for idx in order:
        candidate = boxes[idx]
        duplicate = False
        for kept_idx in keep:
            kept_box = boxes[kept_idx]
            if _box_covers_target_xyxy_np(kept_box, candidate[np.newaxis, :], min_coverage)[0]:
                duplicate = True
                break
            if _boxes_are_near_xyxy_np(kept_box, candidate[np.newaxis, :], center_distance_frac, min_area_ratio)[0]:
                duplicate = True
                break
        if not duplicate:
            keep.append(idx)

    if not keep:
        return boxes[:0], scores[:0]

    keep_array = np.array(keep, dtype=np.int64)
    return boxes[keep_array], scores[keep_array]


def _prepare_inference_tensor(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, float, int, int]:
    """Resize and pad RGB image to detector input while tracking offsets."""
    orig_h, orig_w = image_rgb.shape[:2]
    scale, pad_top, pad_left = compute_resize_metadata(orig_h, orig_w, target_size)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h))
    pad_bottom = max(target_size[0] - new_h - pad_top, 0)
    pad_right = max(target_size[1] - new_w - pad_left, 0)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded, scale, pad_top, pad_left


def _run_detector_on_image(
    detector: Optional[BlazeEar],
    image_rgb: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray]:
    """Run detector on RGB image and return (boxes_xyxy, scores)."""
    if detector is None:
        empty = np.empty((0,), dtype=np.float32)
        return empty.reshape(0, 4), empty

    padded, scale, pad_top, pad_left = _prepare_inference_tensor(image_rgb, target_size)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(device)

    detector.eval()
    with torch.no_grad():
        detections = detector.predict_on_batch(tensor)

    dets = detections[0] if detections else []
    if isinstance(dets, torch.Tensor):
        dets = dets.detach().cpu().numpy()
    else:
        dets = np.asarray(dets)

    if dets.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty.reshape(0, 4), empty

    boxes_norm = dets[:, :4]
    scores = dets[:, -1] if dets.shape[1] > 4 else np.ones(len(dets), dtype=np.float32)
    boxes_xyxy = convert_ymin_xmin_to_xyxy(boxes_norm)
    mapped = map_preprocessed_boxes_to_original(
        boxes_xyxy,
        image_rgb.shape[:2],
        target_size,
        scale,
        pad_top,
        pad_left
    )
    filtered_boxes, filtered_scores = filter_duplicate_boxes(
        mapped.astype(np.float32, copy=False),
        scores.astype(np.float32, copy=False)
    )
    return filtered_boxes, filtered_scores


def _select_multiface_indices(
    dataset: CSVDetectorDataset,
    min_faces: int,
    max_candidates: int
) -> List[int]:
    """Return dataset indices with at least min_faces annotations."""
    selections: List[int] = []
    for idx, sample in enumerate(dataset.samples):
        if len(sample["boxes"]) < min_faces:
            continue
        selections.append(idx)
        if len(selections) >= max_candidates:
            break
    return selections


def _collect_sample_debug_data(
    model: BlazeEar,
    dataset: CSVDetectorDataset,
    sample_idx: int,
    device: torch.device,
    loss_fn: BlazeEarDetectionLoss,
    reference_anchors: torch.Tensor,
    top_k: int = 10
) -> Dict[str, torch.Tensor]:
    """Run model on a dataset sample and prepare tensors for visualization."""
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    anchor_targets = sample["anchor_targets"].to(device)

    with torch.no_grad():
        raw_boxes, raw_scores = model.get_training_outputs(image)

    class_predictions = torch.sigmoid(raw_scores).squeeze(0)
    anchor_predictions = raw_boxes.squeeze(0)[..., :4]
    decoded_boxes = loss_fn.decode_boxes(
        anchor_predictions.unsqueeze(0),
        reference_anchors
    ).squeeze(0)

    available = class_predictions.shape[0]
    top_k = min(top_k, available)
    top_scores, top_indices = torch.topk(class_predictions.squeeze(-1), k=top_k)

    return {
        "image": image,
        "anchor_targets": anchor_targets,
        "class_predictions": class_predictions,
        "anchor_predictions": anchor_predictions,
        "decoded_boxes": decoded_boxes,
        "top_scores": top_scores,
        "top_indices": top_indices
    }


def generate_readme_screenshots(
    dataset: CSVDetectorDataset,
    output_dir: Path,
    baseline_model: Optional[BlazeEar],
    finetuned_model: BlazeEar,
    device: torch.device,
    loss_fn: BlazeEarDetectionLoss,
    reference_anchors: torch.Tensor,
    min_faces: int,
    max_candidates: int,
    limit: int,
    baseline_label: str,
    finetuned_label: str,
    averaged_detector: Optional[BlazeEar],
    averaged_threshold: float
) -> List[Path]:
    """Generate README screenshots via the standard debug visualization pipeline."""
    candidate_indices = _select_multiface_indices(dataset, min_faces, max_candidates)
    if not candidate_indices:
        _log_warn("No images met the multi-face criteria for screenshots")
        return []

    saved_paths: List[Path] = []
    debug_output_dir = output_dir

    for dataset_idx in candidate_indices:
        if len(saved_paths) >= limit:
            break

        data = _collect_sample_debug_data(
            model=finetuned_model,
            dataset=dataset,
            sample_idx=dataset_idx,
            device=device,
            loss_fn=loss_fn,
            reference_anchors=reference_anchors
        )

        image_path, _ = dataset.get_sample_annotations(dataset_idx)
        image_stem = Path(image_path).stem
        filename = f"sample_{len(saved_paths) + 1:04d}_{image_stem}.png"

        debug_path = create_debug_visualization(
            dataset=dataset,
            sample_idx=dataset_idx,
            decoded_boxes=data["decoded_boxes"],
            top_indices=data["top_indices"],
            top_scores=data["top_scores"],
            anchor_targets=data["anchor_targets"],
            class_predictions=data["class_predictions"].squeeze(-1),
            output_dir=debug_output_dir,
            device=device,
            comparison_detector=baseline_model,
            comparison_label=baseline_label,
            primary_label=finetuned_label,
            filename=filename,
            averaged_detector=averaged_detector,
            averaged_threshold=averaged_threshold
        )
        saved_paths.append(debug_path)

    return saved_paths


def evaluate_dataset_performance(
    model: BlazeEar,
    csv_path: Path,
    data_root: Path,
    device: torch.device,
    max_images: int,
    score_threshold: float,
    iou_threshold: float
) -> Dict[str, Union[int, float]]:
    """Evaluate a detector on CSV-listed images and compute summary metrics."""
    df = pd.read_csv(csv_path)
    grouped = df.groupby("image_path", sort=False)
    image_groups = list(grouped)[:max_images]
    _log_info(f"Evaluating on {len(image_groups)} images")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_detections = 0
    total_gt_boxes = 0

    for image_path, group in tqdm(image_groups, desc="Evaluating", unit="img"):
        image_rel_path = Path(str(image_path))
        full_path = data_root / image_rel_path
        img_bgr = cv2.imread(str(full_path))
        if img_bgr is None:
            _log_warn(f"Failed to load {full_path} during evaluation")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred_boxes, scores = _run_detector_on_image(model, img_rgb, device)
        if scores.size > 0:
            mask = scores >= score_threshold
            pred_boxes = pred_boxes[mask]
        else:
            pred_boxes = pred_boxes[:0]
        pred_boxes = pred_boxes.astype(np.float32, copy=False)

        gt_boxes_xywh = group[["x1", "y1", "w", "h"]].to_numpy(dtype=np.float32)
        if gt_boxes_xywh.size > 0:
            gt_boxes = np.column_stack(
                (
                    gt_boxes_xywh[:, 0],
                    gt_boxes_xywh[:, 1],
                    gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2],
                    gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3]
                )
            ).astype(np.float32)
        else:
            gt_boxes = np.empty((0, 4), dtype=np.float32)

        n_dets = len(pred_boxes)
        n_gt = len(gt_boxes)
        total_detections += n_dets
        total_gt_boxes += n_gt

        if n_gt == 0:
            total_fp += n_dets
            continue
        if n_dets == 0:
            total_fn += n_gt
            continue

        matched_gt: Set[int] = set()
        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                x1 = max(pred[0], gt[0])
                y1 = max(pred[1], gt[1])
                x2 = min(pred[2], gt[2])
                y2 = min(pred[3], gt[3])
                inter_w = max(0.0, x2 - x1)
                inter_h = max(0.0, y2 - y1)
                inter = inter_w * inter_h
                area_pred = max(0.0, (pred[2] - pred[0])) * max(0.0, (pred[3] - pred[1]))
                area_gt = max(0.0, (gt[2] - gt[0])) * max(0.0, (gt[3] - gt[1]))
                union = area_pred + area_gt - inter
                if union <= 0:
                    continue
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        total_fn += n_gt - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "images": len(image_groups),
        "gt_boxes": total_gt_boxes,
        "detections": total_detections,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def _print_evaluation_summary(label: str, stats: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics in README-friendly format."""
    _log_heading(f"{label} Evaluation")
    _log_kv("Images", int(stats['images']))
    _log_kv("GT boxes", int(stats['gt_boxes']))
    _log_kv("Detections", int(stats['detections']))
    _log_kv("True Positives", int(stats['tp']))
    _log_kv("False Positives", int(stats['fp']))
    _log_kv("False Negatives", int(stats['fn']))
    _log_kv("Precision", f"{stats['precision']:.4f}")
    _log_kv("Recall", f"{stats['recall']:.4f}")
    _log_kv("F1 Score", f"{stats['f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug BlazeEar training sample (single image end-to-end)"
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=str, default=DEFAULT_DEBUG_WEIGHTS)
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Comma-separated list of sample indices or single index to inspect"
    )
    parser.add_argument(
        "--compare-weights",
        type=str,
        default=None,#DEFAULT_SECONDARY_WEIGHTS,
        help="Optional path to secondary detector weights (.pth or .ckpt) for visual comparison"
    )
    parser.add_argument(
        "--compare-threshold",
        type=float,
        default=DEFAULT_COMPARE_THRESHOLD,
        help="Detection threshold for the secondary detector"
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default=DEFAULT_COMPARE_LABEL,
        help="Label prefix for the secondary detector annotations"
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=DEFAULT_DETECTOR_THRESHOLD_DEBUG,
        help="Detection threshold for the primary detector when rendering screenshots"
    )
    parser.add_argument(
        "--screenshot-output",
        type=str,
        default=DEFAULT_SCREENSHOT_OUTPUT,
        help="Optional directory to save README comparison screenshots; generation is skipped when not set"
    )
    parser.add_argument(
        "--screenshot-count",
        type=int,
        default=DEFAULT_SCREENSHOT_COUNT,
        help="Number of comparison screenshots to export when --screenshot-output is provided"
    )
    parser.add_argument(
        "--screenshot-min-faces",
        type=int,
        default=DEFAULT_SCREENSHOT_MIN_FACES,
        help="Minimum number of faces per image when selecting screenshot candidates"
    )
    parser.add_argument(
        "--screenshot-candidates",
        type=int,
        default=DEFAULT_SCREENSHOT_CANDIDATES,
        help="Maximum number of candidate images (after filtering) to evaluate for screenshots"
    )
    parser.add_argument(
        "--no-averaged-overlay",
        action="store_true",
        help="Disable drawing averaged (NMS) boxes from the finetuned detector"
    )
    parser.add_argument(
        "--averaged-threshold",
        type=float,
        default=DEFAULT_DETECTOR_THRESHOLD_DEBUG,
        help="Score threshold for averaged overlay detections"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Compute validation metrics before running per-sample debugging"
    )
    parser.add_argument(
        "--eval-max-images",
        type=int,
        default=DEFAULT_EVAL_MAX_IMAGES,
        help="Maximum number of images to use during evaluation"
    )
    parser.add_argument(
        "--eval-score-threshold",
        type=float,
        default=DEFAULT_EVAL_SCORE_THRESHOLD,
        help="Score threshold applied to predictions when computing metrics"
    )
    parser.add_argument(
        "--eval-iou-threshold",
        type=float,
        default=DEFAULT_EVAL_IOU_THRESHOLD,
        help="IoU threshold for counting a prediction as true positive"
    )
    parser.add_argument(
        "--eval-label",
        type=str,
        default=None,
        help="Optional heading label for the evaluation summary"
    )
    args = parser.parse_args()

    run_anchor_unit_tests()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dataset = CSVDetectorDataset(
        csv_path=str(csv_path),
        root_dir=args.data_root,
        target_size=(128, 128),
        augment=False
    )

    run_csv_encode_decode_test(dataset)

    debug_image_dir = Path("runs/logs") / "debug_images"
    # if debug_image_dir.exists():
    #     shutil.rmtree(debug_image_dir)

    device = model_utils.setup_device()

    _log_heading("Debug training configuration")
    _log_kv("CSV", str(csv_path))
    _log_kv("Data root", args.data_root)
    _log_kv("Weights", args.weights)
    _log_kv("Device", str(device))
    _log_kv("Dataset samples", len(dataset))
    loss_fn = BlazeEarDetectionLoss(**LOSS_DEBUG_KWARGS).to(device)
    reference_anchors, _, _ = generate_reference_anchors()
    reference_anchors = reference_anchors.to(device)
    comparison_detector = None
    compare_path = args.compare_weights
    if compare_path:
        try:
            comparison_detector = model_utils.load_model(
                compare_path,
                device=device,
                threshold=args.compare_threshold
            )
            _log_info(f"Loaded comparison detector from {compare_path}")
        except FileNotFoundError:
            _log_warn(f"Comparison weights not found at {compare_path}; skipping overlay")
    else:
        _log_info("Comparison overlay disabled (no --compare-weights path provided)")

    model = model_utils.load_model(
        args.weights,
        device=device,
        grad_enabled=False
    )
    if hasattr(model, "min_score_thresh") and args.detector_threshold is not None:
        model.min_score_thresh = args.detector_threshold

    eval_label = args.eval_label or f"Weights: {Path(args.weights).name}"
    averaged_detector = None if args.no_averaged_overlay else model

    if args.index is None:
        rng = np.random.default_rng()
        indices = rng.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    else:
        indices = [int(idx.strip()) for idx in args.index.split(",") if idx.strip()]

    if args.screenshot_output:
        if comparison_detector is None:
            _log_warn("Skipping screenshot export because no comparison detector was provided")
        else:
            screenshot_paths = generate_readme_screenshots(
                dataset=dataset,
                output_dir=Path(args.screenshot_output),
                baseline_model=comparison_detector,
                finetuned_model=model,
                device=device,
                loss_fn=loss_fn,
                reference_anchors=reference_anchors,
                min_faces=args.screenshot_min_faces,
                max_candidates=args.screenshot_candidates,
                limit=args.screenshot_count,
                baseline_label=args.compare_label,
                finetuned_label="Fine-tuned",
                averaged_detector=averaged_detector,
                averaged_threshold=args.averaged_threshold
            )
            _log_info(f"Generated {len(screenshot_paths)} screenshot(s) in {args.screenshot_output}")

    if args.run_eval:
        total_params = sum(p.numel() for p in model.parameters())
        _log_info(f"Total parameters: {total_params:,}")
        eval_stats = evaluate_dataset_performance(
            model=model,
            csv_path=csv_path,
            data_root=Path(args.data_root),
            device=device,
            max_images=args.eval_max_images,
            score_threshold=args.eval_score_threshold,
            iou_threshold=args.eval_iou_threshold
        )
        _print_evaluation_summary(eval_label, eval_stats)

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Index {idx} is out of range (dataset has {len(dataset)} samples).")

        sample_data = _collect_sample_debug_data(
            model=model,
            dataset=dataset,
            sample_idx=idx,
            device=device,
            loss_fn=loss_fn,
            reference_anchors=reference_anchors
        )

        image = sample_data["image"]
        anchor_targets = sample_data["anchor_targets"]
        class_predictions = sample_data["class_predictions"]
        anchor_predictions = sample_data["anchor_predictions"]
        decoded_boxes = sample_data["decoded_boxes"]
        top_scores = sample_data["top_scores"]
        top_indices = sample_data["top_indices"]

        sample_meta = dataset.samples[idx]
        image_rel = sample_meta.get("image_path", "<unknown>")
        _log_heading(f"Sample {idx}")
        _log_kv("Image", image_rel)
        describe_tensor("image", image)

        positives = anchor_targets[:, 0].sum().item()
        _log_kv("Positive anchors", f"{positives}/896")
        if positives > 0:
            first_pos = anchor_targets[anchor_targets[:, 0] == 1][:3]
            _log_subheading("Example positive targets")
            print(first_pos)

        describe_tensor("class_predictions", class_predictions)
        describe_tensor("anchor_predictions", anchor_predictions)

        selected_indices, selected_scores = _select_top_indices(
            anchor_targets, class_predictions.squeeze(-1), top_indices, top_scores, top_k=5
        )
        _log_subheading("Top anchor scores")
        for score, anchor_idx in zip(top_scores, top_indices):
            _log_kv(f"idx {anchor_idx.item():4d}", f"score={score.item():.4f}")

        analyze_scoring_process(
            anchor_targets=anchor_targets,
            class_predictions=class_predictions.squeeze(-1),
            decoded_boxes=decoded_boxes
        )

        gt_boxes = anchor_targets[:, 1:]
        positive_mask = anchor_targets[:, 0] > 0

        if positive_mask.any():
            mean_iou = compute_mean_iou(
                decoded_boxes[positive_mask],
                gt_boxes[positive_mask]
            )
            _log_kv("Mean IoU (positives)", f"{mean_iou.item():.4f}")

            first_gt = gt_boxes[positive_mask][0]
            _log_kv("First GT box", first_gt.tolist())

            positive_indices = torch.nonzero(positive_mask).squeeze(1)
            _log_kv("Positive anchors idx", positive_indices.tolist())

            pos_boxes = decoded_boxes[positive_indices]
            gt_iou = box_iou(first_gt, pos_boxes)
            for anchor_idx, (box, iou) in zip(positive_indices.tolist(), zip(pos_boxes.tolist(), gt_iou.tolist())):
                _log_kv(f"Anchor {anchor_idx}", f"IoU={iou:.4f} box={box}")

            displayed_indices = torch.tensor(selected_indices, dtype=torch.long, device=decoded_boxes.device)
            top_iou = box_iou(first_gt, decoded_boxes[displayed_indices])
            top_iou_list = top_iou.tolist()
            if isinstance(top_iou_list, float):
                top_iou_list = [top_iou_list]
            _log_subheading("Displayed anchor IoU vs first GT")
            for anchor_idx, score, box, iou in zip(
                displayed_indices.tolist(),
                selected_scores,
                decoded_boxes[displayed_indices].tolist(),
                top_iou_list
            ):
                _log_kv(
                    f"idx {anchor_idx:4d}",
                    f"score={score:.4f} IoU={iou:.4f} box={box}"
                )
        else:
            _log_info("No positive anchors in this sample")

        debug_path = create_debug_visualization(
            dataset=dataset,
            sample_idx=idx,
            decoded_boxes=decoded_boxes,
            top_indices=top_indices,
            top_scores=top_scores,
            anchor_targets=anchor_targets,
            class_predictions=class_predictions.squeeze(-1),
            output_dir=debug_image_dir,
            device=device,
            comparison_detector=comparison_detector,
            comparison_label=args.compare_label,
            averaged_detector=averaged_detector,
            averaged_threshold=args.averaged_threshold
        )
        _log_info(f"Saved debug visualization to {debug_path}")


if __name__ == "__main__":
    main()
