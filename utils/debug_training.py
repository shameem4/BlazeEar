import argparse
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
from blazeface import BlazeFace
from blazedetector import BlazeDetector
from blazebase import generate_reference_anchors
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou
from utils import model_utils
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
    DEFAULT_SECONDARY_WEIGHTS,
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


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu()
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
        f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
    )


def box_iou(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one GT box and N predicted boxes."""
    gt = gt.unsqueeze(0)  # [1, 4]
    ymin = torch.maximum(gt[:, 0], pred[:, 0])
    xmin = torch.maximum(gt[:, 1], pred[:, 1])
    ymax = torch.minimum(gt[:, 2], pred[:, 2])
    xmax = torch.minimum(gt[:, 3], pred[:, 3])

    inter_h = torch.clamp(ymax - ymin, min=0)
    inter_w = torch.clamp(xmax - xmin, min=0)
    intersection = inter_h * inter_w

    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area + pred_area - intersection + 1e-6
    return (intersection / union).squeeze(0)


def aligned_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute IoU for aligned [N,4] tensors."""
    ymin = torch.maximum(pred[:, 0], target[:, 0])
    xmin = torch.maximum(pred[:, 1], target[:, 1])
    ymax = torch.minimum(pred[:, 2], target[:, 2])
    xmax = torch.minimum(pred[:, 3], target[:, 3])

    inter_h = torch.clamp(ymax - ymin, min=0)
    inter_w = torch.clamp(xmax - xmin, min=0)
    intersection = inter_h * inter_w

    pred_area = torch.clamp(pred[:, 2] - pred[:, 0], min=0) * torch.clamp(pred[:, 3] - pred[:, 1], min=0)
    target_area = torch.clamp(target[:, 2] - target[:, 0], min=0) * torch.clamp(target[:, 3] - target[:, 1], min=0)
    union = pred_area + target_area - intersection + 1e-6
    return intersection / union


def run_anchor_unit_tests() -> None:
    print("\nRunning anchor sanity checks...")
    # Single box centered perfectly
    box = np.array([[0.4, 0.4, 0.6, 0.6]], dtype=np.float32)
    small, big = encode_boxes_to_anchors(box, input_size=128)
    targets = flatten_anchor_targets(small, big)
    positives = np.where(targets[:, 0] == 1)[0]
    if positives.size == 0:
        raise AssertionError("No anchors assigned to centered box")
    np.testing.assert_allclose(targets[positives[0], 1:], box[0], atol=1e-3)
    print(f"  PASS single box assigned to anchor #{positives[0]} with coords {targets[positives[0], 1:]}")

    # Multiple boxes shouldn't overwrite each other
    multi_boxes = np.array([
        [0.1, 0.1, 0.2, 0.2],
        [0.7, 0.7, 0.85, 0.85]
    ], dtype=np.float32)
    small, big = encode_boxes_to_anchors(multi_boxes, input_size=128)
    targets = flatten_anchor_targets(small, big)
    assigned = np.where(targets[:, 0] == 1)[0]
    if assigned.size < 2:
        raise AssertionError("Not all boxes were assigned to anchors")
    print(f"  PASS multiple boxes mapped to anchors {assigned.tolist()}")


def run_decode_unit_test(loss_fn: BlazeFaceDetectionLoss) -> None:
    print("Running decode sanity check...")
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
    print("  PASS decode matches ground truth for synthetic box")


def run_csv_encode_decode_test(
    dataset: CSVDetectorDataset,
    max_samples: int = 3
) -> None:
    """Ensure encode/decode math is consistent with CSV-derived GT boxes."""
    print("\nRunning CSV encode/decode consistency test...")
    reference_anchors, _, _ = generate_reference_anchors()
    loss_fn = BlazeFaceDetectionLoss(**LOSS_DEBUG_KWARGS)
    sample_indices = list(range(min(max_samples, len(dataset))))

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        anchor_targets = sample["anchor_targets"]
        positive_mask = anchor_targets[:, 0] == 1
        if not bool(positive_mask.any()):
            print(f"  Sample {sample_idx}: no positives, skipping")
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
        print(
            f"  Sample {sample_idx}: positives={len(pos_indices)}, "
            f"max_abs_error={max_error:.6f}, mean_iou={mean_iou:.4f}"
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

    print("\nScoring diagnostics:")
    if pos_count == 0:
        print("  No positive anchors available for scoring analysis.")
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

    print(
        f"  positives={pos_count}, mean_score={mean_score:.3f}, "
        f"mean_iou={mean_iou:.3f}, score/iou corr={corr:.3f}"
    )
    print(
        f"  best IoU anchor #{best_iou_anchor} -> IoU={best_iou_value:.3f}, "
        f"score={best_iou_score:.3f}, score rank={best_iou_rank}"
    )
    print(
        f"  best score anchor #{best_score_anchor} -> score={best_score_value:.3f}, "
        f"IoU={best_score_iou:.3f}, score rank={best_score_rank}"
    )
    print(
        f"  global top score anchor #{highest_idx} "
        f"{'(positive)' if targets[highest_idx,0]==1 else '(background)'} "
        f"IoU={highest_iou:.3f}, score={sorted_scores[0].item():.3f}"
    )

    for k in top_k:
        window = min(k, len(sorted_indices))
        selected = sorted_indices[:window]
        selected_targets = targets.index_select(0, selected.long())
        positive_hits = int(selected_targets[:, 0].sum().item())
        print(f"  positives within top-{window} scores: {positive_hits}/{window}")


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
    comparison_detector: Optional[BlazeFace] = None,
    comparison_label: str = "Mediapipe",
    primary_label: str = "Finetuned",
    filename: Optional[str] = None,
    averaged_detector: Optional[BlazeFace] = None,
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
                print(f"{comparison_label}: {len(comparison_np)} detections")
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"Secondary detector failed on sample {sample_idx}: {exc}")

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
        f"{comparison_label}: {mediapipe_count}",
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
    detector: Optional[BlazeFace],
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
    return mapped, scores.astype(np.float32)


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
    model: BlazeFace,
    dataset: CSVDetectorDataset,
    sample_idx: int,
    device: torch.device,
    loss_fn: BlazeFaceDetectionLoss,
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
    baseline_model: Optional[BlazeFace],
    finetuned_model: BlazeFace,
    device: torch.device,
    loss_fn: BlazeFaceDetectionLoss,
    reference_anchors: torch.Tensor,
    min_faces: int,
    max_candidates: int,
    limit: int,
    baseline_label: str,
    finetuned_label: str,
    averaged_detector: Optional[BlazeFace],
    averaged_threshold: float
) -> List[Path]:
    """Generate README screenshots via the standard debug visualization pipeline."""
    candidate_indices = _select_multiface_indices(dataset, min_faces, max_candidates)
    if not candidate_indices:
        print("No images met the multi-face criteria for screenshots.")
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
    model: BlazeFace,
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
    print(f"Evaluating on {len(image_groups)} images...")

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
            print(f"Warning: failed to load {full_path} during evaluation")
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
    print(f"\n=== {label} Evaluation ===")
    print(f"Images evaluated: {int(stats['images'])}")
    print(f"Total GT boxes: {int(stats['gt_boxes'])}")
    print(f"Total detections: {int(stats['detections'])}")
    print(f"True Positives: {int(stats['tp'])}")
    print(f"False Positives: {int(stats['fp'])}")
    print(f"False Negatives: {int(stats['fn'])}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall: {stats['recall']:.4f}")
    print(f"F1 Score: {stats['f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug BlazeFace training sample (single image end-to-end)"
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
        default=DEFAULT_SECONDARY_WEIGHTS,
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

    device = model_utils.setup_device()
    loss_fn = BlazeFaceDetectionLoss(**LOSS_DEBUG_KWARGS).to(device)
    reference_anchors, _, _ = generate_reference_anchors()
    reference_anchors = reference_anchors.to(device)
    comparison_detector = None
    compare_path = args.compare_weights or DEFAULT_SECONDARY_WEIGHTS
    if compare_path:
        try:
            comparison_detector = model_utils.load_model(
                compare_path,
                device=device,
                threshold=args.compare_threshold
            )
            print(f"Loaded Mediapipe comparison detector from {compare_path}")
        except FileNotFoundError:
            print(f"Warning: comparison weights not found at {compare_path}; skipping Mediapipe overlay")
    else:
        print("Comparison overlay disabled (no weights path provided).")

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
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    else:
        indices = [int(idx.strip()) for idx in args.index.split(",") if idx.strip()]

    if args.screenshot_output:
        if comparison_detector is None:
            raise ValueError("Mediapipe overlay is required for screenshots; ensure comparison weights are available.")
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
        print(f"Generated {len(screenshot_paths)} screenshot(s) in {args.screenshot_output}")

    if args.run_eval:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
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

        print(f"\nLoaded sample {idx} from {csv_path}")
        describe_tensor("image", image)

        positives = anchor_targets[:, 0].sum().item()
        print(f"Positive anchors: {positives}/896")
        if positives > 0:
            first_pos = anchor_targets[anchor_targets[:, 0] == 1][:3]
            print("First positive targets (class, ymin, xmin, ymax, xmax):")
            print(first_pos)

        describe_tensor("class_predictions", class_predictions)
        describe_tensor("anchor_predictions", anchor_predictions)

        selected_indices, selected_scores = _select_top_indices(
            anchor_targets, class_predictions.squeeze(-1), top_indices, top_scores, top_k=5
        )
        print("Top-10 anchor scores:")
        for score, anchor_idx in zip(top_scores, top_indices):
            print(f"  idx={anchor_idx.item():4d} score={score.item():.4f}")

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
            print(f"Mean IoU on positive anchors: {mean_iou.item():.4f}")

            first_gt = gt_boxes[positive_mask][0]
            print(f"GT box (first positive): {first_gt}")

            positive_indices = torch.nonzero(positive_mask).squeeze(1)
            print("Positive anchor indices:", positive_indices.tolist())

            pos_boxes = decoded_boxes[positive_indices]
            gt_iou = box_iou(first_gt, pos_boxes)
            for anchor_idx, (box, iou) in zip(positive_indices.tolist(), zip(pos_boxes.tolist(), gt_iou.tolist())):
                print(f"  Anchor {anchor_idx}: box={box}, IoU={iou:.4f}")

            displayed_indices = torch.tensor(selected_indices, dtype=torch.long, device=decoded_boxes.device)
            top_iou = box_iou(first_gt, decoded_boxes[displayed_indices])
            top_iou_list = top_iou.tolist()
            if isinstance(top_iou_list, float):
                top_iou_list = [top_iou_list]
            print("IoU of displayed anchors relative to first GT:")
            for anchor_idx, score, box, iou in zip(
                displayed_indices.tolist(),
                selected_scores,
                decoded_boxes[displayed_indices].tolist(),
                top_iou_list
            ):
                print(f"  idx={anchor_idx:4d} score={score:.4f} box={box} IoU={iou:.4f}")
        else:
            print("No positive anchors in this sample.")

        debug_path = create_debug_visualization(
            dataset=dataset,
            sample_idx=idx,
            decoded_boxes=decoded_boxes,
            top_indices=top_indices,
            top_scores=top_scores,
            anchor_targets=anchor_targets,
            class_predictions=class_predictions.squeeze(-1),
            output_dir=Path("runs/logs") / "debug_images",
            device=device,
            comparison_detector=comparison_detector,
            comparison_label=args.compare_label,
            averaged_detector=averaged_detector,
            averaged_threshold=args.averaged_threshold
        )
        print(f"Saved debug visualization to {debug_path}")


if __name__ == "__main__":
    main()
