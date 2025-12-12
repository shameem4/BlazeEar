"""
Data preparation script for BlazeEar.

Parses all annotation formats under data/raw using existing utils:
- COCO JSON format
- CSV format
- PTS format
- LFPW TXT format

Creates a unified master CSV and train/val splits under data/splits.
Adds provenance metadata via `annotation_source` to flag whether boxes
derive from human GT labels, YOLO ear detector, pose detector, or a
combination of automated sources.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from pandas import Series as PandasSeries
from PIL import Image
from tqdm import tqdm

from utils.data_decoder import find_all_annotations, decode_all_annotations
from utils.data_utils import split_dataframe_by_images
from utils.iou import compute_iou_np

from ultralytics import YOLO

YOLO_DUPLICATE_IOU_THRESHOLD = 0.45
EAR_KEYPOINT_MIN_CONF = 0.75
EAR_BOX_WIDTH_RATIO = 0.12
EAR_BOX_HEIGHT_RATIO = 0.14
EAR_BOX_MIN_SIZE = 10
YOLO_DETECTION_POINT_PAD = 5.0

ANNOT_LABEL_GT = "GT"
ANNOT_LABEL_EAR = "EAR"
ANNOT_LABEL_POSE = "POSE"


def _format_annotation_label(parts: List[str]) -> str:
    """Return a canonical label string for a combination of annotation sources."""
    ordered: List[str] = []
    for part in parts:
        if part and part not in ordered:
            ordered.append(part)
    return "+".join(ordered) if ordered else "unknown"

PoseEntry = Tuple[np.ndarray, np.ndarray]


def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert various Ultralytics/torch objects to numpy arrays."""
    obj: Any = array_like

    data_attr = getattr(obj, "data", None)
    if data_attr is not None and not isinstance(obj, (np.ndarray, list, tuple)):
        obj = data_attr

    detach_fn = getattr(obj, "detach", None)
    if callable(detach_fn):
        obj = detach_fn()

    cpu_fn = getattr(obj, "cpu", None)
    if callable(cpu_fn):
        obj = cpu_fn()

    numpy_fn = getattr(obj, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn())

    return np.asarray(obj)


def _bbox_is_normalized(bbox: List[float]) -> bool:
    """Check if bbox coordinates are normalized between 0 and 1."""
    if len(bbox) != 4:
        return False
    return all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in bbox)


def _get_image_size(image_path: Path, cache: Dict[Path, Optional[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Cache image dimensions to avoid reopening files repeatedly."""
    if image_path in cache:
        return cache[image_path]

    if not image_path.exists():
        cache[image_path] = None
        return None

    try:
        with Image.open(image_path) as img:
            cache[image_path] = img.size
    except OSError:
        cache[image_path] = None
    return cache[image_path]


def _maybe_denormalize_bbox(bbox: List[float], image_path: Path, cache: Dict[Path, Optional[Tuple[int, int]]]) -> List[float]:
    if not _bbox_is_normalized(bbox):
        return bbox

    size = _get_image_size(image_path, cache)
    if not size:
        return bbox

    width, height = size
    x, y, w, h = bbox
    return [x * width, y * height, w * width, h * height]


def _get_pose_entries(image_path: Path, pose_model: YOLO, cache: Dict[str, List[PoseEntry]]) -> List[PoseEntry]:
    image_key = image_path.as_posix()
    if image_key in cache:
        return cache[image_key]

    detections: List[PoseEntry] = []
    results = pose_model(image_path, verbose=False)
    for result in results:
        keypoints_raw = getattr(result.keypoints, 'data', None) if result.keypoints is not None else None
        boxes_raw = getattr(result.boxes, 'xyxy', None) if result.boxes is not None else None
        if boxes_raw is None and result.boxes is not None:
            boxes_raw = result.boxes
        if keypoints_raw is None or boxes_raw is None:
            continue

        keypoints_data = _to_numpy(keypoints_raw)
        boxes = _to_numpy(boxes_raw)

        if keypoints_data.ndim == 2:
            keypoints_data = np.expand_dims(keypoints_data, axis=0)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)

        for keypoints, box in zip(keypoints_data, boxes):
            detections.append((keypoints, box))

    cache[image_key] = detections
    return detections


def _point_in_xywh(point: Tuple[float, float], box: Tuple[int, int, int, int], pad: float = 0.0) -> bool:
    x, y = point
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _point_in_xyxy(point: Tuple[float, float], box: np.ndarray, pad: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _is_contained_in_existing(candidate: Dict[str, Any], boxes: List[Tuple[int, int, int, int]], pad: int = 0) -> bool:
    """Check whether candidate box is entirely inside an existing box."""
    if not boxes:
        return False

    x1_c, y1_c, w_c, h_c = candidate['x1'], candidate['y1'], candidate['w'], candidate['h']
    x2_c = x1_c + w_c
    y2_c = y1_c + h_c

    for box in boxes:
        x1_b, y1_b, w_b, h_b = box
        x2_b = x1_b + w_b
        y2_b = y1_b + h_b
        if (x1_b - pad) <= x1_c and (y1_b - pad) <= y1_c and (x2_b + pad) >= x2_c and (y2_b + pad) >= y2_c:
            return True
    return False


def _boxes_are_near(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    max_center_distance_frac: float = 0.35,
    min_area_ratio: float = 0.35
) -> bool:
    """Return True when boxes are close in position and reasonably similar in size."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False

    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0

    center_distance = float(np.hypot(acx - bcx, acy - bcy))
    max_dim = max(aw, ah, bw, bh)
    if max_dim <= 0:
        return False
    center_close = center_distance <= max_center_distance_frac * max_dim

    area_a = aw * ah
    area_b = bw * bh
    area_max = max(area_a, area_b)
    if area_max <= 0:
        return False
    area_ratio = min(area_a, area_b) / area_max

    return center_close and area_ratio >= min_area_ratio


def _box_covers_target(
    candidate: Tuple[float, float, float, float],
    target: Tuple[float, float, float, float],
    min_coverage: float = 0.8
) -> bool:
    """Return True if candidate covers at least min_coverage of target area."""
    cx, cy, cw, ch = candidate
    tx, ty, tw, th = target
    if cw <= 0 or ch <= 0 or tw <= 0 or th <= 0:
        return False

    cx2 = cx + cw
    cy2 = cy + ch
    tx2 = tx + tw
    ty2 = ty + th

    inter_w = min(cx2, tx2) - max(cx, tx)
    if inter_w <= 0:
        return False
    inter_h = min(cy2, ty2) - max(cy, ty)
    if inter_h <= 0:
        return False

    inter_area = inter_w * inter_h
    target_area = tw * th
    if target_area <= 0:
        return False

    return (inter_area / target_area) >= min_coverage


def _detection_has_pose_support(
    det_box: np.ndarray,
    pose_entries: List[PoseEntry],
    pad: float = YOLO_DETECTION_POINT_PAD
) -> bool:
    """Return True when any confident pose ear keypoint falls inside det_box."""
    if not pose_entries:
        return False

    for keypoints, _ in pose_entries:
        if keypoints.shape[0] < 5:
            continue
        for idx in (3, 4):  # left/right ear indices in BlazePose
            point = keypoints[idx]
            confidence = float(point[2]) if point.shape[0] >= 3 else 0.0
            if confidence < EAR_KEYPOINT_MIN_CONF:
                continue
            coords = (float(point[0]), float(point[1]))
            if _point_in_xyxy(coords, det_box, pad):
                return True
    return False


def _build_pose_bbox(
    point: Tuple[float, float],
    face_box: np.ndarray,
    image_path: Path,
    cache: Dict[Path, Optional[Tuple[int, int]]]
) -> Optional[Tuple[int, int, int, int]]:
    face_width = max(1.0, float(face_box[2] - face_box[0]))
    face_height = max(1.0, float(face_box[3] - face_box[1]))
    box_width = max(EAR_BOX_MIN_SIZE, face_width * EAR_BOX_WIDTH_RATIO)
    box_height = max(EAR_BOX_MIN_SIZE, face_height * EAR_BOX_HEIGHT_RATIO)

    x = float(point[0]) - box_width / 2.0
    y = float(point[1]) - box_height / 2.0

    size = _get_image_size(image_path, cache)
    if size:
        img_w, img_h = size
        if img_w <= 0 or img_h <= 0:
            return None
        x = max(0.0, min(x, img_w - 1.0))
        y = max(0.0, min(y, img_h - 1.0))
        available_w = img_w - x
        available_h = img_h - y
        if available_w <= 0 or available_h <= 0:
            return None
        box_width = min(box_width, available_w)
        box_height = min(box_height, available_h)

    width_int = max(1, int(round(box_width)))
    height_int = max(1, int(round(box_height)))
    x_int = int(round(x))
    y_int = int(round(y))
    return (x_int, y_int, width_int, height_int)


def _maybe_add_pose_boxes_for_image(
    image_key: str,
    rel_path: str,
    image_path: Path,
    pose_entries: List[PoseEntry],
    detection_boxes: np.ndarray,
    gt_boxes: DefaultDict[str, List[Tuple[int, int, int, int]]],
    cache: Dict[Path, Optional[Tuple[int, int]]],
    all_rows: List[Dict[str, Any]],
    source_name: str,
    pose_augmented: Set[str]
) -> None:
    if image_key in pose_augmented or not pose_entries:
        pose_augmented.add(image_key)
        return

    pose_augmented.add(image_key)
    detections = detection_boxes if detection_boxes.size else np.empty((0, 4), dtype=np.float32)
    new_boxes = 0

    for keypoints, face_box in pose_entries:
        if keypoints.shape[0] < 5:
            continue
        left_point = keypoints[3]
        right_point = keypoints[4]
        for point, side in ((left_point, 'left'), (right_point, 'right')):
            confidence = float(point[2]) if point.shape[0] >= 3 else 0.0
            if confidence < EAR_KEYPOINT_MIN_CONF:
                continue
            coords = (float(point[0]), float(point[1]))
            if any(_point_in_xywh(coords, box) for box in gt_boxes[rel_path]):
                continue
            if any(_point_in_xyxy(coords, det_box, YOLO_DETECTION_POINT_PAD) for det_box in detections):
                continue
            bbox = _build_pose_bbox(coords, face_box, image_path, cache)
            if bbox is None:
                continue
            candidate = {
                'image_path': rel_path,
                'x1': bbox[0],
                'y1': bbox[1],
                'w': bbox[2],
                'h': bbox[3],
                'earside': side,
                'source': f"{source_name}_pose_aug",
                'annotation_source': _format_annotation_label([ANNOT_LABEL_POSE]),
                'confidence': 1.0,
            }
            existing = gt_boxes[rel_path]
            if _is_contained_in_existing(candidate, existing):
                continue
            candidate_box_tuple = (
                float(candidate['x1']),
                float(candidate['y1']),
                float(candidate['w']),
                float(candidate['h'])
            )
            skip_candidate = False
            for box in existing:
                existing_box_tuple = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3])
                )
                if _box_covers_target(candidate_box_tuple, existing_box_tuple):
                    skip_candidate = True
                    break
                if _boxes_are_near(
                    candidate_box_tuple,
                    existing_box_tuple,
                    max_center_distance_frac=0.55
                ):
                    skip_candidate = True
                    break
            if skip_candidate:
                continue
            all_rows.append(candidate)
            gt_boxes[rel_path].append(bbox)
            new_boxes += 1

    # if new_boxes > 0:
    #     print(f"    Added {new_boxes} pose-derived box(es) for {rel_path}")
def collect_all_annotations(raw_dir: Path) -> pd.DataFrame:
    """
    Find and parse all annotation files under data/raw using utils.data_decoder.

    Returns:
        DataFrame with columns: image_path, x1, y1, w, h, earside,
        source, annotation_source, confidence
    """
    # Find all annotation sources
    annotation_sources = find_all_annotations(str(raw_dir))
    print(f"Found {len(annotation_sources)} annotation sources")

    all_rows: List[Dict[str, Any]] = []
    missing_images = 0
    pose_model = YOLO('model_weights/yolo11x-pose.pt')
    ear_detector = YOLO('model_weights/yolov11_ear_detector.pt')
    image_size_cache: Dict[Path, Optional[Tuple[int, int]]] = {}
    ear_detection_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    gt_boxes_by_image: DefaultDict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    pose_results_cache: Dict[str, List[PoseEntry]] = {}
    rel_path_cache: Dict[str, str] = {}
    image_source_map: Dict[str, str] = {}
    pose_augmented_images: Set[str] = set()

    for ann_file, ann_type, image_dir in tqdm(annotation_sources, desc="Processing sources"):
        if ann_type == 'images_only':
            # Skip image-only folders (no bbox annotations)
            continue

        source_name = Path(image_dir).parent.name if ann_type != 'csv' else Path(image_dir).name

        try:
            annotations = decode_all_annotations(ann_file, ann_type, image_dir)
            annotation_counts: DefaultDict[str, int] = defaultdict(int)
            for ann in annotations:
                annotation_counts[Path(ann['image_path']).as_posix()] += 1

            processed_counts: DefaultDict[str, int] = defaultdict(int)

            for ann in annotations:
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue

                bbox = _maybe_denormalize_bbox(bbox, Path(ann['image_path']), image_size_cache)
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                image_path = Path(ann['image_path'])
                image_key = image_path.as_posix()
                if not image_path.exists():
                    missing_images += 1
                    continue

                try:
                    rel_path = image_path.relative_to(raw_dir)
                except ValueError:
                    rel_path = image_path

                rel_path_str = rel_path.as_posix()
                rel_path_cache.setdefault(image_key, rel_path_str)
                image_source_map.setdefault(image_key, source_name)

                pose_entries = _get_pose_entries(image_path, pose_model, pose_results_cache)
                earside = ''
                for keypoints, box in pose_entries:
                    if keypoints.shape[0] <= 4:
                        continue
                    l_ear, r_ear = keypoints[3], keypoints[4]
                    if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                        if l_ear[2] > 0 and r_ear[2] > 0:
                            earside = 'left' if l_ear[0] < r_ear[0] else 'right'
                        elif l_ear[2] > 0:
                            earside = 'left'
                        elif r_ear[2] > 0:
                            earside = 'right'
                        else:
                            earside = ''
                        break

                candidate = {
                    'image_path': rel_path_str,
                    'x1': int(round(x)),
                    'y1': int(round(y)),
                    'w': int(round(w)),
                    'h': int(round(h)),
                    'earside': earside,
                    'source': source_name,
                    'annotation_source': _format_annotation_label([ANNOT_LABEL_GT]),
                    'confidence': 1.0,
                }

                if _is_contained_in_existing(candidate, gt_boxes_by_image[rel_path_str]):
                    continue

                all_rows.append(candidate)

                gt_boxes_by_image[rel_path_str].append(
                    (candidate['x1'], candidate['y1'], candidate['w'], candidate['h'])
                )

                if image_path not in ear_detection_cache:
                    ear_results = ear_detector.predict(
                        source=str(image_path),
                        imgsz=640,
                        conf=0.25,
                        verbose=False
                    )

                    if not ear_results:
                        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                        scores = np.empty((0,), dtype=np.float32)
                    else:
                        res = ear_results[0]
                        if res.boxes is None or len(res.boxes) == 0:
                            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                            scores = np.empty((0,), dtype=np.float32)
                        else:
                            boxes_tensor = res.boxes.xyxy if hasattr(res.boxes, 'xyxy') else res.boxes
                            scores_tensor = res.boxes.conf if hasattr(res.boxes, 'conf') else res.boxes
                            boxes_xyxy = _to_numpy(boxes_tensor)
                            scores = _to_numpy(scores_tensor)

                    ear_detection_cache[image_path] = (boxes_xyxy, scores)

                boxes_xyxy, scores = ear_detection_cache[image_path]
                gt_boxes_for_image = gt_boxes_by_image[rel_path_str]
                has_gt_reference = len(gt_boxes_for_image) > 0

                for det_box, score in zip(boxes_xyxy, scores):
                    det_x1, det_y1, det_x2, det_y2 = det_box
                    det_w = det_x2 - det_x1
                    det_h = det_y2 - det_y1
                    if det_w <= 0 or det_h <= 0:
                        continue

                    det_xywh = np.array([det_x1, det_y1, det_w, det_h], dtype=np.float32)
                    det_box_tuple = (float(det_x1), float(det_y1), float(det_w), float(det_h))
                    overlaps_gt = False
                    for gt_box in gt_boxes_for_image:
                        gt_array = np.array(gt_box, dtype=np.float32)
                        iou = compute_iou_np(gt_array, det_xywh, box1_format="xywh", box2_format="xywh")
                        if iou >= YOLO_DUPLICATE_IOU_THRESHOLD:
                            overlaps_gt = True
                            break
                        gt_box_tuple = (
                            float(gt_box[0]),
                            float(gt_box[1]),
                            float(gt_box[2]),
                            float(gt_box[3])
                        )
                        if _boxes_are_near(det_box_tuple, gt_box_tuple):
                            overlaps_gt = True
                            break

                    if overlaps_gt:
                        continue

                    # Treat YOLO-only detections without GT or pose support as false positives
                    if not has_gt_reference:
                        continue
                    if not _detection_has_pose_support(det_box, pose_entries):
                        continue

                    candidate_aug = {
                        'image_path': rel_path_str,
                        'x1': int(round(det_x1)),
                        'y1': int(round(det_y1)),
                        'w': int(round(det_w)),
                        'h': int(round(det_h)),
                        'earside': 'unknown',
                        'source': f"{source_name}_yolo11_aug",
                        'annotation_source': _format_annotation_label([
                            ANNOT_LABEL_GT,
                            ANNOT_LABEL_EAR
                        ]),
                        'confidence': float(score),
                    }

                    if _is_contained_in_existing(candidate_aug, gt_boxes_by_image[rel_path_str]):
                        continue

                    all_rows.append(candidate_aug)

                processed_counts[image_key] += 1
                if processed_counts[image_key] == annotation_counts[image_key]:
                    rel_for_image = rel_path_cache.get(image_key, rel_path_str)
                    pose_entries_for_image = pose_results_cache.get(image_key, [])
                    source_label = image_source_map.get(image_key, source_name)
                    _maybe_add_pose_boxes_for_image(
                        image_key,
                        rel_for_image,
                        image_path,
                        pose_entries_for_image,
                        boxes_xyxy,
                        gt_boxes_by_image,
                        image_size_cache,
                        all_rows,
                        source_label,
                        pose_augmented_images,
                    )

            print(f"  {ann_type}: {Path(image_dir).relative_to(raw_dir)} -> {len(annotations)} annotations")

        except Exception as e:
            print(f"  Error processing {image_dir}: {e}")

    if missing_images:
        print(f"Skipped {missing_images} annotations referencing missing images")

    if not all_rows:
        columns = pd.Index([
            'image_path',
            'x1',
            'y1',
            'w',
            'h',
            'earside',
            'source',
            'annotation_source',
            'confidence'
        ])
        return pd.DataFrame([], columns=columns)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=['image_path', 'x1', 'y1', 'w', 'h']).reset_index(drop=True)

    return df


def verify_images_exist(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Verify that image files exist and filter out missing ones."""
    valid_rows = []
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        image_path = raw_dir / row['image_path']
        if image_path.exists():
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} images not found, excluding from dataset")

    return pd.DataFrame(valid_rows).reset_index(drop=True)


def print_statistics(df: pd.DataFrame, split_name: str = "Dataset") -> None:
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"{split_name} Statistics")
    print(f"{'='*60}")

    column = cast(PandasSeries, df['image_path'])
    num_images = int(column.nunique())
    num_ears = int(len(df))

    print(f"Total images: {num_images:,}")
    print(f"Total ear annotations: {num_ears:,}")
    print(f"Average ears per image: {num_ears/max(num_images,1):.2f}")

    # Earside breakdown
    left_count = (df['earside'] == 'left').sum()
    right_count = (df['earside'] == 'right').sum()
    unknown_count = (df['earside'] == '').sum()

    print(f"\nEar side breakdown:")
    print(f"  Left ears:    {left_count:,} ({100*left_count/max(num_ears,1):.1f}%)")
    print(f"  Right ears:   {right_count:,} ({100*right_count/max(num_ears,1):.1f}%)")
    print(f"  Unknown side: {unknown_count:,} ({100*unknown_count/max(num_ears,1):.1f}%)")

    # Source breakdown
    if 'source' in df.columns:
        print(f"\nSource breakdown:")
        for source, count in df['source'].value_counts().items():
            img_paths = cast(PandasSeries, df.loc[df['source'] == source, 'image_path'])
            img_count = int(img_paths.nunique())
            print(f"  {source}: {count:,} ears from {img_count:,} images")

    if 'annotation_source' in df.columns:
        print("\nAnnotation source breakdown:")
        for source, count in df['annotation_source'].value_counts().items():
            print(f"  {source}: {count:,}")

    # Box size statistics
    print(f"\nBounding box statistics:")
    print(f"  Width  - min: {df['w'].min()}, max: {df['w'].max()}, mean: {df['w'].mean():.1f}")
    print(f"  Height - min: {df['h'].min()}, max: {df['h'].max()}, mean: {df['h'].mean():.1f}")
    area = df['w'] * df['h']
    print(f"  Area   - min: {area.min()}, max: {area.max()}, mean: {area.mean():.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ear detection dataset from raw annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory containing raw data folders")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                        help="Directory for output CSV files")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of images for validation split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification that image files exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args.no_verify)

    script_dir = Path(__file__).parent
    raw_dir = script_dir / args.raw_dir
    output_dir = script_dir / args.output_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    print(f"Scanning for annotations in: {raw_dir}\n")

    # Collect all annotations using existing decoder
    df = collect_all_annotations(raw_dir)

    if df.empty:
        print("No annotations found!")
        return

    print(f"\nTotal annotations found: {len(df):,}")

    # Verify images exist
    if not args.no_verify:
        print("Verifying that image files exist...")
        df = verify_images_exist(df, raw_dir)

    if df.empty:
        print("No valid annotations remaining!")
        return

    # Save master list
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "master.csv"
    df.to_csv(master_path, index=False)
    print(f"\nMaster CSV saved to: {master_path}")

    print_statistics(df, "Master Dataset")

    # Split into train/val
    train_df, val_df = split_dataframe_by_images(
        df, val_fraction=args.val_fraction, random_seed=args.seed
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nTrain CSV saved to: {train_path}")
    print(f"Val CSV saved to: {val_path}")

    print_statistics(train_df, "Training Set")
    print_statistics(val_df, "Validation Set")

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
