#!/usr/bin/env python3
"""Interactive YOLOv11 ear detection demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

# Allow running as `python utils/yolo11_demo.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.data_utils import load_image_boxes_from_csv

DEFAULT_WEIGHTS = "runs/yolo11/ear-detector/weights/best.pt"
DEFAULT_CSV = "data/splits/train.csv"
DEFAULT_DATA_ROOT = "data/raw"
WINDOW_NAME = "YOLOv11 Ear Demo"


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse detections from a fine-tuned YOLOv11 ear model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to YOLOv11 weights")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="CSV file with image paths")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Root folder for CSV-relative images")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="0", help="Torch device passed to Ultralytics (e.g. '0' or 'cpu')")
    parser.add_argument("--start-idx", type=int, default=0, help="Image index to start from")
    parser.add_argument("--headless", action="store_true", help="Skip OpenCV UI and log detections to stdout")
    parser.add_argument("--max-images", type=int, default=0, help="Limit iterations in headless mode (0 = entire set)")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box thickness in pixels")
    parser.add_argument("--font-scale", type=float, default=0.6, help="Font scale for confidence labels")
    parser.add_argument("--font-thickness", type=int, default=2, help="Font thickness for labels")
    return parser.parse_args()


def load_image_list(csv_path: Path) -> List[str]:
    image_paths, _ = load_image_boxes_from_csv(str(csv_path))
    return image_paths


def format_detection_summary(index: int, total: int, num_dets: int, image_path: str) -> str:
    return f"[{index}/{total}] {image_path} detections={num_dets}"


def _to_numpy(array_like: Any) -> np.ndarray:
    obj = array_like
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    thickness: int,
    font_scale: float,
    font_thickness: int
) -> np.ndarray:
    display = image.copy()
    for bbox, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"ear {score:.2f}"
        cv2.putText(
            display,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
            cv2.LINE_AA
        )
    return display


def run_demo(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - requires runtime dependency
        raise SystemExit(
            "Ultralytics must be installed. Please run 'pip install ultralytics' first."
        ) from exc

    repo_root = _REPO_ROOT
    weights = _resolve_path(args.weights, repo_root)
    csv_path = _resolve_path(args.csv, repo_root)
    data_root = _resolve_path(args.data_root, repo_root)

    image_paths = load_image_list(csv_path)
    if not image_paths:
        raise SystemExit(f"No image paths found in {csv_path}")

    current_idx = max(0, min(args.start_idx, len(image_paths) - 1))

    print(f"Loading YOLO model from {weights}")
    model = YOLO(str(weights))

    remaining = args.max_images if args.max_images > 0 else len(image_paths)
    window_active = not args.headless
    if window_active:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\nControls:\n  A / Left Arrow  - Previous image\n  D / Right Arrow - Next image\n  Q / ESC         - Quit\n")

    while True:
        rel_path = image_paths[current_idx]
        full_path = data_root / rel_path

        if not full_path.exists():
            print(f"Warning: missing image {full_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            if args.headless:
                remaining -= 1
                if remaining <= 0:
                    break
            continue

        image = cv2.imread(str(full_path))
        if image is None:
            print(f"Warning: failed to read {full_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            if args.headless:
                remaining -= 1
                if remaining <= 0:
                    break
            continue

        results = model.predict(
            source=image,
            imgsz=args.img_size,
            conf=args.conf,
            device=args.device,
            verbose=False
        )

        boxes_xyxy: np.ndarray
        scores: np.ndarray
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_xyxy = _to_numpy(results[0].boxes.xyxy)
            scores = _to_numpy(results[0].boxes.conf)
        else:
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)

        summary = format_detection_summary(current_idx + 1, len(image_paths), len(boxes_xyxy), rel_path)

        if args.headless:
            print(summary)
            current_idx = (current_idx + 1) % len(image_paths)
            remaining -= 1
            if remaining <= 0:
                break
            continue

        display = draw_boxes(
            image,
            boxes_xyxy,
            scores,
            thickness=args.line_thickness,
            font_scale=args.font_scale,
            font_thickness=args.font_thickness
        )

        cv2.putText(
            display,
            summary,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        if key in (ord('a'), 81):
            current_idx = (current_idx - 1) % len(image_paths)
        elif key in (ord('d'), 83):
            current_idx = (current_idx + 1) % len(image_paths)

    if window_active:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
