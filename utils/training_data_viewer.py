from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, cast

import cv2
import numpy as np
import pandas as pd
from pandas import Series as PandasSeries

# Allow running as `python utils/training_data_viewer.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import config, drawing
from utils.data_utils import load_image_boxes_from_csv


Box = tuple[int, int, int, int]
MetadataMap = Dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize training annotations from a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=config.DEFAULT_TRAIN_CSV,
        help="Path to CSV file (train.csv by default)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.DEFAULT_DATA_ROOT,
        help="Root directory prepended to CSV image paths",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Initial image index",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV UI and log summaries only",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images processed (0 = all)",
    )
    parser.add_argument(
        "--scale-width",
        type=int,
        default=0,
        help="Resize display width while keeping aspect ratio (0 = original)",
    )
    return parser.parse_args()


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    candidate = Path(path_str).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _group_metadata_by_image(csv_path: Path) -> MetadataMap:
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError(f"CSV {csv_path} missing required column 'image_path'")

    groups: MetadataMap = {}
    for image_path, group in df.groupby("image_path"):
        groups[str(image_path)] = group.reset_index(drop=True)
    return groups


def _format_summary_lines(group: pd.DataFrame | None) -> List[str]:
    if group is None or group.empty:
        return ["Annotations: 0"]

    lines: List[str] = [f"Annotations: {len(group)}"]

    if "earside" in group.columns:
        counts = (
            group["earside"]
            .fillna("unknown")
            .replace("", "unknown")
            .value_counts()
            .to_dict()
        )
        earside_str = ", ".join(f"{side}={count}" for side, count in counts.items())
        lines.append(f"Ear sides: {earside_str}")

    if "source" in group.columns:
        source_counts = group["source"].fillna("unknown").value_counts().head(3)
        summary = ", ".join(f"{name} ({count})" for name, count in source_counts.items())
        lines.append(f"Sources: {summary}")

    if "confidence" in group.columns:
        numeric_conf = pd.to_numeric(group["confidence"], errors="coerce")
        if not isinstance(numeric_conf, pd.Series):
            numeric_conf = pd.Series(numeric_conf)
        conf_series = cast(PandasSeries, numeric_conf).dropna()
        if not conf_series.empty:
            lines.append(
                f"Confidence: mean={conf_series.mean():.2f} min={conf_series.min():.2f} max={conf_series.max():.2f}"
            )
    return lines


def _log_image_summary(idx: int, total: int, image_path: str, group: pd.DataFrame | None) -> None:
    print(f"[{idx + 1}/{total}] {image_path}")
    for line in _format_summary_lines(group):
        print(f"  {line}")


def _scale_image(image: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return image
    height, width = image.shape[:2]
    if width == 0:
        return image
    scale = target_width / float(width)
    new_size = (target_width, int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _headless_iteration(
    image_paths: List[str],
    image_to_boxes: Dict[str, List[Box]],
    metadata: MetadataMap,
    data_root: Path,
    start_idx: int,
    max_images: int,
) -> None:
    count = max_images if max_images > 0 else len(image_paths)
    idx = start_idx
    processed = 0

    while processed < count and image_paths:
        image_path = image_paths[idx]
        full_path = data_root / image_path
        boxes = image_to_boxes.get(image_path, [])
        _log_image_summary(idx, len(image_paths), image_path, metadata.get(image_path))
        print(f"    Exists: {'yes' if full_path.exists() else 'no'} | Boxes: {len(boxes)}")
        idx = (idx + 1) % len(image_paths)
        processed += 1


def main() -> None:
    args = parse_args()

    csv_path = _resolve_path(args.csv, _REPO_ROOT)
    data_root = _resolve_path(args.data_root, _REPO_ROOT)

    image_paths, image_to_boxes = load_image_boxes_from_csv(str(csv_path))
    metadata = _group_metadata_by_image(csv_path)

    if not image_paths:
        raise SystemExit(f"No images found in CSV: {csv_path}")

    current_idx = max(0, min(args.start_idx, len(image_paths) - 1))

    if args.headless:
        _headless_iteration(
            image_paths,
            image_to_boxes,
            metadata,
            data_root,
            current_idx,
            args.max_images if args.max_images > 0 else len(image_paths),
        )
        return

    window_name = "Training Data Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_path = image_paths[current_idx]
        boxes = image_to_boxes.get(image_path, [])
        full_image_path = data_root / image_path

        if not full_image_path.exists():
            print(f"Missing image: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        image = cv2.imread(str(full_image_path))
        if image is None:
            print(f"Failed to load image: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        display = image.copy()
        drawing.draw_ground_truth_boxes(display, boxes, color=(0, 165, 255), label="Ann")
        drawing.draw_info_text(
            display,
            current_idx,
            len(image_paths),
            len(boxes),
            0,
            0,
            0.0,
            image_path,
            detection_only=False,
        )

        summary_lines = _format_summary_lines(metadata.get(image_path))
        _log_image_summary(current_idx, len(image_paths), image_path, metadata.get(image_path))
        for idx_line, line in enumerate(summary_lines):
            y = display.shape[0] - (len(summary_lines) - idx_line) * 20 - 10
            y = max(y, 30)
            cv2.putText(
                display,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        display = _scale_image(display, args.scale_width)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("a"), 81):
            current_idx = (current_idx - 1) % len(image_paths)
        elif key in (ord("d"), 83):
            current_idx = (current_idx + 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
