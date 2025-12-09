from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, cast

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - optional for headless servers
    tk = None
    filedialog = None

# Allow running as `python utils/image_annotation_viewer.py` from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_decoder import decode_annotation, find_annotation

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_MAP = {
    "coco": (0, 0, 255),
    "csv": (0, 255, 0),
    "pts": (255, 0, 255),
    "lfpw_txt": (0, 165, 255),
}
DEFAULT_BOX_COLOR = (255, 255, 255)
ArrayLike = NDArray[np.uint8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse a folder of images with annotations using OpenCV controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default=None,
        help="Folder that contains images (falls back to folder picker if omitted)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include subdirectories when gathering images",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Initial image index to show",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV UI and just log annotation info (useful for remote sessions)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit the number of images processed in headless mode (0 = all)",
    )
    parser.add_argument(
        "--scale-width",
        type=int,
        default=0,
        help="Resize display width while preserving aspect ratio (0 = original size)",
    )
    return parser.parse_args()


def pick_directory() -> Path:
    if tk is None or filedialog is None:
        raise SystemExit("tkinter is unavailable; pass --folder to choose a directory explicitly.")

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select image folder")
    if not folder:
        raise SystemExit("No folder selected.")
    return Path(folder)


def resolve_folder(folder_arg: str | None) -> Path:
    if folder_arg:
        folder = Path(folder_arg).expanduser()
    else:
        folder = pick_directory()

    if not folder.exists():
        raise SystemExit(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise SystemExit(f"Path is not a directory: {folder}")
    return folder


def collect_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        paths = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(paths)


def draw_annotations_on_image(
    image_path: Path,
    annotations: Sequence[dict],
    annotation_type: str | None
) -> ArrayLike:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cast(ArrayLike, img)

    color = COLOR_MAP.get(annotation_type or "", DEFAULT_BOX_COLOR)

    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox:
            x, y, w, h = bbox
            pt1 = (int(round(x)), int(round(y)))
            pt2 = (int(round(x + w)), int(round(y + h)))
            cv2.rectangle(image, pt1, pt2, color, 2)

        keypoints = ann.get("keypoints")
        if keypoints:
            for idx in range(0, len(keypoints), 3):
                kx = int(round(keypoints[idx]))
                ky = int(round(keypoints[idx + 1]))
                visibility = keypoints[idx + 2] if idx + 2 < len(keypoints) else 2
                if visibility > 0:
                    cv2.circle(image, (kx, ky), 3, color, -1)

    info_lines = [
        image_path.name,
        f"Annotation: {annotation_type or 'none'}",
        f"Detections: {len(annotations)}",
        "Controls: [A] prev  [D] next  [Q] quit",
    ]
    draw_text_overlay(image, info_lines)
    return image


def draw_text_overlay(image: ArrayLike, lines: Sequence[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25


def scale_image(image: ArrayLike, target_width: int) -> ArrayLike:
    if target_width <= 0:
        return image
    h, w = image.shape[:2]
    if w == 0:
        return image
    scale = target_width / float(w)
    new_size = (target_width, int(round(h * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return cast(ArrayLike, resized)


def main() -> None:
    args = parse_args()
    folder = resolve_folder(args.folder)
    image_paths = collect_images(folder, args.recursive)

    if not image_paths:
        raise SystemExit(f"No images found under {folder} (extensions: {', '.join(SUPPORTED_EXTENSIONS)})")

    current_idx = max(0, min(args.start_idx, len(image_paths) - 1))

    if args.headless:
        remaining = args.max_images if args.max_images > 0 else len(image_paths)
        print(f"Headless mode enabled; iterating over {remaining} image(s).")
        while remaining > 0:
            image_path = image_paths[current_idx]
            annotation_path, annotation_type = find_annotation(str(image_path))
            annotations = (
                decode_annotation(annotation_path, str(image_path), annotation_type)
                if annotation_path
                else []
            )
            print(
                f"[{current_idx + 1}/{len(image_paths)}] {image_path.name} "
                f"annotation={annotation_type or 'none'} count={len(annotations)}"
            )
            current_idx = (current_idx + 1) % len(image_paths)
            remaining -= 1
        return

    window_name = "Annotation Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_path = image_paths[current_idx]
        annotation_path, annotation_type = find_annotation(str(image_path))
        annotations = (
            decode_annotation(annotation_path, str(image_path), annotation_type)
            if annotation_path
            else []
        )

        display = draw_annotations_on_image(image_path, annotations, annotation_type)
        display = scale_image(display, args.scale_width)
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
