#!/usr/bin/env python3
"""Fine-tune an Ultralytics YOLOv11 detector on ear metadata stored in CSV files.

This script mirrors the existing BlazeEar CSV metadata pipeline and performs three steps:
1. Convert the CSV annotations under ``data/splits`` into a YOLO-compatible directory that
   contains ``images/{split}`` and ``labels/{split}`` trees.
2. Generate a minimal dataset YAML file that Ultralytics expects.
3. Launch a YOLOv11 fine-tuning run starting from a pretrained checkpoint
   (``yolo11n.pt`` by default).

Example:
    python finetune_yolov11.py \
        --train-csv data/splits/train.csv \
        --val-csv data/splits/val.csv \
        --data-root data/raw \
        --output-dir data/yolo11_ears \
        --weights yolov11m.pt \
        --epochs 100
"""
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "data/yolo11_ears"
DEFAULT_YOLO_WEIGHTS = "model_weights/yolo11n.pt"
DEFAULT_PROJECT = "runs/yolo11"
DEFAULT_RUN_NAME = "ear-detector"
CLASS_NAME = "ear"


@dataclass
class SplitStats:
    """Simple bookkeeping for dataset preparation."""

    split: str
    images: int = 0
    boxes: int = 0
    missing_images: List[str] = field(default_factory=list)
    skipped_boxes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv11 on ear annotations defined by CSV metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-csv", default="data/splits/train.csv", help="Training metadata CSV")
    parser.add_argument("--val-csv", default="data/splits/val.csv", help="Validation metadata CSV")
    parser.add_argument("--data-root", default="data/raw", help="Root directory that contains the raw images referenced by the CSV files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where YOLO-formatted images/labels will be written")
    parser.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS, help="Ultralytics pretrained weights to fine-tune (auto-downloaded if missing)")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size for YOLO (square)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers used by Ultralytics")
    parser.add_argument("--device", default=("0" if torch.cuda.is_available() else "cpu"), help="Computation device string understood by Ultralytics (e.g., '0', '0,1', 'cpu')")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Ultralytics project directory for logging")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Name for the Ultralytics run (sub-directory under project)")
    parser.add_argument("--link-images", action="store_true", help="Attempt to hard-link images instead of copying when building the YOLO dataset")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation if the output directory already exists")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Warm start from project/name checkpoints instead of default weights. "
            "Starts a new Ultralytics run initialized with the previous best/last.pt."
        )
    )
    return parser.parse_args()


def _locate_best_checkpoint(project: str, name: str) -> Path | None:
    run_dir = Path(project) / name / "weights"
    print(f"Looking for best checkpoint in {run_dir}")
    best = run_dir / "best.pt"
    if best.exists():
        print(f"Found best checkpoint: {best}")
        return best
    last = run_dir / "last.pt"
    return last if last.exists() else None


def copy_or_link(src: Path, dst: Path, use_link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if use_link:
        try:
            os.link(src, dst)
            return
        except OSError:
            # Fall back to copying if the filesystem does not support links.
            pass

    shutil.copy2(src, dst)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def convert_row(row, width: int, height: int) -> str | None:
    w = max(float(row.w), 1e-6)
    h = max(float(row.h), 1e-6)
    x_center = clamp((float(row.x1) + w * 0.5) / width)
    y_center = clamp((float(row.y1) + h * 0.5) / height)
    w_norm = clamp(w / width)
    h_norm = clamp(h / height)

    if w_norm <= 0 or h_norm <= 0:
        return None

    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def convert_split(
    csv_path: Path,
    split: str,
    data_root: Path,
    output_dir: Path,
    link_images: bool
) -> SplitStats:
    df = pd.read_csv(csv_path)
    stats = SplitStats(split=split)

    if df.empty:
        print(f"Warning: {csv_path} is empty. Ultralytics will see zero samples for {split}.")
        return stats

    grouped = df.groupby("image_path", sort=False)
    num_groups = grouped.ngroups
    image_root = output_dir / "images" / split
    label_root = output_dir / "labels" / split

    for image_rel, rows in tqdm(grouped, total=num_groups, desc=f"Preparing {split}"):
        rel_path = str(image_rel)
        src = data_root / rel_path
        if not src.exists():
            stats.missing_images.append(str(src))
            continue

        stats.images += 1
        stats.boxes += len(rows)

        copy_or_link(src, image_root / rel_path, link_images)

        label_path = label_root / Path(rel_path).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src) as img:
            width, height = img.size

        yolo_lines: List[str] = []
        for row in rows.itertuples(index=False):
            line = convert_row(row, width, height)
            if line:
                yolo_lines.append(line)
            else:
                stats.skipped_boxes += 1

        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    return stats


def prepare_dataset(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "dataset.yaml"

    if args.skip_prep and yaml_path.exists():
        print(f"Skipping dataset preparation, reusing {yaml_path}")
        return yaml_path

    stats_train = convert_split(Path(args.train_csv), "train", Path(args.data_root), output_dir, args.link_images)
    stats_val = convert_split(Path(args.val_csv), "val", Path(args.data_root), output_dir, args.link_images)

    print(
        f"Prepared {stats_train.images} train images/{stats_train.boxes} boxes; "
        f"{stats_val.images} val images/{stats_val.boxes} boxes"
    )
    if stats_train.skipped_boxes or stats_val.skipped_boxes:
        print(
            f"Skipped {stats_train.skipped_boxes + stats_val.skipped_boxes} boxes with invalid geometry."
        )
    if stats_train.missing_images or stats_val.missing_images:
        total_missing = len(stats_train.missing_images) + len(stats_val.missing_images)
        print(f"Warning: {total_missing} images were referenced in CSVs but not found under {args.data_root}.")

    yaml_contents = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n  0: {CLASS_NAME}\n"
    )
    yaml_path.write_text(yaml_contents, encoding="utf-8")
    print(f"Wrote YOLO dataset definition to {yaml_path}")
    return yaml_path


def run_training(args: argparse.Namespace, data_yaml: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required. Install it with 'pip install ultralytics' before running this script."
        ) from exc

    resume_arg = False
    if args.resume:
        best_ckpt = _locate_best_checkpoint(args.project, args.name)
        if not best_ckpt:
            raise SystemExit(
                "--resume was specified but no checkpoint was found in project/name/weights. "
                "Start a fresh run first or specify the correct project/name."
            )
        model_path = str(best_ckpt)
        print(
            f"Continuing training from checkpoint: {best_ckpt}\n"
            "Ultralytics will start a *new* run initialized from these weights."
        )
    else:
        model_path = args.weights

    model = YOLO(model_path)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=resume_arg
    )

    best = getattr(results, "best", None)
    if best:
        print(f"Fine-tuning complete. Best weights saved to: {best}")
    else:
        print("Fine-tuning complete. Inspect the Ultralytics run directory for checkpoints.")


def main() -> None:
    args = parse_args()
    data_yaml = prepare_dataset(args)
    run_training(args, data_yaml)


if __name__ == "__main__":
    main()
