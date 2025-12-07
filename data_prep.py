"""
Data preparation script for BlazeEar.

Parses all annotation formats under data/raw using existing utils:
- COCO JSON format
- CSV format
- PTS format
- LFPW TXT format

Creates a unified master CSV and train/val splits under data/splits.
"""
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from utils.data_decoder import find_all_annotations, decode_all_annotations
from utils.data_utils import split_dataframe_by_images


def detect_earside(text: str) -> Optional[str]:
    """
    Detect left/right ear from text (filename, category name, etc.).

    Returns 'left', 'right', or None if not detected.
    """
    text_lower = text.lower()

    left_patterns = [r'\bleft\b', r'\bl_ear\b', r'\bl-ear\b', r'_l\.', r'^l_', r'_l_']
    right_patterns = [r'\bright\b', r'\br_ear\b', r'\br-ear\b', r'_r\.', r'^r_', r'_r_']

    for pattern in left_patterns:
        if re.search(pattern, text_lower):
            return 'left'

    for pattern in right_patterns:
        if re.search(pattern, text_lower):
            return 'right'

    return None


def collect_all_annotations(raw_dir: Path) -> pd.DataFrame:
    """
    Find and parse all annotation files under data/raw using utils.data_decoder.

    Returns:
        DataFrame with columns: image_path, x1, y1, w, h, earside, source
    """
    # Find all annotation sources
    annotation_sources = find_all_annotations(str(raw_dir))
    print(f"Found {len(annotation_sources)} annotation sources")

    all_rows = []

    for ann_file, ann_type, image_dir in tqdm(annotation_sources, desc="Processing sources"):
        if ann_type == 'images_only':
            # Skip image-only folders (no bbox annotations)
            continue

        source_name = Path(image_dir).parent.name if ann_type != 'csv' else Path(image_dir).name

        try:
            annotations = decode_all_annotations(ann_file, ann_type, image_dir)

            for ann in annotations:
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                image_path = Path(ann['image_path'])

                # Make path relative to raw_dir
                try:
                    rel_path = image_path.relative_to(raw_dir)
                except ValueError:
                    rel_path = image_path

                # Detect earside from filename
                earside = detect_earside(str(rel_path)) or ''

                all_rows.append({
                    'image_path': rel_path.as_posix(),
                    'x1': int(round(x)),
                    'y1': int(round(y)),
                    'w': int(round(w)),
                    'h': int(round(h)),
                    'earside': earside,
                    'source': source_name,
                })

            print(f"  {ann_type}: {Path(image_dir).relative_to(raw_dir)} -> {len(annotations)} annotations")

        except Exception as e:
            print(f"  Error processing {image_dir}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=['image_path', 'x1', 'y1', 'w', 'h', 'earside', 'source'])

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

    num_images = df['image_path'].nunique()
    num_ears = len(df)

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
            img_count = df[df['source'] == source]['image_path'].nunique()
            print(f"  {source}: {count:,} ears from {img_count:,} images")

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
