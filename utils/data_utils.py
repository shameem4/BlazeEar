"""
Utilities for loading annotation CSVs and creating train/val splits.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple, cast

import numpy as np
import pandas as pd

Box = Tuple[int, int, int, int]


def load_image_boxes_from_csv(csv_path: str | Path) -> tuple[list[str], dict[str, list[Box]]]:
    """
    Load a CSV containing bounding boxes and group them by image.

    Args:
        csv_path: Path to CSV file with columns: image_path, x1, y1, w, h

    Returns:
        (sorted_image_paths, mapping image_path -> list of (x1, y1, w, h))
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"image_path", "x1", "y1", "w", "h"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    grouped: DefaultDict[str, List[Box]] = defaultdict(list)
    for _, row in df.sort_values("image_path").iterrows():
        if not isinstance(row, pd.Series):
            continue
        image_path = str(row["image_path"])
        grouped[image_path].append(
            (int(row["x1"]), int(row["y1"]), int(row["w"]), int(row["h"]))
        )

    image_paths = sorted(grouped.keys())
    return image_paths, dict(grouped)


def split_dataframe_by_images(
    df: pd.DataFrame,
    image_column: str = "image_path",
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an annotation DataFrame into train/val sets grouped by image.

    Args:
        df: DataFrame containing at least `image_column`
        image_column: Column used to identify unique images
        val_fraction: Fraction of images to allocate to validation
        random_seed: RNG seed for shuffling

    Returns:
        (train_df, val_df) with indices reset
    """
    if image_column not in df.columns:
        raise ValueError(f"Column '{image_column}' not found in DataFrame")

    image_ids = df[image_column].drop_duplicates().tolist()
    if not image_ids:
        raise ValueError("No images found to split.")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(image_ids)

    val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
    n_val = int(round(len(image_ids) * val_fraction))
    if len(image_ids) > 1:
        if n_val == 0:
            n_val = 1
        elif n_val >= len(image_ids):
            n_val = len(image_ids) - 1

    val_ids = set(image_ids[:n_val])
    val_id_list = list(val_ids)
    val_mask = df[image_column].isin(val_id_list)
    val_df = df[val_mask].copy().reset_index(drop=True)
    train_df = df[~val_mask].copy().reset_index(drop=True)
    return cast(pd.DataFrame, train_df), cast(pd.DataFrame, val_df)
