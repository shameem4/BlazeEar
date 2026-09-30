"""
Utilities for loading annotation CSVs and creating train/val splits.
"""

from __future__ import annotations

import re
from collections import defaultdict
import warnings
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, cast

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


SOURCE_PHOTO_RE = re.compile(r"(.+?)_(jpg|jpeg|png)\.rf\.[0-9a-f]+\.", re.IGNORECASE)


def source_photo_key(image_path: str) -> str:
    """Identify the original photo behind a Roboflow export filename.

    Roboflow re-exports the same source photo several times under different
    hashes, with photometric alterations applied: `1-590-_jpg.rf.7212e68d….jpg`
    and `1-590-_jpg.rf.ea4a5118….jpg` are the same scene. In this dataset 231
    of 13482 files are such copies, covering 220 source photos.

    They matter twice over. A per-file train/val split puts 53 source photos on
    both sides, which is leakage nothing else detects; and a review queue shows
    the same ear several times, which wastes the one resource relabelling is
    actually limited by.

    Falls back to the full path when the filename is not a Roboflow export.
    """
    path = Path(image_path)
    match = SOURCE_PHOTO_RE.match(path.name)
    if not match:
        return str(image_path)
    return f"{path.parent.parent.name}/{match.group(1)}"


def base_source(source: str) -> str:
    """Strip the pseudo-label suffixes to get the originating dataset.

    `source` carries both the dataset and how a row was annotated, so
    "Human Ear.v3i.coco" and "Human Ear.v3i.coco_pose_aug" are the same images
    with extra rows, not different data. Stratifying on the raw column would
    treat them as separate strata.
    """
    for suffix in ("_pose_aug", "_yolo11_aug"):
        source = source.replace(suffix, "")
    return source


def split_dataframe_by_images(
    df: pd.DataFrame,
    image_column: str = "image_path",
    val_fraction: float = 0.2,
    random_seed: int = 42,
    stratify_column: str | None = None,
    group_keys: Dict[str, str] | None = None,
    shuffle_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an annotation DataFrame into train/val, grouped so no image straddles.

    Args:
        df: DataFrame containing at least `image_column`
        image_column: Column identifying unique images
        val_fraction: Fraction of image groups to allocate to validation
        random_seed: RNG seed
        stratify_column: Split each value of this column separately, so every
            source is represented in both halves in proportion. Without it, a
            small source can land almost entirely on one side. Pass "source"
            and it is reduced with `base_source` first.
        group_keys: image -> group id. Images sharing a group id always land on
            the same side. Defaults to the content-hash photo index when one has
            been built, falling back to `source_photo_key`. Filenames alone miss
            most of it: the same scene was ingested under different sequence
            numbers, so content hashing finds 559 duplicate files where the
            filename key finds 231. Pass an empty dict to split per file.
        shuffle_rows: Shuffle output row order. The previous implementation
            preserved the input order, which is source-by-source, so any prefix
            of val.csv was a single dataset -- that is how a 192-image
            evaluation slice ended up sampling one source.

    Returns:
        (train_df, val_df) with indices reset

    Note: duplicate grouping is not subject grouping. Subject identity is not
    recoverable here. Every source is a Roboflow
    export with hashed filenames, so the same person across several photos
    cannot be detected, and train/val may share subjects.
    """
    if image_column not in df.columns:
        raise ValueError(f"Column '{image_column}' not found in DataFrame")

    image_ids = df[image_column].drop_duplicates().tolist()
    if not image_ids:
        raise ValueError("No images found to split.")

    if group_keys is None:
        from utils.photo_index import load_index
        try:
            index = load_index()
        except Exception as exc:
            index = {}
            warnings.warn(
                f'Could not read the photo index ({exc}). Falling back to '
                'filename-derived photo keys, which are weaker: they missed '
                '121 near-duplicate photos that then straddled train and val. '
                'Run utils/photo_index.py to rebuild it.',
                RuntimeWarning, stacklevel=2)
        else:
            if not index:
                warnings.warn(
                    'The photo index is empty or missing, so grouping falls '
                    'back to filename-derived keys. Those missed 121 '
                    'near-duplicate photos that then straddled train and val. '
                    'Run utils/photo_index.py to build it.',
                    RuntimeWarning, stacklevel=2)
        group_keys = {img: (index.get(img) or source_photo_key(img)) for img in image_ids}
    group_of = {img: group_keys.get(img, img) for img in image_ids}

    strata: DefaultDict[str, list] = defaultdict(list)
    if stratify_column and stratify_column in df.columns:
        values = df.groupby(image_column)[stratify_column].first()
        if stratify_column == "source":
            values = values.map(base_source)
        stratum_of_group: Dict[str, str] = {}
        for img in image_ids:
            stratum_of_group.setdefault(group_of[img], str(values.get(img, "")))
        for group, stratum in stratum_of_group.items():
            strata[stratum].append(group)
    else:
        strata[""] = list(dict.fromkeys(group_of[img] for img in image_ids))

    rng = np.random.default_rng(random_seed)
    val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
    val_groups: set = set()

    for stratum in sorted(strata):
        groups = sorted(strata[stratum])
        rng.shuffle(groups)
        n_val = int(round(len(groups) * val_fraction))
        if len(groups) > 1:
            n_val = max(1, min(n_val, len(groups) - 1))
        val_groups.update(groups[:n_val])

    val_mask = df[image_column].map(lambda img: group_of.get(img, img) in val_groups)
    val_df = df[val_mask].copy()
    train_df = df[~val_mask].copy()

    if shuffle_rows:
        val_df = val_df.sample(frac=1.0, random_state=random_seed)
        train_df = train_df.sample(frac=1.0, random_state=random_seed)

    return (
        cast(pd.DataFrame, train_df.reset_index(drop=True)),
        cast(pd.DataFrame, val_df.reset_index(drop=True)),
    )
