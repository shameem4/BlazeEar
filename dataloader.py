"""
CSV-based data loading utilities for BlazeFace detector training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import augmentation
from utils.data_utils import split_dataframe_by_images
from utils.anchor_utils import assign_anchor_targets, generate_anchors_from_priors
from utils.config import ANCHOR_IGNORE_IOU, ANCHOR_TOP_K, IGNORE_ANNOTATION_SOURCE


def collate_detector_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate fixed-size anchor targets."""
    batch_size = len(batch)
    gt_counts = torch.tensor([sample["gt_boxes"].shape[0] for sample in batch], dtype=torch.long)
    max_gt = int(gt_counts.max().item()) if batch_size > 0 else 0
    if max_gt == 0:
        gt_boxes = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
    else:
        gt_boxes = torch.zeros((batch_size, max_gt, 4), dtype=torch.float32)
        for idx, sample in enumerate(batch):
            count = sample["gt_boxes"].shape[0]
            if count > 0:
                gt_boxes[idx, :count] = sample["gt_boxes"]

    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        # -1 for sample producers that do not track an index (e.g. the NPY path).
        "sample_index": torch.tensor(
            [int(sample.get("sample_index", -1)) for sample in batch], dtype=torch.long
        ),
        "anchor_targets": torch.stack([sample["anchor_targets"] for sample in batch]),
        "anchor_ignore": torch.stack([
            sample.get("anchor_ignore", torch.zeros(sample["anchor_targets"].shape[0], dtype=torch.bool))
            for sample in batch
        ]),
        "gt_boxes": gt_boxes,
        "gt_box_counts": gt_counts
    }


# =============================================================================
# Dataset
# =============================================================================

class CSVDetectorDataset(Dataset):
    """
    Dataset for detector training from a CSV with columns:
    image_path, x1, y1, w, h
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target_size: Tuple[int, int] = (128, 128),
        augment: bool = True,
        max_samples: Optional[int] = None,
        augment_max_side: int = 256,
        anchor_top_k: int = ANCHOR_TOP_K,
        anchor_ignore_iou: float = ANCHOR_IGNORE_IOU
    ):
        # Avoid oversubscribing CPU threads in DataLoader workers.
        try:
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.augment = augment
        self.augment_max_side = int(augment_max_side)
        # One anchor tensor for the whole dataset; assignment is driven by it
        # rather than by hardcoded per-cell square sizes.
        self.anchors = generate_anchors_from_priors().numpy()
        self.anchor_top_k = int(anchor_top_k)
        self.anchor_ignore_iou = float(anchor_ignore_iou)

        df = pd.read_csv(self.csv_path)
        required_cols = {"image_path", "x1", "y1", "w", "h"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        grouped = df.groupby("image_path", sort=False)
        self.samples: List[Dict] = []
        missing_files = 0
        for image_path, group in grouped:
            image_path_str = str(image_path)
            full_path = self.root_dir / image_path_str
            if not full_path.exists():
                missing_files += 1
                continue
            if "annotation_source" in group.columns:
                is_ignore = group["annotation_source"] == IGNORE_ANNOTATION_SOURCE
            else:
                is_ignore = pd.Series(False, index=group.index)
            boxes = group.loc[~is_ignore, ["x1", "y1", "w", "h"]].values.astype(np.float32)
            ignore_boxes = group.loc[is_ignore, ["x1", "y1", "w", "h"]].values.astype(np.float32)
            self.samples.append({
                "image_path": image_path_str,
                "boxes": boxes,
                "ignore_boxes": ignore_boxes,
            })

        if max_samples:
            self.samples = self.samples[:max_samples]

        total_boxes = sum(len(sample["boxes"]) for sample in self.samples)
        print(f"Loaded {len(self.samples)} images ({total_boxes} boxes) from {self.csv_path}")
        if missing_files:
            print(f"Skipped {missing_files} entries with missing files in {self.csv_path}")

        self._missing_logged: Set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        full_path = self.root_dir / image_path
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        image = cv2.imread(str(full_path))
        if image is None:
            raise RuntimeError(f"Could not read image data: {full_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _downscale_for_augment(self, image: np.ndarray) -> np.ndarray:
        """Cap the working resolution before augmenting.

        Geometric augmentation has to happen before the letterbox resize, but
        warping at full source resolution is wasted work when the model input is
        128px: sources here are 640x640, which is 25x the pixels of the target.
        Capping at `augment_max_side` (2x the model input) keeps every detail
        that survives the final resize.

        It also matches inference more closely. BlazeDetector.resize_pad goes
        source -> 256 -> 128, so the model sees two-step resampling at inference
        but saw a single 640 -> 128 step in training.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.augment_max_side:
            return image
        scale = self.augment_max_side / longest
        return cv2.resize(
            image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA
        )

    def _augment_image(self, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, bboxes

        # Boxes are normalized, so downscaling leaves them unchanged.
        image = self._downscale_for_augment(image)

        # Color augmentations
        if np.random.random() > 0.5:
            image = augmentation.augment_saturation(image)
        if np.random.random() > 0.5:
            image = augmentation.augment_brightness(image)
        if np.random.random() > 0.5:
            image = augmentation.augment_color_jitter(image)
        if np.random.random() > 0.5:
            image = augmentation.augment_photometric_jitter(image)
        
        # Geometric augmentations
        if np.random.random() > 0.5 and len(bboxes) > 0:
            image, bboxes = augmentation.augment_horizontal_flip(image, bboxes)
        if np.random.random() > 0.5 and len(bboxes) > 0:
            image, bboxes = augmentation.augment_scale(image, bboxes, scale_range=(0.75, 1.25))
        if np.random.random() > 0.5 and len(bboxes) > 0:
            image, bboxes = augmentation.augment_rotation(image, bboxes, angle_range=(-20, 20))
        
        # Occlusion augmentations (less frequent). Each is passed the labelled
        # boxes so it can refuse to hide an ear whose positive label it would
        # leave behind; sizes are fractions of the image, since these now run
        # before the letterbox at the augmentation working resolution rather
        # than on the finished 128x128 tensor.
        if np.random.random() > 0.6 and len(bboxes) > 0:
            image = augmentation.augment_face_cutout(image, bboxes)
        if np.random.random() > 0.6 and len(bboxes) > 0:
            image = augmentation.augment_targeted_ear_occlusion(image, bboxes)
        if np.random.random() > 0.7:
            image = augmentation.augment_synthetic_occlusion(image, bboxes, num_occlusions=1)
        if np.random.random() > 0.7:
            image = augmentation.augment_cutout(image, bboxes, num_holes=1)

        return image, bboxes

    def _resize_and_pad(self, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target_h, target_w = self.target_size
        orig_h, orig_w = image.shape[:2]

        if orig_h >= orig_w:
            scale = target_h / orig_h
            new_h, new_w = target_h, int(round(orig_w * scale))
        else:
            scale = target_w / orig_w
            new_w, new_h = target_w, int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h))
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        if len(bboxes) > 0:
            abs_boxes = bboxes.copy()
            abs_boxes[:, [0, 2]] *= orig_h
            abs_boxes[:, [1, 3]] *= orig_w

            # Only the coordinate columns. Scaling the whole array also scales
            # any trailing per-box metadata, which silently turned an ignore
            # flag of 1.0 into 0.2 and reclassified the box as ground truth.
            abs_boxes[:, :4] *= scale
            abs_boxes[:, [0, 2]] += pad_top
            abs_boxes[:, [1, 3]] += pad_left

            abs_boxes[:, [0, 2]] = np.clip(abs_boxes[:, [0, 2]], 0, target_h - 1)
            abs_boxes[:, [1, 3]] = np.clip(abs_boxes[:, [1, 3]], 0, target_w - 1)

            abs_boxes[:, [0, 2]] /= target_h
            abs_boxes[:, [1, 3]] /= target_w
            bboxes = abs_boxes

        return padded, bboxes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        total_samples = len(self.samples)
        if total_samples == 0:
            raise IndexError("Dataset is empty")

        idx = idx % total_samples
        attempts = 0

        while attempts < total_samples:
            sample = self.samples[idx]
            try:
                image = self._load_image(sample["image_path"])
            except (FileNotFoundError, RuntimeError) as exc:
                rel_path = sample["image_path"]
                if rel_path not in self._missing_logged:
                    print(f"Warning: {exc}; skipping sample {rel_path}")
                    self._missing_logged.add(rel_path)
                idx = (idx + 1) % total_samples
                attempts += 1
                continue

            orig_h, orig_w = image.shape[:2]

            def _normalize(boxes_px: np.ndarray) -> np.ndarray:
                if len(boxes_px) == 0:
                    return np.zeros((0, 4), dtype=np.float32)
                x1, y1, w, h = (boxes_px[:, 0], boxes_px[:, 1],
                                boxes_px[:, 2], boxes_px[:, 3])
                return np.stack([
                    np.clip(y1 / orig_h, 0, 1), np.clip(x1 / orig_w, 0, 1),
                    np.clip((y1 + h) / orig_h, 0, 1), np.clip((x1 + w) / orig_w, 0, 1),
                ], axis=1).astype(np.float32)

            gt = _normalize(sample["boxes"])
            ignore_regions = _normalize(sample.get("ignore_boxes", np.zeros((0, 4))))

            # Carry an ignore flag as a fifth column so the geometric
            # augmentations, which drop and reorder boxes, cannot desync it
            # from the boxes it describes.
            flags = np.concatenate([
                np.zeros((len(gt), 1), dtype=np.float32),
                np.ones((len(ignore_regions), 1), dtype=np.float32),
            ])
            bboxes = np.concatenate([
                np.concatenate([gt, ignore_regions], axis=0), flags
            ], axis=1) if len(flags) else np.zeros((0, 5), dtype=np.float32)

            # Augment at native resolution, then letterbox once. Doing it the
            # other way round scaled and rotated an image that had already been
            # padded to 128x128, so augment_scale added a second letterbox and
            # augment_rotation turned the padding bars into diagonal wedges that
            # never occur at inference -- all at a resolution where an ear is
            # roughly 10x15 px.
            image, bboxes = self._augment_image(image, bboxes)
            image, bboxes = self._resize_and_pad(image, bboxes)

            is_ignore = bboxes[:, 4] > 0.5 if len(bboxes) else np.zeros(0, dtype=bool)
            gt_boxes = bboxes[~is_ignore, :4]
            ignore_boxes = bboxes[is_ignore, :4]

            anchor_targets, anchor_ignore = assign_anchor_targets(
                gt_boxes,
                self.anchors,
                top_k=self.anchor_top_k,
                ignore_iou=self.anchor_ignore_iou,
                ignore_boxes=ignore_boxes,
            )

            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            return {
                "image": image,
                # Index into self.samples, so callers can recover the source
                # image and its per-box annotation provenance. With augment=False
                # the returned gt_boxes stay in CSV row order, one-to-one with
                # self.samples[idx]["boxes"].
                "sample_index": idx,
                "anchor_targets": torch.from_numpy(anchor_targets).float(),
                "anchor_ignore": torch.from_numpy(anchor_ignore),
                "gt_boxes": torch.from_numpy(np.ascontiguousarray(gt_boxes)).float()
            }

        raise RuntimeError("No readable images remain in dataset.")

    def get_sample_annotations(self, idx: int) -> Tuple[str, np.ndarray]:
        """Return original image path and (x1, y1, w, h) boxes."""
        sample = self.samples[idx]
        return sample["image_path"], sample["boxes"].copy()


# =============================================================================
# Helpers
# =============================================================================

def create_dataloader(
    csv_path: str,
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (128, 128),
    augment: bool = True,
    max_samples: Optional[int] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    dataset = CSVDetectorDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        target_size=target_size,
        augment=augment,
        max_samples=max_samples
    )
    effective_prefetch: Optional[int]
    if num_workers > 0:
        effective_prefetch = max(1, int(prefetch_factor))
    else:
        effective_prefetch = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_detector_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=effective_prefetch
    )



def create_train_val_split(
    csv_path: str,
    output_dir: str,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[Path, Path]:
    """Split CSV into train/val files grouped by image."""
    df = pd.read_csv(csv_path)
    train_df, val_df = split_dataframe_by_images(
        df, image_column="image_path", val_fraction=val_split, random_seed=random_seed
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_path = output_dir_path / "train.csv"
    val_path = output_dir_path / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    total_images = len(df["image_path"].drop_duplicates())
    print("Created train/val split:")
    print(
        f"  Train: {len(train_df)} rows "
        f"({len(train_df['image_path'].unique())}/{total_images} images) -> {train_path}"
    )
    print(
        f"  Val:   {len(val_df)} rows "
        f"({len(val_df['image_path'].unique())}/{total_images} images) -> {val_path}"
    )

    return train_path, val_path
