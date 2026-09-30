"""Tests for background images -- crops that contain no ear.

Roughly 30% of the face crops the pipeline produces contain no visible ear.
Training without them leaves the model never having seen a face whose ears
are hidden, which is a third of what it is handed at inference. The dataloader
keys on annotation rows, so such an image needs a placeholder row that
contributes neither a box nor an ignore region.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from utils.config import IGNORE_ANNOTATION_SOURCE, NEGATIVE_ANNOTATION_SOURCE


@pytest.fixture
def dataset_root(tmp_path):
    """Three images: one with an ear, one background, one ignore-only."""
    for name in ('ear.jpg', 'background.jpg', 'ignored.jpg'):
        Image.fromarray(
            np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        ).save(tmp_path / name)
    rows = [
        {'image_path': 'ear.jpg', 'x1': 50, 'y1': 60, 'w': 30, 'h': 60,
         'earside': 'left', 'source': 'ds', 'annotation_source': 'GT',
         'confidence': 1.0},
        {'image_path': 'background.jpg', 'x1': 0, 'y1': 0, 'w': 0, 'h': 0,
         'earside': 'none', 'source': 'face_crop',
         'annotation_source': NEGATIVE_ANNOTATION_SOURCE, 'confidence': 0.0},
        {'image_path': 'ignored.jpg', 'x1': 20, 'y1': 20, 'w': 40, 'h': 40,
         'earside': 'left', 'source': 'ds',
         'annotation_source': IGNORE_ANNOTATION_SOURCE, 'confidence': 1.0},
    ]
    csv = tmp_path / 'data.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    return tmp_path, csv


def build(dataset_root, **kwargs):
    from dataloader import CSVDetectorDataset
    root, csv = dataset_root
    return CSVDetectorDataset(csv_path=str(csv), root_dir=str(root), **kwargs)


class TestDatasetMembership:
    def test_all_three_images_are_loaded(self, dataset_root):
        assert len(build(dataset_root)) == 3

    def test_the_placeholder_contributes_no_box(self, dataset_root):
        dataset = build(dataset_root)
        by_path = {s['image_path']: s for s in dataset.samples}
        assert len(by_path['background.jpg']['boxes']) == 0

    def test_the_placeholder_contributes_no_ignore_region(self, dataset_root):
        """
        This is what separates it from an IGNORE row. An ignore region would
        stop the model being scored on that ground; a background image is
        ground the model must be scored on, and stay silent over.
        """
        dataset = build(dataset_root)
        by_path = {s['image_path']: s for s in dataset.samples}
        assert len(by_path['background.jpg']['ignore_boxes']) == 0

    def test_an_ignore_row_still_becomes_an_ignore_region(self, dataset_root):
        dataset = build(dataset_root)
        by_path = {s['image_path']: s for s in dataset.samples}
        assert len(by_path['ignored.jpg']['ignore_boxes']) == 1
        assert len(by_path['ignored.jpg']['boxes']) == 0

    def test_a_real_box_is_untouched(self, dataset_root):
        dataset = build(dataset_root)
        by_path = {s['image_path']: s for s in dataset.samples}
        assert len(by_path['ear.jpg']['boxes']) == 1


class TestSampleContents:
    def test_a_background_sample_yields_an_image_and_no_boxes(self, dataset_root):
        dataset = build(dataset_root, augment=False)
        index = [s['image_path'] for s in dataset.samples].index('background.jpg')
        sample = dataset[index]
        assert sample['image'].shape[-1] == 128 and sample['image'].shape[-2] == 128
        assert len(sample['gt_boxes']) == 0

    def test_every_anchor_of_a_background_image_is_negative(self, dataset_root):
        dataset = build(dataset_root, augment=False)
        index = [s['image_path'] for s in dataset.samples].index('background.jpg')
        targets = dataset[index]['anchor_targets']
        # Column 0 is the objectness target.
        assert float(targets[:, 0].max()) == 0.0

    def test_no_anchor_of_a_background_image_is_ignored(self, dataset_root):
        dataset = build(dataset_root, augment=False)
        index = [s['image_path'] for s in dataset.samples].index('background.jpg')
        sample = dataset[index]
        if 'anchor_ignore' in sample:
            assert float(sample['anchor_ignore'].max()) == 0.0

    def test_augmentation_survives_a_background_image(self, dataset_root):
        """Augmentation unpacks box columns; zero boxes must not trip it."""
        dataset = build(dataset_root, augment=True)
        index = [s['image_path'] for s in dataset.samples].index('background.jpg')
        for _ in range(20):
            sample = dataset[index]
            assert torch.isfinite(sample['image']).all()


class TestBatching:
    def test_a_mixed_batch_collates(self, dataset_root):
        from dataloader import collate_detector_fn
        dataset = build(dataset_root, augment=False)
        batch = collate_detector_fn([dataset[i] for i in range(len(dataset))])
        assert batch['image'].shape[0] == 3
        assert int(batch['gt_box_counts'].sum()) == 1

    def test_an_all_background_batch_collates(self, dataset_root):
        from dataloader import collate_detector_fn
        dataset = build(dataset_root, augment=False)
        index = [s['image_path'] for s in dataset.samples].index('background.jpg')
        batch = collate_detector_fn([dataset[index], dataset[index]])
        assert int(batch['gt_box_counts'].sum()) == 0
