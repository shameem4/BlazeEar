"""Tests for train/val splitting."""
import numpy as np
import pandas as pd
import pytest

from utils.data_utils import base_source, split_dataframe_by_images


def make_df(n_per_source=(20, 20), sources=('A', 'B'), boxes_per_image=2):
    rows = []
    for source, n in zip(sources, n_per_source):
        for i in range(n):
            for b in range(boxes_per_image):
                rows.append({
                    'image_path': f'{source}/img{i:03d}.jpg',
                    'x1': 10 + b, 'y1': 10, 'w': 20, 'h': 40,
                    'source': source,
                })
    return pd.DataFrame(rows)


class TestBaseSource:
    def test_strips_pseudo_label_suffixes(self):
        assert base_source('Human Ear.v3i.coco_pose_aug') == 'Human Ear.v3i.coco'
        assert base_source('ear.v4i.coco_yolo11_aug') == 'ear.v4i.coco'

    def test_leaves_plain_sources_alone(self):
        assert base_source('tf_records.v1i.coco') == 'tf_records.v1i.coco'


class TestGrouping:
    def test_no_image_appears_in_both_halves(self):
        train, val = split_dataframe_by_images(make_df(), val_fraction=0.25)
        assert set(train.image_path) & set(val.image_path) == set()

    def test_all_rows_are_preserved(self):
        df = make_df()
        train, val = split_dataframe_by_images(df, val_fraction=0.25)
        assert len(train) + len(val) == len(df)

    def test_every_box_of_an_image_moves_together(self):
        df = make_df(boxes_per_image=3)
        train, val = split_dataframe_by_images(df, val_fraction=0.25)
        for part in (train, val):
            counts = part.groupby('image_path').size()
            assert (counts == 3).all()

    def test_group_keys_keep_duplicates_together(self):
        """Near-duplicate images must not straddle the split."""
        df = make_df(n_per_source=(10,), sources=('A',))
        images = df.image_path.unique()
        # Declare every image a member of one of two duplicate clusters.
        groups = {img: f'cluster{i % 2}' for i, img in enumerate(images)}
        train, val = split_dataframe_by_images(
            df, val_fraction=0.5, group_keys=groups)
        for part in (train, val):
            clusters = {groups[p] for p in part.image_path.unique()}
            assert len(clusters) <= 1, 'a duplicate cluster was split'

    def test_raises_on_missing_column(self):
        with pytest.raises(ValueError):
            split_dataframe_by_images(pd.DataFrame({'a': [1]}), image_column='nope')

    def test_raises_on_empty_frame(self):
        with pytest.raises(ValueError):
            split_dataframe_by_images(pd.DataFrame({'image_path': []}))


class TestStratification:
    def test_each_source_is_represented_proportionally(self):
        df = make_df(n_per_source=(100, 100, 100), sources=('A', 'B', 'C'))
        _, val = split_dataframe_by_images(
            df, val_fraction=0.2, stratify_column='source')
        per_source = val.drop_duplicates('image_path').source.value_counts()
        assert set(per_source.index) == {'A', 'B', 'C'}
        assert per_source.min() >= 18 and per_source.max() <= 22

    def test_a_small_source_still_reaches_validation(self):
        """
        Without stratification a source of 6 images can land entirely in train
        at a 15% split, and nothing reports that it is unmeasured.
        """
        df = make_df(n_per_source=(400, 6), sources=('big', 'tiny'))
        _, val = split_dataframe_by_images(
            df, val_fraction=0.15, stratify_column='source')
        assert 'tiny' in set(val.source)

    def test_stratification_folds_pseudo_label_suffixes(self):
        df = make_df(n_per_source=(50, 50), sources=('ds.coco', 'ds.coco_pose_aug'))
        # Both are the same dataset, so they form one stratum, not two.
        _, val = split_dataframe_by_images(
            df, val_fraction=0.2, stratify_column='source')
        assert len(val) > 0

    def test_unknown_stratify_column_is_ignored(self):
        train, val = split_dataframe_by_images(
            make_df(), val_fraction=0.25, stratify_column='not_a_column')
        assert len(train) and len(val)


class TestRowOrder:
    def test_rows_are_shuffled_by_default(self):
        """
        Preserving input order made any prefix of val.csv a single source,
        which is how an evaluation slice of the first 192 images sampled one
        dataset and produced a metric that moved with the slice size.
        """
        df = make_df(n_per_source=(150, 150), sources=('A', 'B'))
        _, val = split_dataframe_by_images(
            df, val_fraction=0.5, stratify_column='source')
        prefix = val.drop_duplicates('image_path').head(40).source
        assert prefix.nunique() == 2, 'prefix of val is a single source'

    def test_shuffle_can_be_disabled(self):
        df = make_df()
        _, val = split_dataframe_by_images(df, val_fraction=0.25, shuffle_rows=False)
        assert list(val.image_path) == sorted(val.image_path, key=list(df.image_path).index)


class TestDeterminism:
    def test_same_seed_gives_the_same_split(self):
        df = make_df()
        a, _ = split_dataframe_by_images(df, val_fraction=0.25, random_seed=7)
        b, _ = split_dataframe_by_images(df, val_fraction=0.25, random_seed=7)
        assert list(a.image_path) == list(b.image_path)

    def test_different_seeds_differ(self):
        df = make_df(n_per_source=(200,), sources=('A',))
        _, a = split_dataframe_by_images(df, val_fraction=0.25, random_seed=1)
        _, b = split_dataframe_by_images(df, val_fraction=0.25, random_seed=2)
        assert set(a.image_path) != set(b.image_path)


class TestSourcePhotoGrouping:
    """
    Roboflow re-exports one photo under several hashes with photometric
    alterations. Splitting per file puts the same scene on both sides.
    """

    def test_recognises_roboflow_exports(self):
        from utils.data_utils import source_photo_key
        a = 'ear annotations..v3i.coco/train/1-590-_jpg.rf.7212e68de243cb47dfe24cc.jpg'
        b = 'ear annotations..v3i.coco/train/1-590-_jpg.rf.ea4a51185388213f10bf372.jpg'
        assert source_photo_key(a) == source_photo_key(b)

    def test_different_photos_stay_distinct(self):
        from utils.data_utils import source_photo_key
        a = 'ds/train/1-590-_jpg.rf.7212e68de243cb47dfe24cc.jpg'
        b = 'ds/train/1-327-_jpg.rf.7212e68de243cb47dfe24cc.jpg'
        assert source_photo_key(a) != source_photo_key(b)

    def test_non_roboflow_names_fall_back_to_the_path(self):
        from utils.data_utils import source_photo_key
        assert source_photo_key('a/b/plain.jpg') == 'a/b/plain.jpg'

    def test_copies_never_straddle_the_split(self):
        rows = []
        for i in range(40):
            for copy in range(2):
                rows.append({
                    'image_path': f'ds/train/img{i:03d}_jpg.rf.{copy:032x}.jpg',
                    'x1': 10, 'y1': 10, 'w': 20, 'h': 40, 'source': 'ds',
                })
        df = pd.DataFrame(rows)
        train, val = split_dataframe_by_images(df, val_fraction=0.5)

        from utils.data_utils import source_photo_key
        train_keys = {source_photo_key(p) for p in train.image_path}
        val_keys = {source_photo_key(p) for p in val.image_path}
        assert train_keys & val_keys == set()

    def test_per_file_splitting_is_still_available(self):
        rows = [{'image_path': f'ds/train/img{i:03d}_jpg.rf.{c:032x}.jpg',
                 'x1': 10, 'y1': 10, 'w': 20, 'h': 40, 'source': 'ds'}
                for i in range(40) for c in range(2)]
        train, val = split_dataframe_by_images(
            pd.DataFrame(rows), val_fraction=0.5, group_keys={})
        assert len(train) and len(val)
