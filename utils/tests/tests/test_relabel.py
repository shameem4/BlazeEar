"""Tests for the relabelling queue logic.

`review` is interactive and untested here; `propose` and `apply` are not, and a
bug in either silently corrupts the training set.
"""
import numpy as np
import pandas as pd
import pytest

from relabel import (
    MATCH_IOU,
    categorise,
    cmd_apply,
    human_boxes_by_image,
    iou_matrix,
)


def xyxy(x1, y1, x2, y2):
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


class TestIoUMatrix:
    def test_identical(self):
        b = xyxy(0, 0, 10, 10)
        assert abs(iou_matrix(b, b)[0, 0] - 1.0) < 1e-6

    def test_disjoint(self):
        assert iou_matrix(xyxy(0, 0, 10, 10), xyxy(50, 50, 60, 60))[0, 0] == 0.0

    def test_empty_sides(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        assert iou_matrix(empty, xyxy(0, 0, 1, 1)).shape == (0, 1)
        assert iou_matrix(xyxy(0, 0, 1, 1), empty).shape == (1, 0)


class TestCategorise:
    def test_agreeing_proposal_is_not_queued(self):
        human = xyxy(10, 10, 30, 50)
        proposals, missed = categorise(human.copy(), np.array([0.9]), human)
        assert proposals == []
        assert missed == []

    def test_disjoint_proposal_is_a_new_ear(self):
        human = xyxy(10, 10, 30, 50)
        proposal = xyxy(200, 10, 220, 50)
        rows, missed = categorise(proposal, np.array([0.8]), human)
        assert len(rows) == 1 and rows[0]['category'] == 'new'
        # The human box has no matching proposal, so it is also flagged.
        assert len(missed) == 1 and missed[0]['category'] == 'missed'

    def test_partial_overlap_is_a_conflict(self):
        human = xyxy(10, 10, 30, 50)
        proposal = xyxy(22, 10, 42, 50)          # IoU above 0.1, below MATCH_IOU
        rows, _ = categorise(proposal, np.array([0.8]), human)
        assert len(rows) == 1 and rows[0]['category'] == 'conflict'
        assert 0.1 < rows[0]['best_iou_with_human'] < MATCH_IOU

    def test_second_ear_found_on_a_single_label_image(self):
        """The case this pipeline exists for."""
        human = xyxy(40, 60, 60, 100)
        proposals = np.array([[40, 60, 60, 100], [180, 60, 200, 100]], dtype=np.float32)
        rows, missed = categorise(proposals, np.array([0.95, 0.88]), human)
        assert [r['category'] for r in rows] == ['new']
        assert missed == []

    def test_box_geometry_is_converted_to_xywh(self):
        rows, _ = categorise(xyxy(200, 10, 220, 50), np.array([0.8]),
                             np.zeros((0, 4), dtype=np.float32))
        assert rows[0]['x1'] == 200 and rows[0]['y1'] == 10
        assert rows[0]['w'] == 20 and rows[0]['h'] == 40

    def test_image_with_no_human_boxes(self):
        rows, missed = categorise(xyxy(1, 1, 5, 5), np.array([0.7]),
                                  np.zeros((0, 4), dtype=np.float32))
        assert len(rows) == 1 and rows[0]['category'] == 'new'
        assert missed == []

    def test_no_proposals_flags_every_human_box(self):
        human = np.array([[10, 10, 30, 50], [80, 10, 100, 50]], dtype=np.float32)
        rows, missed = categorise(np.zeros((0, 4), dtype=np.float32),
                                  np.zeros((0,)), human)
        assert rows == []
        assert len(missed) == 2


class TestHumanBoxes:
    def test_pseudo_labels_are_excluded(self, tmp_path):
        csv = tmp_path / 'm.csv'
        pd.DataFrame({
            'image_path': ['a.jpg', 'a.jpg', 'b.jpg'],
            'x1': [1, 2, 3], 'y1': [1, 2, 3], 'w': [4, 4, 4], 'h': [8, 8, 8],
            'annotation_source': ['GT', 'POSE', 'GT+EAR'],
        }).to_csv(csv, index=False)
        boxes = human_boxes_by_image(str(csv))
        assert len(boxes['a.jpg']) == 1, 'POSE must not count as human'
        assert len(boxes['b.jpg']) == 1

    def test_converts_to_xyxy(self, tmp_path):
        csv = tmp_path / 'm.csv'
        pd.DataFrame({
            'image_path': ['a.jpg'], 'x1': [10], 'y1': [20], 'w': [30], 'h': [40],
            'annotation_source': ['GT'],
        }).to_csv(csv, index=False)
        assert np.allclose(human_boxes_by_image(str(csv))['a.jpg'][0], [10, 20, 40, 60])


class TestApply:
    def _setup(self, tmp_path, reviews):
        master = tmp_path / 'master.csv'
        pd.DataFrame({
            'image_path': ['a.jpg', 'a.jpg'],
            'x1': [10, 99], 'y1': [10, 99], 'w': [20, 20], 'h': [40, 40],
            'annotation_source': ['GT', 'POSE'],
            'source': ['ds', 'ds'], 'confidence': [1.0, 1.0],
        }).to_csv(master, index=False)

        queue = tmp_path / 'q.csv'
        pd.DataFrame({
            'image_path': ['a.jpg'] * len(reviews),
            'x1': [200] * len(reviews), 'y1': [10] * len(reviews),
            'w': [20] * len(reviews), 'h': [40] * len(reviews),
            'confidence': [0.9] * len(reviews),
            'best_iou_with_human': [0.0] * len(reviews),
            'category': ['new'] * len(reviews),
            'review': reviews,
        }).to_csv(queue, index=False)
        return master, queue, tmp_path / 'out.csv'

    def _run(self, master, queue, out, drop_pose=True):
        import argparse
        cmd_apply(argparse.Namespace(queue=str(queue), csv=str(master),
                                     output=str(out), drop_pose=drop_pose))
        return pd.read_csv(out)

    def test_accepted_proposals_are_added(self, tmp_path):
        master, queue, out = self._setup(tmp_path, ['accepted'])
        result = self._run(master, queue, out)
        added = result[result.annotation_source == 'GT+REVIEW']
        assert len(added) == 1 and added.iloc[0].x1 == 200

    def test_rejected_and_pending_are_not_added(self, tmp_path):
        master, queue, out = self._setup(tmp_path, ['rejected', 'pending', 'skipped'])
        result = self._run(master, queue, out)
        assert (result.annotation_source == 'GT+REVIEW').sum() == 0

    def test_pose_rows_are_dropped_by_default(self, tmp_path):
        master, queue, out = self._setup(tmp_path, ['accepted'])
        result = self._run(master, queue, out)
        assert (result.annotation_source == 'POSE').sum() == 0
        assert (result.annotation_source == 'GT').sum() == 1

    def test_pose_rows_can_be_kept(self, tmp_path):
        master, queue, out = self._setup(tmp_path, ['accepted'])
        result = self._run(master, queue, out, drop_pose=False)
        assert (result.annotation_source == 'POSE').sum() == 1

    def test_accepted_missed_labels_are_not_added_as_boxes(self, tmp_path):
        """
        Accepting a 'missed' item means "the existing label is wrong", not
        "add this box". It must never be appended to the dataset.
        """
        master, queue, out = self._setup(tmp_path, ['accepted'])
        q = pd.read_csv(queue)
        q.loc[0, 'category'] = 'missed'
        q.to_csv(queue, index=False)
        result = self._run(master, queue, out)
        assert (result.annotation_source == 'GT+REVIEW').sum() == 0

    def test_box_wrong_is_never_added(self, tmp_path):
        """
        'box_wrong' means a real ear with a bad extent. Adding it would put a
        badly-localized positive into training, which corrupts box regression
        more than the missing label costs in recall.
        """
        master, queue, out = self._setup(tmp_path, ['box_wrong'])
        result = self._run(master, queue, out)
        assert (result.annotation_source == 'GT+REVIEW').sum() == 0

    def test_box_wrong_is_queued_for_correction(self, tmp_path):
        master, queue, out = self._setup(tmp_path, ['box_wrong'])
        self._run(master, queue, out)
        fix = out.with_name(out.stem + '_needs_box_fix.csv')
        assert fix.exists()
        assert len(pd.read_csv(fix)) == 1
