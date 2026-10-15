#!/usr/bin/env python3
"""Where does the anchor budget go, and which priors carry the small ears?

The second stage localises an ear and then fails the score gate on it, and no
threshold recovers that -- the ranking inside each size band is already good
while the base rate is not. This measures the training-side cause.

    python measure_anchor_budget.py

Two things come out of it. First, positive supervision is NOT starved by size:
`ANCHOR_TOP_K = 3` makes every box claim its three best anchors regardless of
prior fit, so the mean is exactly 3.00 positives per ear in every band. That
is the design working, and it rules out the intuitive explanation.

Second, the budget is spent where the ears are not. `HARD_NEGATIVE_RATIO`
mines negatives GLOBALLY across all anchors, so a prior's negative exposure
scales with its anchor count while its positive exposure scales with how often
ears land on it. The two 32x32 levels hold 57% of the slots and receive under
8% of the positives -- and they are the levels a small ear has to land on.

Two gradients that run the same way come out of the band table: positive match
quality (median IoU of the assigned anchors) and ignore-band protection (how
many near-miss anchors are excused rather than scored as background).

Two conventions this script has to get right, both of which produce a
confident wrong answer when got wrong:

- `assign_anchor_targets` wants `[ymin, xmin, ymax, xmax]` NORMALISED, not
  `[x1, y1, w, h]`. Passing the latter yields ~0.03 positives per ear and an
  entirely artefactual story about starved supervision.
- `anchors_to_corners` emits `[ymin, xmin, ymax, xmax]`, not xyxy. Reading it
  as xyxy gives a median positive IoU of 0.000. The repo's own `_iou_matrix`
  is used here so the convention cannot drift.
"""
import argparse
import collections
import csv
import os
import pathlib

import numpy as np

from utils.anchor_utils import (_iou_matrix, anchors_to_corners,
                                assign_anchor_targets, get_anchors)
from utils.config import (ANCHOR_TOP_K, DEFAULT_INPUT_SIZE,
                          HARD_NEGATIVE_RATIO, HUMAN_ANNOTATION_SOURCES)

# Ear size in pixels within the model's input, as the band boundaries.
BANDS = ((0, 32), (32, 48), (48, 64), (64, 80), (80, 96), (96, 120),
         (120, None))


def band_of(pixels):
    for low, high in BANDS:
        if high is None:
            return f'{low}+'
        if low <= pixels < high:
            return f'{low}-{high}'
    return None


def band_names():
    return [f'{low}+' if high is None else f'{low}-{high}'
            for low, high in BANDS]


def distinct_priors(anchors):
    """Anchor indices grouped by their (w, h) prior, smallest area first."""
    priors = {}
    for index, (width, height) in enumerate(anchors[:, 2:]):
        key = (round(float(width), 4), round(float(height), 4))
        priors.setdefault(key, []).append(index)
    return {key: priors[key]
            for key in sorted(priors, key=lambda wh: wh[0] * wh[1])}


def human_crop_rows(csv_path):
    return [row for row in csv.DictReader(open(csv_path))
            if row['annotation_source'] in HUMAN_ANNOTATION_SOURCES
            and float(row['h']) > 0 and float(row['crop_side']) > 0]


def measure(args):
    anchors = get_anchors(input_size=args.input_size).cpu().numpy()
    size = args.input_size
    stats = {
        'ears': collections.Counter(),
        'positives': collections.defaultdict(list),
        'positive_iou': collections.defaultdict(list),
        'ignored': collections.defaultdict(list),
        'prior_positives': collections.Counter(),
        'prior_by_band': collections.defaultdict(collections.Counter),
    }

    for row in human_crop_rows(args.csv):
        side = float(row['crop_side'])
        x, y, w, h = (float(row[k]) for k in ('x1', 'y1', 'w', 'h'))
        band = band_of(max(w, h) / side * size)
        if band is None:
            continue
        box = np.array([[y / side, x / side, (y + h) / side, (x + w) / side]],
                       dtype=np.float32)
        targets, ignored = assign_anchor_targets(box, anchors,
                                                 top_k=args.top_k)
        matched = np.flatnonzero(targets[:, 0] > 0.5)

        stats['ears'][band] += 1
        stats['positives'][band].append(len(matched))
        stats['ignored'][band].append(int(ignored.sum()))
        if len(matched):
            ious = _iou_matrix(box, anchors_to_corners(anchors[matched]))[0]
            stats['positive_iou'][band].extend(ious.tolist())
        for index in matched:
            key = (round(float(anchors[index, 2]), 4),
                   round(float(anchors[index, 3]), 4))
            stats['prior_positives'][key] += 1
            stats['prior_by_band'][band][key] += 1

    return anchors, stats


def report(anchors, stats, size):
    priors = distinct_priors(anchors)
    print(f'{len(anchors)} anchors at {size}px, {len(priors)} distinct priors')
    for (width, height), indices in priors.items():
        print(f'   {width * size:6.1f} x {height * size:6.1f} px   '
              f'n={len(indices):5d}')

    total_ears = sum(stats['ears'].values())
    print(f'\n{total_ears} human training ears assigned\n')
    print(f"{'band':>8s} {'ears':>6s} {'% of data':>10s} "
          f"{'mean +anchors':>14s} {'median +IoU':>12s} {'mean ignored':>13s}")
    for band in band_names():
        if not stats['ears'][band]:
            continue
        ious = stats['positive_iou'][band]
        print(f"{band:>8s} {stats['ears'][band]:6d} "
              f"{100 * stats['ears'][band] / total_ears:9.1f}% "
              f"{np.mean(stats['positives'][band]):14.2f} "
              f"{np.median(ious) if ious else float('nan'):12.3f} "
              f"{np.mean(stats['ignored'][band]):13.2f}")

    print('\npositive assignments per prior, and the mining pressure against '
          f'it.\nHARD_NEGATIVE_RATIO={HARD_NEGATIVE_RATIO} is mined GLOBALLY '
          f'across all {len(anchors)} anchors,\nso a prior\'s negative '
          "exposure scales with its anchor COUNT while its\npositive exposure "
          'scales with how often ears actually land on it:')
    print(f"\n{'prior (px)':>18s} {'anchors':>8s} {'positives':>10s} "
          f"{'pos per anchor':>15s} {'share of pos':>13s}")
    total_positives = max(sum(stats['prior_positives'].values()), 1)
    for (width, height), indices in priors.items():
        count = stats['prior_positives'][(width, height)]
        print(f'{width * size:7.1f} x {height * size:6.1f} {len(indices):8d} '
              f'{count:10d} {count / len(indices):15.2f} '
              f'{100 * count / total_positives:12.1f}%')

    print('\nwhich prior carries each band (top prior, share):')
    for band in band_names():
        counts = stats['prior_by_band'][band]
        if not counts:
            continue
        (width, height), hits = counts.most_common(1)[0]
        print(f'  {band:>8s} -> {width * size:6.1f} x {height * size:6.1f} px  '
              f'({100 * hits / sum(counts.values()):.0f}% of its positives)')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--csv', default='data/splits/crops_train.csv',
                        help='face-crop split whose human ear boxes to assign')
    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument('--top-k', type=int, default=ANCHOR_TOP_K,
                        help='anchors each box claims, as the loss does it')
    return parser


def main():
    args = build_parser().parse_args()
    os.chdir(pathlib.Path(__file__).resolve().parent)
    anchors, stats = measure(args)
    report(anchors, stats, args.input_size)


if __name__ == '__main__':
    main()
