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

Second, the negative budget concentrates on a prior that has almost no
positives. `--mine` measures this directly rather than inferring it from slot
counts, which is the mistake this script's first version made: mining is
adaptive, so a prior's negative exposure follows how CONFUSING it is, not how
many slots it owns. At convergence one prior -- 129.8 x 178.2 px, 7.1% of
slots, 0.6% of positives -- absorbs 54% of the whole negative budget, a 91x
over-representation, while the three priors carrying 63% of positives receive
0.6% between them. It is learned: at initialisation that same prior takes
2.1%.

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
                          HARD_NEGATIVE_PRIOR_CAP, HARD_NEGATIVE_RATIO,
                          HUMAN_ANNOTATION_SOURCES)

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


def measure_mining(args):
    """Where the mined negatives actually land, per prior.

    Reproduces the loss's own selection by CALLING it, rather than by
    reimplementing a global top-k here. An earlier version of this function
    did reimplement it, and once HARD_NEGATIVE_PRIOR_CAP existed that copy
    silently ignored the cap -- so it reported what uncapped mining would pick
    on capped weights and read as though the cap had failed to apply.
    """
    import torch
    from torch.utils.data import DataLoader
    from dataloader import CSVDetectorDataset, collate_detector_fn
    from blazeear import BlazeEar
    from utils.anchor_utils import anchor_options

    model = BlazeEar()
    model.generate_anchors(anchor_options)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state),
                              strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)

    anchors = get_anchors(input_size=args.input_size).cpu().numpy()
    anchor_tensor = get_anchors(input_size=args.input_size).to(device)
    prior_of = {i: (round(float(w), 4), round(float(h), 4))
                for i, (w, h) in enumerate(anchors[:, 2:])}
    from loss_functions import BlazeEarDetectionLoss
    loss = BlazeEarDetectionLoss(
        hard_negative_prior_cap=args.prior_cap)
    print(f'mining with HARD_NEGATIVE_PRIOR_CAP = {args.prior_cap}')

    dataset = CSVDetectorDataset(args.csv, args.images, augment=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        collate_fn=collate_detector_fn, num_workers=4)

    mined, positives = collections.Counter(), collections.Counter()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.batches:
                break
            _, logits = model(batch['image'].to(device))
            logits = logits.squeeze(-1)
            targets = batch['anchor_targets'].to(device)
            positive = targets[..., 0] > 0.5
            excluded = positive.clone()
            ignore = batch.get('anchor_ignore')
            if ignore is not None:
                excluded = excluded | ignore.to(device).bool()

            per_image = logits.shape[0]
            budget = max(int(positive.float().sum().item()
                             * HARD_NEGATIVE_RATIO) // per_image, 10)
            budget = min(budget, int((~excluded).sum(1).min().item()))
            masked = logits.masked_fill(excluded,
                                        torch.finfo(logits.dtype).min)
            selected = loss._mine_negatives(masked, budget, anchor_tensor)
            # _mine_negatives returns values, not indices, so recover which
            # anchors they were. Ties are vanishingly unlikely in float logits
            # and would only shuffle counts within a prior.
            for row in range(masked.shape[0]):
                picked = torch.isin(masked[row], selected[row])
                for i in torch.nonzero(picked, as_tuple=True)[0].tolist():
                    mined[prior_of[int(i)]] += 1
            for i in torch.nonzero(positive.cpu())[:, 1].numpy():
                positives[prior_of[int(i)]] += 1
    return anchors, mined, positives


def report_mining(anchors, mined, positives, size):
    total_mined = max(sum(mined.values()), 1)
    total_pos = max(sum(positives.values()), 1)
    print('\nwhere the mined negatives actually land. Mining is ADAPTIVE -- it '
          'takes a\nglobal top-k of background logits -- so this follows how '
          'confusing a prior is,\nnot how many slots it owns, and slot counts '
          'do not predict it:')
    print(f"\n{'prior (px)':>18s} {'slots':>6s} {'% mined neg':>12s} "
          f"{'% positives':>12s} {'neg:pos':>9s}")
    for (width, height), indices in distinct_priors(anchors).items():
        share_neg = 100 * mined[(width, height)] / total_mined
        share_pos = 100 * positives[(width, height)] / total_pos
        ratio = f'{share_neg / share_pos:8.1f}x' if share_pos else '     inf'
        print(f'{width * size:7.1f} x {height * size:6.1f} {len(indices):6d} '
              f'{share_neg:11.1f}% {share_pos:11.1f}% {ratio}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mine', action='store_true',
                        help='also measure where the mined negatives land, '
                             'which needs a forward pass and a checkpoint')
    parser.add_argument('--checkpoint',
                        default='model_weights/BlazeEar_best_256px.pth',
                        help='weights to mine against; pass "" for untrained, '
                             'which is what mining does at the start of a run')
    parser.add_argument('--images', default='data/face_crops',
                        help='image root for --mine')
    parser.add_argument('--batches', type=int, default=40,
                        help='batches of 32 to accumulate for --mine')
    parser.add_argument('--prior-cap', type=float,
                        default=HARD_NEGATIVE_PRIOR_CAP,
                        help='the cap to mine UNDER; pass 1.0 to see what '
                             'uncapped mining would select on these weights')
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
    if args.mine:
        report_mining(*measure_mining(args), args.input_size)


if __name__ == '__main__':
    main()
