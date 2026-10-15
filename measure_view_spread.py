#!/usr/bin/env python3
"""How much does one ear's score move between looks? A proxy for the flicker.

`docs/index.html` confirms a track at 0.85 and keeps it at 0.70. The recall
that buys over a flat 0.85 is exactly the ears whose score *straddles* 0.85
between frames: seen once above it, then held below. That quantity was
asserted when the tracker shipped, not measured, because every split here is
still images.

The four TTA views are the available proxy. They are four looks at one ear --
crop expands 1.35 / 1.50 / 1.65 and a horizontal flip -- and fusion takes the
max, so the spread across views stands in for the spread across frames.

Read the proxy's limits before the numbers. Varying the crop expand perturbs
the face box, which is a fair analogue of the box jitter a webcam produces
frame to frame. It says nothing about lighting, motion blur, or a head that
is actually moving, and the flip is not a temporal transformation at all --
so the scale-only columns are the more honest ones. This bounds one component
of frame-to-frame variation, not all of it.

    python measure_view_spread.py --csv data/splits/val_v2.csv

Each ear is queried in every view independently, at its own ground-truth
location, so a view that fails to produce a candidate scores 0 rather than
dropping out of the sample. Views are built from the same `crop_window` /
`flip_boxes_x` primitives `detect_two_stage` uses; only the fusion is held
back, since per-view identity is the whole point.
"""
import argparse
import collections
import csv
import os
import pathlib

import cv2
import numpy as np
import torch

from evaluate_two_stage import detect_single_stage, load_ear_model
from make_face_crops import crop_window, load_face_detector
from utils.config import (DEFAULT_INPUT_SIZE, FACE_CROP_MAX_FACES,
                          FACE_CROP_THRESHOLD, HUMAN_ANNOTATION_SOURCES,
                          TTA_SCORE_THRESHOLD, DETECTION_SCORE_THRESHOLD)
from utils.detection_eval import pairwise_iou
from utils.tta import VIEWS, flip_boxes_x

IOU_HIT = 0.5
MIDLINE = 0.15


def human_ears(path):
    out = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        if row['annotation_source'] not in HUMAN_ANNOTATION_SOURCES:
            continue
        w, h = float(row['w']), float(row['h'])
        if w > 0 and h > 0:
            out[row['image_path']].append(
                [float(row['y1']), float(row['x1']),
                 float(row['y1']) + h, float(row['x1']) + w])
    return out


def per_view(args):
    face = load_face_detector(FACE_CROP_THRESHOLD, args.device)
    ear = load_ear_model(args.crop_checkpoint, args.floor, args.device,
                         input_size=args.input_size)
    views = VIEWS[:args.views]

    ear_scores, fp_rows = [], []
    for rel, boxes in human_ears(args.csv).items():
        image = cv2.imread(os.path.join(args.data_root, str(rel)))
        if image is None:
            continue
        gt = torch.tensor(boxes, dtype=torch.float32)
        with torch.no_grad():
            faces = face.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        if not len(faces):
            continue
        windows = [crop_window(f, e, image.shape)
                   for e, _ in views for f in faces[:args.max_faces]]

        scores = np.zeros((len(boxes), len(views)))
        view_dets = []
        for v, (expand, flip) in enumerate(views):
            vb, vs = [], []
            for f in faces[:args.max_faces]:
                x0, y0, side = crop_window(f, expand, image.shape)
                patch = image[y0:y0 + side, x0:x0 + side]
                if patch.size == 0:
                    continue
                if flip:
                    patch = patch[:, ::-1].copy()
                b, s = detect_single_stage(ear, patch)
                if not len(s):
                    continue
                if flip:
                    b = flip_boxes_x(b, patch.shape[1])
                vb.append(b + torch.tensor([y0, x0, y0, x0],
                                           dtype=torch.float32))
                vs.append(s)
            if not vb:
                view_dets.append((torch.zeros((0, 4)), torch.zeros(0)))
                continue
            vb, vs = torch.cat(vb), torch.cat(vs)
            view_dets.append((vb, vs))
            # This view's own best look at each ear.
            ious = pairwise_iou(vb, gt)
            for j in range(len(boxes)):
                hit = (ious[:, j] >= IOU_HIT).nonzero().flatten()
                if len(hit):
                    scores[j, v] = float(vs[hit].max())

        for j in range(len(boxes)):
            ear_scores.append(scores[j])

        # False positives, kept per view so agreement can be counted.
        for v, (vb, vs) in enumerate(view_dets):
            if not len(vs):
                continue
            ious = pairwise_iou(vb, gt)
            best = ious.max(1).values if ious.shape[1] else torch.zeros(len(vb))
            for i in range(len(vb)):
                if float(best[i]) >= IOU_HIT:
                    continue
                ymin, xmin, ymax, xmax = vb[i].tolist()
                cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
                x0, y0, side = min(
                    windows, key=lambda w: max(abs(cx - (w[0] + w[2] / 2)),
                                               abs(cy - (w[1] + w[2] / 2))))
                fp_rows.append({'image': rel, 'view': v,
                                'score': float(vs[i]),
                                'box': [ymin, xmin, ymax, xmax],
                                'u': (cx - (x0 + side / 2)) / side})
    return np.array(ear_scores), fp_rows


def cluster_fp_views(fp_rows, n_views):
    """How many distinct views saw each false-positive location."""
    by_image = collections.defaultdict(list)
    for r in fp_rows:
        by_image[r['image']].append(r)
    counts = []
    for rows in by_image.values():
        boxes = torch.tensor([r['box'] for r in rows], dtype=torch.float32)
        ious = pairwise_iou(boxes, boxes)
        order = np.argsort([-r['score'] for r in rows])
        used = np.zeros(len(rows), bool)
        for idx in order:
            if used[idx]:
                continue
            members = ((ious[idx] >= IOU_HIT).numpy() & ~used).nonzero()[0]
            used[members] = True
            counts.append({
                'views': len({rows[m]['view'] for m in members}),
                'score': max(rows[m]['score'] for m in members),
                'u': rows[idx]['u'],
            })
    return counts


def report(scores, fp_rows, n_views, confirm, keep):
    seen = scores.max(1) > 0
    s = scores[seen]
    print(f'\n{len(scores)} human ears, {seen.sum()} found by at least one '
          f'view at the {n_views}-view setting\n')
    spread = s.max(1) - s.min(1)
    nz = np.where(s > 0, s, np.nan)
    print('per-ear score spread across views (0 means a view missed it)')
    print(f'  max - min       median {np.median(spread):.3f}   '
          f'p75 {np.percentile(spread, 75):.3f}   '
          f'p90 {np.percentile(spread, 90):.3f}')
    found = np.nansum(~np.isnan(nz), 1)
    print('  views that found the ear: '
          + '  '.join(f'{k}:{100 * (found == k).mean():.1f}%'
                      for k in range(1, n_views + 1)))

    print(f'\nwhat the tracker can and cannot reach (confirm {confirm}, '
          f'keep {keep})')
    always = (s.min(1) >= confirm)
    never = (s.max(1) < confirm)
    straddle = (s.max(1) >= confirm) & (s.min(1) < confirm)
    useful = straddle & (s.min(1) >= keep)
    below = straddle & (s.min(1) < keep)
    n = len(s)
    for lab, m in ((f'every view >= {confirm}  (tracker irrelevant)', always),
                   (f'no view reaches {confirm} (tracker cannot help)', never),
                   (f'straddles {confirm}       (the tracker\'s case)',
                    straddle)):
        print(f'  {lab}: {100 * m.mean():5.1f}%  ({m.sum()} ears)')
    print(f'      of those, worst view still >= {keep} (held): '
          f'{100 * useful.sum() / max(straddle.sum(), 1):.1f}%  '
          f'({useful.sum()} ears, {100 * useful.sum() / n:.1f}% of all found)')
    print(f'      worst view below {keep} (lost anyway):        '
          f'{100 * below.sum() / max(straddle.sum(), 1):.1f}%  ({below.sum()})')

    print(f'\nwhat CONTINUE_MARGIN should be: of the {straddle.sum()} '
          f'straddling ears,\nthe share whose WORST view still clears a '
          f'given keep score')
    print('  keep   held   as % of all found   margin below confirm')
    for k in (0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50):
        held = straddle & (s.min(1) >= k)
        print(f'  {k:.2f}  {held.sum():5d}   {100 * held.sum() / n:15.1f}%'
              f'   {confirm - k:19.2f}')

    # A >=2-view rule can only be judged against what it costs on ears, so
    # cross-tabulate: an ear found by one view alone, whose single look is
    # what would confirm it, is one this rule deletes outright.
    found_k = np.nansum(~np.isnan(nz), 1)
    print('\nears by how many views found them, and whether any view '
          'confirms')
    print('  views   ears   any view >= confirm   would a >=2-view rule '
          'delete it')
    for k in range(1, n_views + 1):
        m = found_k == k
        if not m.sum():
            continue
        conf = m & (s.max(1) >= confirm)
        lost = conf.sum() if k == 1 else 0
        print(f'  {k}     {m.sum():5d}   {conf.sum():17d}   {lost:23d}')

    counts = cluster_fp_views(fp_rows, n_views)
    print(f'\n{len(counts)} false-positive locations, by how many views saw '
          f'them')
    print('  views  all FPs   midline FPs   median score   max >= confirm')
    for k in range(1, n_views + 1):
        grp = [c for c in counts if c['views'] == k]
        if not grp:
            continue
        mid = [c for c in grp if abs(c['u']) < MIDLINE]
        hi = [c for c in grp if c['score'] >= confirm]
        print(f'  {k}      {len(grp):6d}   {len(mid):11d}   '
              f'{np.median([c["score"] for c in grp]):12.3f}   '
              f'{100 * len(hi) / len(grp):13.1f}%')
    single = [c for c in counts if c['views'] == 1 and c['score'] >= confirm]
    multi = [c for c in counts if c['views'] > 1 and c['score'] >= confirm]
    print(f'\nFalse positives that would CONFIRM a track (score >= {confirm}): '
          f'{len(single) + len(multi)}\n  seen by one view only: {len(single)}'
          f'  ({100 * len(single) / max(len(single) + len(multi), 1):.1f}%)'
          f'  -- these a >=2-view rule would remove')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--csv', default='data/splits/val_v2.csv')
    p.add_argument('--data-root', default='data/raw')
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--views', type=int, default=len(VIEWS))
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--floor', type=float, default=0.10,
                   help='per-view score floor; below this a view reads as 0')
    p.add_argument('--confirm', type=float, default=TTA_SCORE_THRESHOLD)
    p.add_argument('--keep', type=float, default=0.70)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    scores, fp_rows = per_view(args)
    report(scores, fp_rows, args.views, args.confirm, args.keep)
    print(f'\n(plain gate {DETECTION_SCORE_THRESHOLD}, fused '
          f'{TTA_SCORE_THRESHOLD}; views = {list(VIEWS[:args.views])})')


if __name__ == '__main__':
    main()
