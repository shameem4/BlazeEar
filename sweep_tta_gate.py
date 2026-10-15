#!/usr/bin/env python3
"""Sweep the FUSED score gate end to end, and say where the false positives are.

`DETECTION_SCORE_THRESHOLD` was re-picked on a 2,193-ear crop-space sweep, and
`TTA_SCORE_THRESHOLD` was then set to follow it at 0.70 for a single deployed
default. That second move was never measured: TTA re-crops the original frame
at several expands, so it has no crop-space equivalent at all -- the sweep has
to run the whole two-stage path. The only prior TTA evidence is a 355-ear
end-to-end sweep in which 0.70 was the worst of four points.

Two questions, one pass:

  1. What does the fused gate actually cost and buy, per gate, with enough
     ears to resolve it.
  2. WHERE the admitted false positives sit. A face crop is centred on the
     face, so the nose bridge is the crop centre and an ear is lateral. Each
     detection's centre is reported in units of the nearest face box's side,
     so |u| ~ 0 is the midline and |u| ~ 0.3-0.5 is where ears live. That
     turns "flickering box on the nose bridge" into a population or a one-off.

    python sweep_tta_gate.py --crop-checkpoint runs/checkpoints/BlazeEar_best.pth

Detections come from `evaluate_two_stage.detect_two_stage`, the same path the
evaluator scores, rather than a reimplementation -- a hand-rolled crop or
letterbox is the single most productive source of confident wrong answers in
this repository. Boxes are [ymin, xmin, ymax, xmax] throughout, which is the
convention `crop_window` and `ground_truth` already speak.
"""
import argparse
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch

from evaluate_two_stage import (detect_two_stage, ground_truth, load_ear_model)
from make_face_crops import crop_window, load_face_detector
from utils.config import (DEFAULT_DATA_ROOT, DEFAULT_INPUT_SIZE,
                          FACE_CROP_EXPAND, FACE_CROP_MAX_FACES,
                          FACE_CROP_THRESHOLD)
from utils.detection_eval import pairwise_iou
from utils.tta import VIEWS

GATES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
IOU_HIT = 0.5
MIDLINE = 0.15      # |u| below this counts as the facial midline


def collect(args):
    """One pass: every detection's score, whether it hits, and where it is."""
    face = load_face_detector(args.face_threshold, args.device)
    ear = load_ear_model(args.crop_checkpoint, args.score_threshold,
                         args.device, input_size=args.input_size)
    views = VIEWS[:args.tta] if args.tta else None

    frame = pd.read_csv(args.csv)
    rows = []                      # one per detection
    n_gt = n_images = n_noface = 0
    for rel, group in frame.groupby('image_path'):
        image = cv2.imread(os.path.join(args.data_root, str(rel)))
        if image is None:
            continue
        gt, ignore = ground_truth(group)
        boxes, scores, n_faces = detect_two_stage(
            face, ear, image, args.expand, args.max_faces, views=views)
        n_images += 1
        n_gt += len(gt)
        if not n_faces:
            n_noface += len(gt)

        # Face geometry for the position read. The detector is deterministic,
        # so this second pass reproduces the crops detect_two_stage just used.
        with torch.no_grad():
            faces = face.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        windows = [crop_window(f, FACE_CROP_EXPAND, image.shape)
                   for f in faces[:args.max_faces]]

        iou_gt = (pairwise_iou(boxes, gt) if len(boxes) and len(gt)
                  else torch.zeros((len(boxes), 0)))
        iou_ig = (pairwise_iou(boxes, ignore) if len(boxes) and len(ignore)
                  else torch.zeros((len(boxes), 0)))
        for i in range(len(boxes)):
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
            u = v = np.nan
            if windows:
                x0, y0, side = min(
                    windows, key=lambda w: max(abs(cx - (w[0] + w[2] / 2)),
                                               abs(cy - (w[1] + w[2] / 2))))
                u = (cx - (x0 + side / 2)) / side
                v = (cy - (y0 + side / 2)) / side
            # Which GT this detection claims, if any. Identity matters: two
            # detections on one ear are one TP and one duplicate FP, not two
            # TPs, and recall counts GT boxes rather than detections.
            gt_idx = -1
            if iou_gt.shape[1]:
                j = int(iou_gt[i].argmax())
                if float(iou_gt[i, j]) >= IOU_HIT:
                    gt_idx = j
            rows.append({
                'score': float(scores[i]),
                'gt_idx': gt_idx,
                'ignore_hit': (bool(iou_ig[i].max() >= IOU_HIT)
                               if iou_ig.shape[1] else False),
                'u': u, 'v': v, 'image': str(rel),
            })
    return pd.DataFrame(rows), n_gt, n_images, n_noface


def report(det, n_gt, n_images, n_noface, label):
    # An auto-labelled box is neither credited nor charged.
    pred = det[~((det.gt_idx < 0) & det.ignore_hit)]

    print(f'\n{label}: {n_images} images, {n_gt} human ears, '
          f'{len(pred)} candidates above {det.score.min():.2f}')
    if n_noface:
        print(f'  {n_noface} ears in images where the face stage found '
              f'nothing -- a miss at every gate')

    def match(keep):
        """Greedy by descending score; each GT box can be claimed once."""
        tp = dup = 0
        for _, g in keep.groupby('image', sort=False):
            seen = set()
            for r in g.sort_values('score', ascending=False).itertuples(False):
                if r.gt_idx < 0:
                    continue
                if r.gt_idx in seen:
                    dup += 1
                else:
                    seen.add(r.gt_idx)
                    tp += 1
        return tp, dup

    print('\n gate  precision  recall      F1   FP/image   midline FP/image')
    print(' ' + '-' * 62)
    for g in GATES:
        keep = pred[pred.score >= g]
        tp, dup = match(keep)
        fp = len(keep) - tp
        rec = tp / n_gt if n_gt else 0.0
        prec = tp / len(keep) if len(keep) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        mid = keep[(keep.gt_idx < 0) & (keep.u.abs() < MIDLINE)]
        print(f' {g:.2f}    {prec:.3f}    {rec:.3f}   {f1:.4f}   '
              f'{fp / n_images:6.3f}      {len(mid) / n_images:6.3f}')

    fp = pred[pred.gt_idx < 0]
    print(f'\nwhere the false positives are, in face-box sides from the crop '
          f'centre\n(|u| < {MIDLINE} is the facial midline, i.e. nose/brow; '
          f'ears sit near |u| 0.3-0.5)')
    print('\n score band      n   midline   median |u|   median v')
    print(' ' + '-' * 58)
    for lo, hi in ((0.60, 0.70), (0.70, 0.85), (0.85, 1.01)):
        band = fp[(fp.score >= lo) & (fp.score < hi)].dropna(subset=['u'])
        if not len(band):
            print(f' {lo:.2f}-{hi:.2f}       0')
            continue
        m = band.u.abs() < MIDLINE
        print(f' {lo:.2f}-{hi:.2f}  {len(band):6d}   {100 * m.mean():5.1f}%   '
              f'{band.u.abs().median():9.3f}   {band.v.median():+8.3f}')

    admitted = fp[(fp.score >= 0.70) & (fp.score < 0.85)].dropna(subset=['u'])
    if len(admitted):
        mid = admitted[admitted.u.abs() < MIDLINE]
        print(f'\nThe band the gate change admitted, 0.70 <= s < 0.85: '
              f'{len(admitted)} false positives,\n{len(mid)} of them on the '
              f'midline ({100 * len(mid) / len(admitted):.1f}%), '
              f'{len(mid) / n_images:.3f} per image.')
        if len(mid):
            top = mid.nlargest(min(5, len(mid)), 'score')
            print('\nhighest-scoring midline false positives '
                  '(these are the flickering ones):')
            for r in top.itertuples(index=False):
                print(f'  {r.score:.3f}  u={r.u:+.3f} v={r.v:+.3f}  {r.image}')


def midline_analysis(det, n_gt, n_images, gates=(0.70, 0.85)):
    """What a geometric midline veto would cost, against what it buys.

    The score gate trades recall for precision along one curve. Position is a
    different axis: the crop is centred on the face, so |u| ~ 0 is the nose
    and brow, and an ear is lateral by construction. If true ears are scarce
    near the midline and false positives are not, a veto there buys precision
    at a far better rate than raising the gate -- and if they are equally
    scarce, this idea is dead and the table below says so.
    """
    pred = det[~((det.gt_idx < 0) & det.ignore_hit)].dropna(subset=['u'])
    tp_all, fp_all = pred[pred.gt_idx >= 0], pred[pred.gt_idx < 0]
    print('\n|u| distribution, true positives vs false (all scores >= '
          f'{det.score.min():.2f})')
    for lab, g in (('true ears', tp_all), ('false positives', fp_all)):
        q = g.u.abs().quantile([0.01, 0.05, 0.10, 0.25, 0.50]).round(3)
        print(f'  {lab:<16} n={len(g):5d}  p1 {q[0.01]}  p5 {q[0.05]}  '
              f'p10 {q[0.10]}  p25 {q[0.25]}  median {q[0.50]}')

    for gate in gates:
        keep = pred[pred.score >= gate]
        tp0 = int((keep.gt_idx >= 0).sum())
        fp0 = int((keep.gt_idx < 0).sum())
        print(f'\na midline veto at gate {gate:.2f}  '
              f'(baseline {tp0} TP, {fp0} FP, {fp0 / n_images:.3f} FP/image)')
        print('  veto |u| <   TP lost   FP removed   recall     FP/image')
        print('  ' + '-' * 54)
        for band in (0.05, 0.10, 0.15, 0.20, 0.25):
            veto = keep.u.abs() < band
            tp_lost = int(((keep.gt_idx >= 0) & veto).sum())
            fp_cut = int(((keep.gt_idx < 0) & veto).sum())
            rec = (tp0 - tp_lost) / n_gt if n_gt else 0.0
            print(f'  {band:.2f}        {tp_lost:5d}      {fp_cut:5d}'
                  f'       {rec:.3f}    {(fp0 - fp_cut) / n_images:6.3f}')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--csv', default='data/splits/val_v2.csv')
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--tta', type=int, default=4,
                   help='views to fuse; 0 is the plain single-crop path')
    p.add_argument('--expand', type=float, default=FACE_CROP_EXPAND)
    p.add_argument('--face-threshold', type=float, default=FACE_CROP_THRESHOLD)
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--score-threshold', type=float, default=0.50,
                   help='pre-gate floor; the sweep cannot go below it')
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    label = f'TTA {args.tta} views' if args.tta else 'plain, 1 view'
    det, n_gt, n_images, n_noface = collect(args)
    report(det, n_gt, n_images, n_noface, label=label)
    midline_analysis(det, n_gt, n_images)


if __name__ == '__main__':
    main()
