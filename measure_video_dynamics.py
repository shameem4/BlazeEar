#!/usr/bin/env python3
"""Candidate dynamics across real video frames, with no annotation needed.

The demo page's tracker has three constants -- `minHits`, `maxMisses` and
`CONTINUE_MARGIN` -- and only the last was ever measured, by the TTA-view
proxy. The other two are about frame *ordering*, which no still-image proxy
reaches. This measures them on actual frame sequences.

Nothing here needs labels, which is the point. Two facts make that work:

  * `minHits` and `maxMisses` are questions about candidate PERSISTENCE --
    how long a candidate location survives, how often its score crosses the
    confirm threshold between adjacent frames. Whether the candidate is an
    ear does not enter.
  * Geometry splits the population for free. A face crop is centred on the
    face, so |u| ~ 0 is the nose and brow and an ear is lateral. Midline
    candidates are mostly false positives and lateral ones mostly ears. It is
    a noisy proxy -- 11.4% of real ears are inside |u| < 0.15 on val_v2 -- but
    it is enough for the question actually being asked, which is whether
    midline candidates persist the way ears do.

That question is load-bearing and has been answered wrongly here twice by
reasoning rather than measurement. If midline candidates are TRANSIENT, then
`minHits` filters them and the demo can afford a lower confirm score. If they
are PERSISTENT -- a nose bridge does not move -- then no amount of temporal
evidence separates them and the entry bar has to stay high.

    python measure_video_dynamics.py --videos DIR --limit-videos 20

Distribution caveat, which belongs with any number this prints: whatever is
in `--videos` is the measured distribution. Exercise or hand-held footage has
smaller, faster, more turned faces than a webcam selfie, so it will overstate
instability relative to the demo's actual use. Phase 2 -- a small labelled set
from a licensed source, for the true-positive/false-positive split and the
real recall figure -- is deliberately not attempted here.
"""
import argparse
import collections
import os
import pathlib

import cv2
import numpy as np
import torch

from evaluate_two_stage import detect_two_stage, load_ear_model
from make_face_crops import crop_window, load_face_detector
from utils.config import (DEFAULT_INPUT_SIZE, FACE_CROP_EXPAND,
                          FACE_CROP_MAX_FACES, FACE_CROP_THRESHOLD,
                          TTA_SCORE_THRESHOLD)
from utils.tta import VIEWS
from utils.box_tracker import BoxTracker

MATCH_IOU = 0.3        # the tracker default; utils/box_tracker.py owns it
MIDLINE = 0.15
LATERAL = 0.25


def iou(a, b):
    y0 = max(a[0], b[0]); x0 = max(a[1], b[1])
    y1 = min(a[2], b[2]); x1 = min(a[3], b[3])
    inter = max(0.0, y1 - y0) * max(0.0, x1 - x0)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
            - inter)
    return inter / area if area > 0 else 0.0


def link_frames(frames):
    """Greedy nearest-box linking into candidate tracks.

    Deliberately the same matching rule the demo tracker uses, so the
    lifetimes measured here are the lifetimes it would see. Scores are kept
    per frame, with None for a frame the candidate was absent from, because
    the gaps are what `maxMisses` is about.
    """
    tracks = []
    for t, dets in enumerate(frames):
        live = [tr for tr in tracks if t - tr['last'] <= 1]
        claimed = set()
        for tr in live:
            best, best_iou = -1, MATCH_IOU
            for i, d in enumerate(dets):
                if i in claimed:
                    continue
                v = iou(tr['box'], d['box'])
                if v >= best_iou:
                    best, best_iou = i, v
            if best < 0:
                continue
            claimed.add(best)
            tr['box'] = dets[best]['box']
            tr['scores'].append(dets[best]['score'])
            tr['u'].append(dets[best]['u'])
            tr['last'] = t
        for i, d in enumerate(dets):
            if i in claimed:
                continue
            tracks.append({'box': d['box'], 'scores': [d['score']],
                           'u': [d['u']], 'first': t, 'last': t})
    return tracks


def simulate(frames, confirm, keep, min_hits, max_misses, fps=30.0):
    """Replay the SHIPPED tracker over a real frame sequence.

    This used to be a hand-written mirror of the tracker in docs/index.html,
    and the mirror drifted: it assigned the raw detection box while the page
    matched IoU against the smoothed one, which is worth 189 -> 212 lateral
    confirmations on this corpus. So it now calls `utils.box_tracker`, the
    same code the demo loads, and a divergence of that kind is no longer
    expressible. The box smoother is left at None here deliberately: matching
    reads the raw box, so smoothing cannot change which detection joins which
    track, and leaving it out keeps this measurement independent of a display
    choice.

    Returns one record per confirmed track so the outcome can be split by
    geometry. `u` is collected on every matched frame including the
    pre-confirmation ones, which is what the original did and what the
    position classification was measured against.
    """
    tracker = BoxTracker(confirm_score=confirm, continue_score=keep,
                         min_hits=min_hits, max_misses=max_misses,
                         match_iou=MATCH_IOU, smoother=None)
    us, drawn = {}, {}
    for i, dets in enumerate(frames):
        detections = [{'xmin': d['box'][1], 'ymin': d['box'][0],
                       'width': d['box'][3] - d['box'][1],
                       'height': d['box'][2] - d['box'][0],
                       'confidence': d['score']} for d in dets]
        out = tracker.update(detections, i / fps)
        # Read the tracker's own state, so pre-confirmation frames are counted
        # the way the original mirror counted them.
        for tr in tracker.tracks:
            if tr['matched'] and tr['det_index'] is not None:
                us.setdefault(tr['id'], []).append(dets[tr['det_index']]['u'])
        for r in out:
            drawn[r['id']] = drawn.get(r['id'], 0) + 1
    return [{'u': us.get(tid, []), 'drawn': n} for tid, n in drawn.items()]


def candidates_for_frame(face, ear, image, views, max_faces):
    boxes, scores, _ = detect_two_stage(
        face, ear, image, FACE_CROP_EXPAND, max_faces, views=views)
    if not len(boxes):
        with torch.no_grad():
            faces = face.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        return [], [[f[0], f[1], f[2], f[3]] for f in faces[:max_faces]]
    with torch.no_grad():
        faces = face.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
    windows = [crop_window(f, FACE_CROP_EXPAND, image.shape)
               for f in faces[:max_faces]]
    out = []
    # The face stage's own stability, because everything downstream inherits
    # it: if the crop moves between frames the ear model sees a different
    # input and its score moves with it, whatever the ear model does.
    face_boxes = [[f[0], f[1], f[2], f[3]] for f in faces[:max_faces]]
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i].tolist()
        cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
        u = np.nan
        if windows:
            x0, y0, side = min(
                windows, key=lambda w: max(abs(cx - (w[0] + w[2] / 2)),
                                           abs(cy - (w[1] + w[2] / 2))))
            u = (cx - (x0 + side / 2)) / side
        out.append({'box': [ymin, xmin, ymax, xmax],
                    'score': float(scores[i]), 'u': u})
    return out, face_boxes


def collect(args):
    face = load_face_detector(FACE_CROP_THRESHOLD, args.device)
    ear = load_ear_model(args.crop_checkpoint, args.floor, args.device,
                         input_size=args.input_size)
    views = VIEWS[:args.views] if args.views else None

    paths = sorted(p for p in pathlib.Path(args.videos).iterdir()
                   if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv'))
    if args.limit_videos:
        paths = paths[:args.limit_videos]

    per_video, n_frames = [], 0
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        frames, face_seq = [], []
        while len(frames) < args.max_frames:
            ok, image = cap.read()
            if not ok:
                break
            if args.stride > 1 and (len(frames) * args.stride) % args.stride:
                continue
            for _ in range(args.stride - 1):
                cap.read()
            dets, fb = candidates_for_frame(face, ear, image, views,
                                            args.max_faces)
            frames.append(dets)
            face_seq.append(fb)
        cap.release()
        if len(frames) >= args.min_frames:
            per_video.append((frames, face_seq))
            n_frames += len(frames)
    return per_video, n_frames, len(paths)


def classify(u_list):
    u = np.nanmedian(np.abs(u_list)) if len(u_list) else np.nan
    if np.isnan(u):
        return 'unknown'
    if u < MIDLINE:
        return 'midline'
    if u > LATERAL:
        return 'lateral'
    return 'between'


def report(per_video, n_frames, n_paths, args):
    tracks = []
    for frames, _ in per_video:
        tracks += link_frames(frames)

    # Face-stage stability first: it bounds everything after it.
    counts, box_iou, gaps = [], [], 0
    for _, face_seq in per_video:
        prev = None
        for fb in face_seq:
            counts.append(len(fb))
            if not fb:
                gaps += 1
                prev = None
                continue
            if prev is not None:
                box_iou.append(max(iou(a, b) for a in fb for b in prev))
            prev = fb
    counts = np.array(counts)
    print('\nFACE stage stability, which bounds everything downstream')
    print(f'  frames with no face at all: {100 * gaps / len(counts):.1f}% '
          f'({gaps} of {len(counts)})')
    print(f'  faces per frame: mean {counts.mean():.2f}, '
          f'0 in {100 * (counts == 0).mean():.1f}% of frames')
    if box_iou:
        b = np.array(box_iou)
        print(f'  face-box IoU between adjacent frames: median {np.median(b):.3f}'
              f'   p10 {np.percentile(b, 10):.3f}   '
              f'below 0.5 in {100 * (b < 0.5).mean():.1f}% of transitions')
        print('  -> a crop whose IoU with the previous frame is under 0.5 is '
              'a substantially\n     different input to the ear model, so '
              'its score is free to jump')
    print(f'\n{len(per_video)} of {n_paths} videos usable, {n_frames} frames, '
          f'{len(tracks)} candidate tracks, floor {args.floor}')
    if not tracks:
        print('no candidates -- nothing to measure')
        return

    groups = collections.defaultdict(list)
    for tr in tracks:
        groups[classify(tr['u'])].append(tr)

    print('\ncandidate PERSISTENCE, by position (the load-bearing question)')
    print('  class      tracks   median life   p90 life   ever >= '
          f'{args.confirm}   mean peak')
    for name in ('midline', 'between', 'lateral', 'unknown'):
        g = groups.get(name)
        if not g:
            continue
        life = np.array([len(tr['scores']) for tr in g])
        peak = np.array([max(tr['scores']) for tr in g])
        print(f'  {name:<9} {len(g):7d}   {np.median(life):11.1f}   '
              f'{np.percentile(life, 90):8.1f}   '
              f'{100 * (peak >= args.confirm).mean():12.1f}%   '
              f'{peak.mean():9.3f}')

    print(f'\nscore CROSSINGS of {args.confirm} between adjacent frames '
          f'(the flicker itself)')
    print('  class      tracks >=2 frames   mean crossings   crossings/frame')
    for name in ('midline', 'lateral'):
        g = [tr for tr in groups.get(name, []) if len(tr['scores']) >= 2]
        if not g:
            continue
        cross = [int(np.sum(np.diff(
            (np.array(tr['scores']) >= args.confirm).astype(int)) != 0))
            for tr in g]
        frames = sum(len(tr['scores']) for tr in g)
        print(f'  {name:<9} {len(g):16d}   {np.mean(cross):14.2f}   '
              f'{sum(cross) / frames:15.3f}')

    print('\nwhat (minHits, maxMisses) would confirm, by position')
    print('  minHits  maxMisses   midline confirmed   lateral confirmed   '
          'ratio')
    for mh in (1, 2, 3, 4):
        for mm in (args.max_misses,):
            conf = []
            for frames, _ in per_video:
                conf += simulate(frames, args.confirm, args.keep, mh, mm)
            by = collections.Counter(classify(tr['u']) for tr in conf)
            mid, lat = by.get('midline', 0), by.get('lateral', 0)
            print(f'  {mh:^7}  {mm:^9}   {mid:17d}   {lat:17d}   '
                  f'{(lat / mid if mid else float("inf")):5.2f}')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--videos', required=True)
    p.add_argument('--crop-checkpoint',
                   default='runs/checkpoints_crop/BlazeEar_best.pth')
    p.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument('--views', type=int, default=len(VIEWS),
                   help='0 for the plain path; the demo runs 4')
    p.add_argument('--max-faces', type=int, default=FACE_CROP_MAX_FACES)
    p.add_argument('--limit-videos', type=int, default=20)
    p.add_argument('--max-frames', type=int, default=60)
    p.add_argument('--min-frames', type=int, default=10)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--floor', type=float, default=0.50)
    p.add_argument('--confirm', type=float, default=TTA_SCORE_THRESHOLD)
    p.add_argument('--keep', type=float, default=0.70)
    p.add_argument('--max-misses', type=int, default=3)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    os.chdir(pathlib.Path(__file__).resolve().parent)
    report(*collect(args), args=args)


if __name__ == '__main__':
    main()
