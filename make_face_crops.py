"""Build a face-cropped ear dataset, the way MediaPipe chains its detectors.

MediaPipe's own pipelines run a coarse detector on the full frame and then a
fine model on a crop around each hit. Ears in this dataset have a median
height of 14 px once a full frame is squeezed into BlazeEar's 128 px input,
which is below the smallest BlazeFace anchor. Cropping to a face first raises
that to roughly 30 px at 1.5x expansion.

The cost is recall: an ear whose face BlazeFace misses never reaches the
second stage at all. `--report-only` prints that budget without writing
anything.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

from blazeear import BlazeEar
from blazebase import load_mediapipe_weights
from utils.anchor_utils import generate_reference_anchors
from utils.config import (
    DEFAULT_DATA_ROOT,
    HUMAN_ANNOTATION_SOURCES,
    IGNORE_ANNOTATION_SOURCE,
)

FACE_WEIGHTS = 'model_weights/blazeface.pth'
# Fraction of an ear that must survive the crop for it to stay ground truth.
MIN_VISIBLE_FRACTION = 0.5


def load_face_detector(score_threshold=0.5, device='cuda'):
    """MediaPipe's front-facing BlazeFace, at its original anchor geometry.

    The v2 ear anchors are fitted to ear boxes, so the face detector has to
    carry the square anchors its weights were trained with.
    """
    detector = BlazeEar(use_batchnorm=False)
    load_mediapipe_weights(detector, FACE_WEIGHTS, strict=False)
    detector.eval().to(device)
    detector.min_score_thresh = score_threshold
    anchors, _, _ = generate_reference_anchors(fixed_anchor_size=True)
    detector.anchors = anchors.to(device)
    return detector


def crop_window(face_box, expand, image_shape):
    """A square window of `expand` x the face box, clipped to the image."""
    ymin, xmin, ymax, xmax = face_box[:4]
    height, width = image_shape[:2]
    side = expand * max(xmax - xmin, ymax - ymin)
    # Clamping the side first keeps the window square whenever the image
    # allows it; shifting the centre afterwards keeps it inside the frame.
    side = min(side, width, height)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    x0 = int(round(min(max(cx - side / 2, 0), width - side)))
    y0 = int(round(min(max(cy - side / 2, 0), height - side)))
    return x0, y0, int(round(side))


def remap_boxes(rows, window):
    """Move annotation rows into crop coordinates.

    An ear is kept when its centre falls inside the window. A kept ear that
    the window cuts through is clipped, and demoted to an ignore region when
    less than MIN_VISIBLE_FRACTION of it survives -- a sliver of an ear is
    neither a box the model should learn nor a miss it should be punished for.
    """
    x0, y0, side = window
    out, demoted = [], 0
    for row in rows:
        x1, y1, w, h = float(row.x1), float(row.y1), float(row.w), float(row.h)
        source = str(row.annotation_source)
        cx, cy = x1 + w / 2, y1 + h / 2
        is_ignore = source == IGNORE_ANNOTATION_SOURCE
        if not is_ignore and not (x0 <= cx < x0 + side and y0 <= cy < y0 + side):
            continue
        cx1, cy1 = max(x1 - x0, 0), max(y1 - y0, 0)
        cx2, cy2 = min(x1 + w - x0, side), min(y1 + h - y0, side)
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw <= 1 or ch <= 1:
            continue
        if not is_ignore and (cw * ch) < MIN_VISIBLE_FRACTION * (w * h):
            source = IGNORE_ANNOTATION_SOURCE
            demoted += 1
        out.append({
            'x1': int(round(cx1)), 'y1': int(round(cy1)),
            'w': int(round(cw)), 'h': int(round(ch)),
            'earside': row.earside, 'source': row.source,
            'annotation_source': source, 'confidence': row.confidence,
        })
    return out, demoted


def build(args):
    frame = pd.read_csv(args.csv)
    groups = list(frame.groupby('image_path', sort=False))
    if args.limit:
        groups = groups[:args.limit]
    detector = load_face_detector(args.face_threshold, args.device)

    out_rows = []
    stats = {'images': 0, 'no_face': 0, 'faces': 0, 'crops': 0,
             'ears_total': 0, 'ears_kept': 0, 'ears_no_face': 0,
             'ears_demoted': 0}
    heights = []

    os.makedirs(args.out_images, exist_ok=True)
    for image_path, rows in groups:
        image = cv2.imread(os.path.join(args.data_root, str(image_path)))
        if image is None:
            continue
        stats['images'] += 1
        rows = list(rows.itertuples(index=False))
        human = [r for r in rows
                 if str(r.annotation_source) in HUMAN_ANNOTATION_SOURCES]
        stats['ears_total'] += len(human)

        with torch.no_grad():
            faces = detector.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cpu().numpy()
        if not len(faces):
            stats['no_face'] += 1
            stats['ears_no_face'] += len(human)
            continue
        stats['faces'] += len(faces)

        stem = os.path.splitext(os.path.basename(str(image_path)))[0]
        subdir = str(image_path).split('/')[0].replace(' ', '_')
        for index, face in enumerate(faces[:args.max_faces]):
            window = crop_window(face, args.expand, image.shape)
            mapped, demoted = remap_boxes(rows, window)
            kept = [m for m in mapped
                    if m['annotation_source'] in HUMAN_ANNOTATION_SOURCES]
            if not kept and not args.keep_empty_crops:
                continue
            x0, y0, side = window
            patch = image[y0:y0 + side, x0:x0 + side]
            if patch.size == 0:
                continue
            name = f'{subdir}/{stem}_f{index}.jpg'
            os.makedirs(os.path.join(args.out_images, subdir), exist_ok=True)
            cv2.imwrite(os.path.join(args.out_images, name), patch,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats['crops'] += 1
            stats['ears_kept'] += len(kept)
            for m in kept:
                heights.append(128.0 * m['h'] / side)
            for m in mapped:
                m['image_path'] = name
                m['origin_image'] = image_path
                m['crop_x0'], m['crop_y0'], m['crop_side'] = x0, y0, side
                out_rows.append(m)
            stats['ears_demoted'] += demoted

        if stats['images'] % 500 == 0:
            print(f"  {stats['images']} images, {stats['crops']} crops",
                  flush=True)

    reach = stats['ears_kept'] / max(stats['ears_total'], 1)
    print(f"\nimages            {stats['images']}")
    print(f"no face found     {stats['no_face']} "
          f"({100 * stats['no_face'] / max(stats['images'], 1):.1f}%)")
    print(f"crops written     {stats['crops']}")
    print(f"ears in source    {stats['ears_total']}")
    print(f"ears in crops     {stats['ears_kept']} ({100 * reach:.1f}% reach)")
    print(f"  lost, no face   {stats['ears_no_face']}")
    print(f"  clipped to ignore {stats['ears_demoted']}")
    if heights:
        heights = np.array(heights)
        print(f"ear height at 128 px input: median {np.median(heights):.1f} px, "
              f"p10 {np.percentile(heights, 10):.1f}, "
              f"p90 {np.percentile(heights, 90):.1f}")
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/splits/master_relabelled.csv')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--out-images', default='data/face_crops')
    parser.add_argument('--out-csv', default='data/splits/face_crops.csv')
    parser.add_argument('--expand', type=float, default=1.5,
                        help='square crop side, in multiples of the face box')
    parser.add_argument('--face-threshold', type=float, default=0.5)
    parser.add_argument('--max-faces', type=int, default=8)
    parser.add_argument('--keep-empty-crops', action='store_true',
                        help='write crops that contain no ear, as negatives')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    result = build(args)
    if len(result):
        columns = ['image_path', 'x1', 'y1', 'w', 'h', 'earside', 'source',
                   'annotation_source', 'confidence', 'origin_image',
                   'crop_x0', 'crop_y0', 'crop_side']
        result[columns].to_csv(args.out_csv, index=False)
        print(f'\nwrote {len(result)} rows to {args.out_csv}')


if __name__ == '__main__':
    main()
