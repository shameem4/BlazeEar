"""
Relabelling pipeline: propose, review, apply.

82% of images in this dataset carry exactly one human-labelled ear, so most
frontal images contain a real but unlabelled second ear. Hard negative mining
selects the highest-scoring background anchors, which is precisely that ear, so
the detector is actively trained to suppress real ears. The POSE pseudo-labels
that were meant to fill the gap do not: 97.8% of them have zero overlap with any
human box, landing on eyebrows, hair and background objects.

This fills the gap properly. A detector trained only on human-verified boxes
proposes the missing ears, every proposal that disagrees with the existing
labels is queued, and a human accepts or rejects it. Proposals are proposals:
nothing enters the dataset without review.

    python relabel.py propose --weights runs/.../best.pt
    python relabel.py review          # interactive, resumable
    python relabel.py apply --output data/splits/master_relabelled.csv

`review` needs a display. `propose` and `apply` do not.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.config import HUMAN_ANNOTATION_SOURCES

DEFAULT_QUEUE = 'data/relabel/proposals.csv'

# A proposal matching an existing human box at this IoU is the same ear.
MATCH_IOU = 0.5

# Categories:
#   new      proposal overlaps no human box -- a candidate missing ear
#   conflict proposal overlaps a human box loosely (disputed extent)
#   missed   a human box no proposal matched -- possible bad ground truth
CATEGORIES = ('new', 'conflict', 'missed')


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4], both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return (inter / np.clip(area_a + area_b - inter, 1e-9, None)).astype(np.float32)


def human_boxes_by_image(csv_path: str) -> Dict[str, np.ndarray]:
    """image_path -> [N,4] xyxy of human-verified boxes."""
    df = pd.read_csv(csv_path)
    if 'annotation_source' in df.columns:
        df = df[df['annotation_source'].isin(HUMAN_ANNOTATION_SOURCES)]
    out: Dict[str, np.ndarray] = {}
    for image_path, rows in df.groupby('image_path', sort=False):
        boxes = rows[['x1', 'y1', 'w', 'h']].to_numpy(dtype=np.float32)
        xyxy = np.stack([
            boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        ], axis=1)
        out[str(image_path)] = xyxy
    return out


def categorise(
    proposals: np.ndarray,
    scores: np.ndarray,
    human: np.ndarray
) -> Tuple[List[dict], List[dict]]:
    """Split proposals into new/conflict, and find unmatched human boxes."""
    ious = iou_matrix(proposals, human)
    proposal_rows: List[dict] = []

    for i in range(len(proposals)):
        best = float(ious[i].max()) if human.size else 0.0
        if best >= MATCH_IOU:
            continue                      # agrees with an existing label
        category = 'conflict' if best > 0.1 else 'new'
        x1, y1, x2, y2 = proposals[i]
        proposal_rows.append({
            'x1': float(x1), 'y1': float(y1),
            'w': float(x2 - x1), 'h': float(y2 - y1),
            'confidence': float(scores[i]),
            'best_iou_with_human': best,
            'category': category,
        })

    missed_rows: List[dict] = []
    if human.size:
        per_human = ious.max(axis=0) if len(proposals) else np.zeros(len(human))
        for j in range(len(human)):
            if per_human[j] < MATCH_IOU:
                x1, y1, x2, y2 = human[j]
                missed_rows.append({
                    'x1': float(x1), 'y1': float(y1),
                    'w': float(x2 - x1), 'h': float(y2 - y1),
                    'confidence': 0.0,
                    'best_iou_with_human': float(per_human[j]),
                    'category': 'missed',
                })
    return proposal_rows, missed_rows


def cmd_propose(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    human = human_boxes_by_image(args.csv)
    image_paths = list(human.keys())
    print(f'{len(image_paths)} images with human labels in {args.csv}')

    model = YOLO(args.weights)
    rows: List[dict] = []
    root = Path(args.data_root)

    batch = args.batch_size
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        results = model.predict(
            [str(root / p) for p in chunk], conf=args.conf, verbose=False,
            device=args.device,
        )
        for image_path, result in zip(chunk, results):
            boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                     else np.zeros((0, 4), dtype=np.float32))
            scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                      else np.zeros((0,), dtype=np.float32))
            proposals, missed = categorise(boxes, scores, human[image_path])
            for row in proposals + missed:
                row['image_path'] = image_path
                row['review'] = 'pending'
                rows.append(row)
        if (start // batch) % 20 == 0:
            print(f'  {min(start + batch, len(image_paths))}/{len(image_paths)} images', flush=True)

    queue = pd.DataFrame(rows, columns=[
        'image_path', 'x1', 'y1', 'w', 'h', 'confidence',
        'best_iou_with_human', 'category', 'review',
    ])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)

    print()
    print(f'wrote {len(queue)} queue items to {args.output}')
    if len(queue):
        for category in CATEGORIES:
            subset = queue[queue.category == category]
            print(f'  {category:9s} {len(subset):6d}  on {subset.image_path.nunique():5d} images')
        high = queue[(queue.category == 'new') & (queue.confidence >= 0.5)]
        print(f'\n{len(high)} high-confidence new ears (conf >= 0.5) -- review these first.')


def cmd_review(args: argparse.Namespace) -> None:
    import cv2

    queue = pd.read_csv(args.queue)
    if args.category:
        selection = queue[(queue.category == args.category) & (queue.review == 'pending')]
    else:
        selection = queue[queue.review == 'pending']
    selection = selection[selection.confidence >= args.min_conf]
    selection = selection.sort_values('confidence', ascending=False)

    if selection.empty:
        print('Nothing pending matches those filters.')
        return

    print(f'{len(selection)} items to review.  a=accept  r=reject  s=skip  u=undo  q=save and quit')
    human = human_boxes_by_image(args.csv)
    root = Path(args.data_root)
    history: List[int] = []
    order = list(selection.index)
    position = 0

    while position < len(order):
        idx = order[position]
        row = queue.loc[idx]
        image = cv2.imread(str(root / str(row.image_path)))
        if image is None:
            queue.loc[idx, 'review'] = 'unreadable'
            position += 1
            continue

        for x1, y1, x2, y2 in human.get(str(row.image_path), np.zeros((0, 4))):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        x, y = int(row.x1), int(row.y1)
        cv2.rectangle(image, (x, y), (x + int(row.w), y + int(row.h)), (0, 215, 255), 2)

        done = position + 1
        for i, text in enumerate([
            f'{done}/{len(order)}  {row.category}  conf {row.confidence:.2f}',
            'green = existing label   yellow = proposal',
            'a accept   r reject   s skip   u undo   q quit',
        ]):
            cv2.putText(image, text, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(image, text, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('relabel', image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('u') and history:
            position = history.pop()
            queue.loc[order[position], 'review'] = 'pending'
            continue
        decision = {ord('a'): 'accepted', ord('r'): 'rejected', ord('s'): 'skipped'}.get(key)
        if decision is None:
            continue
        queue.loc[idx, 'review'] = decision
        history.append(position)
        position += 1

    cv2.destroyAllWindows()
    queue.to_csv(args.queue, index=False)
    counts = queue.review.value_counts().to_dict()
    print(f'saved {args.queue}: {counts}')


def cmd_apply(args: argparse.Namespace) -> None:
    queue = pd.read_csv(args.queue)
    accepted = queue[(queue.review == 'accepted') & (queue.category != 'missed')]
    rejected_gt = queue[(queue.review == 'accepted') & (queue.category == 'missed')]
    # 'box_wrong' means the reviewer confirmed a real ear but the proposed
    # extent is wrong. Adding it would put a badly-localized positive into
    # training, which damages box regression more than the missing label costs
    # in recall, so it is recorded for a correction pass and never merged.
    box_wrong = queue[queue.review == 'box_wrong']

    master = pd.read_csv(args.csv)
    keep = master
    if args.drop_pose and 'annotation_source' in master.columns:
        before = len(master)
        keep = master[master['annotation_source'].isin(HUMAN_ANNOTATION_SOURCES)]
        print(f'dropped {before - len(keep)} pseudo-label rows')

    additions = pd.DataFrame({
        'image_path': accepted.image_path,
        'x1': accepted.x1.round().astype(int),
        'y1': accepted.y1.round().astype(int),
        'w': accepted.w.round().astype(int),
        'h': accepted.h.round().astype(int),
        'annotation_source': 'GT+REVIEW',
        'source': 'relabel',
        'confidence': accepted.confidence,
    })
    combined = pd.concat([keep, additions], ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f'{len(keep)} kept + {len(additions)} reviewed additions -> {args.output}')
    if len(box_wrong):
        path = Path(args.output).with_name(Path(args.output).stem + '_needs_box_fix.csv')
        box_wrong.to_csv(path, index=False)
        print(f'{len(box_wrong)} real ears with a wrong box were NOT added; '
              f'queued for correction in {path}')
    if len(rejected_gt):
        print(f'{len(rejected_gt)} existing labels were confirmed bad; '
              'they are NOT removed automatically -- inspect them before deleting.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    common = dict(csv='data/splits/master.csv', data_root='data/raw')

    p = sub.add_parser('propose', help='run the labeller and build a review queue')
    p.add_argument('--weights', required=True)
    p.add_argument('--csv', default=common['csv'])
    p.add_argument('--data-root', default=common['data_root'])
    p.add_argument('--output', default=DEFAULT_QUEUE)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='0')
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser('review', help='accept or reject queued proposals')
    p.add_argument('--queue', default=DEFAULT_QUEUE)
    p.add_argument('--csv', default=common['csv'])
    p.add_argument('--data-root', default=common['data_root'])
    p.add_argument('--category', choices=CATEGORIES)
    p.add_argument('--min-conf', type=float, default=0.0)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser('apply', help='merge accepted proposals into a new CSV')
    p.add_argument('--queue', default=DEFAULT_QUEUE)
    p.add_argument('--csv', default=common['csv'])
    p.add_argument('--output', default='data/splits/master_relabelled.csv')
    p.add_argument('--drop-pose', action='store_true', default=True)
    p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
