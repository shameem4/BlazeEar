"""
Build one reviewable batch from a relabelling queue.

Renders a crop per queue item -- the proposal in amber, existing labels in
green, with surrounding context -- plus a self-contained page ready to publish
as an artifact. Review happens in batches because a few hundred judgements in
one sitting is where attention goes, and a tired reviewer is a worse label
source than no reviewer.

    python make_review_batch.py --queue data/relabel/proposals.csv \
        --batch 2 --size 75 --out /tmp/batch2

The output directory holds index.html and crops/, which publish together.
Items already carrying a review decision are skipped, so re-running after a
round of review produces the next unreviewed batch.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from relabel import add_dedup_keys, human_boxes_by_image

TEMPLATE = Path(__file__).parent / 'utils' / 'review_template.html'

# Context multiple around the proposal, and the rendered crop width.
CONTEXT = 3.0
CROP_WIDTH = 360


def render_crops(rows: pd.DataFrame, data_root: Path, human, out_dir: Path,
                 inline: bool = False, quality: int = 72) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    for position, row in enumerate(rows.itertuples(index=True)):
        image = cv2.imread(str(data_root / row.image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        centre_x, centre_y = row.x1 + row.w / 2, row.y1 + row.h / 2
        half = max(row.w, row.h) * CONTEXT / 2
        x1, y1 = int(max(0, centre_x - half)), int(max(0, centre_y - half))
        x2, y2 = int(min(width, centre_x + half)), int(min(height, centre_y + half))
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        crop = image[y1:y2, x1:x2].copy()
        for hx1, hy1, hx2, hy2 in human.get(row.image_path, np.zeros((0, 4))):
            cv2.rectangle(crop, (int(hx1 - x1), int(hy1 - y1)),
                          (int(hx2 - x1), int(hy2 - y1)), (80, 220, 80), 2)
        cv2.rectangle(crop, (int(row.x1 - x1), int(row.y1 - y1)),
                      (int(row.x1 + row.w - x1), int(row.y1 + row.h - y1)), (30, 170, 240), 3)

        scaled_height = max(1, int(crop.shape[0] * CROP_WIDTH / crop.shape[1]))
        crop = cv2.resize(crop, (CROP_WIDTH, scaled_height))
        name = f'c{position:04d}.jpg'
        if inline:
            ok, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            name = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('ascii')
        else:
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])

        items.append({
            'id': int(row.Index), 'file': name, 'conf': round(float(row.confidence), 3),
            'image_path': row.image_path, 'x1': float(row.x1), 'y1': float(row.y1),
            'w': float(row.w), 'h': float(row.h),
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', default='data/relabel/proposals.csv')
    parser.add_argument('--csv', default='data/splits/master.csv')
    parser.add_argument('--data-root', default='data/raw')
    parser.add_argument('--out', required=True, help='Output directory for the batch')
    parser.add_argument('--batch', type=int, default=1, help='Batch number, shown in the page')
    parser.add_argument('--size', type=int, default=75, help='Items in this batch')
    parser.add_argument('--category', default='new', help="'new', 'conflict' or 'missed'")
    parser.add_argument('--min-conf', type=float, default=0.25)
    parser.add_argument('--sort-by', choices=['size', 'confidence'], default='size',
                        help='Size first by default: in the pilot it predicted whether a '
                             'proposal becomes a usable label far better than confidence '
                             '(accept rate 0.69 for the largest quartile against 0.17 for '
                             'the smallest, while "is it a real ear" barely moved).')
    parser.add_argument('--image-size', type=float, default=640.0)
    parser.add_argument('--inline-images', action='store_true',
                        help='Embed crops in the page as data URIs instead of separate '
                             'files. An artifact publishes at most 255 files, so a batch '
                             'larger than ~250 needs this. The page gets bigger but '
                             'loads in one request.')
    parser.add_argument('--quality', type=int, default=72, help='JPEG quality for crops')
    args = parser.parse_args()

    queue = pd.read_csv(args.queue)
    if 'dedup_key' not in queue.columns:
        queue = add_dedup_keys(queue)
    pending = queue[(queue.category == args.category)
                    & (queue.review == 'pending')
                    & (queue.confidence >= args.min_conf)]
    if args.sort_by == 'size':
        pending = pending.assign(
            _size=pending[['w', 'h']].max(axis=1) / args.image_size
        ).sort_values('_size', ascending=False).drop(columns='_size')
    else:
        pending = pending.sort_values('confidence', ascending=False)
    # One representative per ear. Roboflow re-exports the same photo under
    # several hashes, so without this the reviewer judges the same ear more
    # than once; the decision is copied to the copies at apply time.
    before = len(pending)
    if 'dedup_key' in pending.columns:
        pending = pending.drop_duplicates('dedup_key', keep='first')
    duplicates_hidden = before - len(pending)
    total_pending = len(pending)
    rows = pending.head(args.size)

    if rows.empty:
        print(f'Nothing pending in category {args.category!r}.')
        return

    out_dir = Path(args.out)
    items = render_crops(rows, Path(args.data_root), human_boxes_by_image(args.csv),
                         out_dir / 'crops', inline=args.inline_images,
                         quality=args.quality)

    remaining = total_pending - len(items)
    page = TEMPLATE.read_text(encoding='utf-8')
    page = (page
            .replace('__ITEMS__', json.dumps(items, separators=(',', ':')))
            .replace('__BATCH__', str(args.batch))
            .replace('__REMAINING__', str(max(0, remaining))))
    (out_dir / 'index.html').write_text(page, encoding='utf-8')

    if args.inline_images:
        size_mb = (out_dir / 'index.html').stat().st_size / 1e6
        print(f'batch {args.batch}: {len(items)} items inlined, page {size_mb:.2f} MB -> {out_dir}')
    else:
        size_mb = sum(f.stat().st_size for f in (out_dir / 'crops').glob('*.jpg')) / 1e6
        print(f'batch {args.batch}: {len(items)} items, {size_mb:.2f} MB of crops -> {out_dir}')
    print(f'{remaining} distinct ears still pending in category {args.category!r}')
    if duplicates_hidden:
        print(f'{duplicates_hidden} duplicate copies hidden; their decisions are '
              f'copied from the representative at apply time')
    if not args.inline_images:
        print('files:', ' '.join(sorted(p.name for p in (out_dir / "crops").glob("*.jpg"))[:3]), '...')


if __name__ == '__main__':
    main()
