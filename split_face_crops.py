"""Split a crop dataset along an existing image-level split.

A crop inherits the side its origin photo is already on, so the crop
validation set covers the same photos as the full-frame one and the two mAPs
answer the same question. Splitting the crops afresh would let two crops of
one photo land on opposite sides.
"""
import argparse

import pandas as pd

from utils.config import HUMAN_ANNOTATION_SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crops', default='data/splits/face_crops.csv')
    parser.add_argument('--train-reference', default='data/splits/train_v2.csv')
    parser.add_argument('--val-reference', default='data/splits/val_v2.csv')
    parser.add_argument('--out-prefix', default='data/splits/crops')
    args = parser.parse_args()

    crops = pd.read_csv(args.crops)
    train_images = set(pd.read_csv(args.train_reference).image_path)
    val_images = set(pd.read_csv(args.val_reference).image_path)

    def side_of(path):
        if path in train_images:
            return 'train'
        return 'val' if path in val_images else None

    side = crops.origin_image.map(side_of)
    for name in ('train', 'val'):
        part = crops[side == name]
        out = f'{args.out_prefix}_{name}.csv'
        part.to_csv(out, index=False)
        ears = part.annotation_source.isin(HUMAN_ANNOTATION_SOURCES).sum()
        empty = part.image_path.nunique() - part[
            part.annotation_source.isin(HUMAN_ANNOTATION_SOURCES)
        ].image_path.nunique()
        print(f'{name:5s} {len(part):6d} rows  {part.image_path.nunique():6d} crops  '
              f'{ears:6d} ears  {empty:5d} ear-free crops  '
              f'{part.origin_image.nunique():6d} photos -> {out}')

    unplaced = int(side.isna().sum())
    straddling = len(set(crops[side == 'train'].origin_image)
                     & set(crops[side == 'val'].origin_image))
    print(f'rows whose origin photo is in neither reference split: {unplaced}')
    print(f'photos appearing on both sides: {straddling}')
    if straddling:
        raise SystemExit('a photo straddles the split; the two mAPs are not comparable')


if __name__ == '__main__':
    main()
