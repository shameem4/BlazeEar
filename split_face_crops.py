"""Split the crop dataset along the existing image-level split.

A crop inherits the side its origin photo is already on, so the crop val set
covers the same photos as the full-frame val set and the two numbers are
comparable.
"""
import sys
import pandas as pd

tag = sys.argv[1]
crops = pd.read_csv(f'data/splits/face_crops_{tag}.csv')
train_images = set(pd.read_csv('data/splits/train_v2.csv').image_path)
val_images = set(pd.read_csv('data/splits/val_v2.csv').image_path)

side = crops.origin_image.map(
    lambda p: 'train' if p in train_images else ('val' if p in val_images else None))
unplaced = side.isna().sum()
for name in ('train', 'val'):
    part = crops[side == name]
    out = f'data/splits/crops_{tag}_{name}.csv'
    part.to_csv(out, index=False)
    human = part.annotation_source.isin(('GT', 'GT+EAR', 'GT+REVIEW')).sum()
    print(f'{name:5s} {len(part):6d} rows  {part.image_path.nunique():6d} crops  '
          f'{human:6d} ears  {part.origin_image.nunique():6d} photos -> {out}')
print(f'unplaced rows: {unplaced}')
overlap = (set(crops[side == "train"].origin_image)
           & set(crops[side == "val"].origin_image))
print(f'photos on both sides: {len(overlap)}')
