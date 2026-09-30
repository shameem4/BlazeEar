"""Build the set of images no model in this repo's history ever trained on.

Comparing checkpoints across the v2 work is harder than picking a validation
split, because the splits themselves changed. 85.3% of the current validation
images sit inside the ORIGINAL model's training set, so scoring it there
measures memorisation and flatters it.

Membership by filename is not enough either: the original split was per file,
and Roboflow re-exports one photo under several hashes, so a file held out of
training can still have a near-duplicate inside it. Identity here is the
content-hash photo index.

What comes out is small (279 images) but clean for every checkpoint.
"""
import argparse

import pandas as pd

from utils.config import HUMAN_ANNOTATION_SOURCES
from utils.photo_index import load_index

DEFAULT_TRAIN_SPLITS = ('data/splits/train.csv', 'data/splits/train_v2.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', default='data/splits/master_relabelled.csv')
    parser.add_argument('--train-splits', nargs='+', default=list(DEFAULT_TRAIN_SPLITS),
                        help='every training split any compared model used')
    parser.add_argument('--out', default='data/splits/common_heldout.csv')
    args = parser.parse_args()

    index = load_index()
    as_photos = lambda paths: {index.get(p, p) for p in paths}

    trained = set()
    for split in args.train_splits:
        paths = pd.read_csv(split).image_path
        trained |= as_photos(paths)
        print(f'{split}: {paths.nunique()} images')

    master = pd.read_csv(args.master)
    held_out = master[master.image_path.map(
        lambda p: index.get(p, p) not in trained)]
    held_out.to_csv(args.out, index=False)

    ears = held_out.annotation_source.isin(HUMAN_ANNOTATION_SOURCES).sum()
    print(f'\nheld out by every listed split: {held_out.image_path.nunique()} images, '
          f'{ears} human ears, {held_out.source.nunique()} sources -> {args.out}')
    if held_out.image_path.nunique() < 200:
        print('Warning: this set is small; treat differences of a few points '
              'as noise.')


if __name__ == '__main__':
    main()
