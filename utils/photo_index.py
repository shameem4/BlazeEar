"""
Content-based photo identity.

Filenames are not a reliable identity for this dataset. Roboflow exports
re-encode the same scene under different hashes, and the same photo was also
ingested under different sequence numbers -- `1-626-`, `1-1161-`, `1-232-` and
`1-1046-` are one photo, so a filename-derived key groups none of them.

A perceptual hash groups them by what the image actually shows. This matters in
two places: a review queue must not ask about one ear several times, and a
train/val split must not put the same scene on both sides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np

DEFAULT_INDEX = Path('data/relabel/photo_index.json')

# Difference-hash side length. 16 gives a 256-bit hash; two images are treated
# as the same photo when they differ in at most HAMMING_THRESHOLD bits, which
# tolerates Roboflow's photometric alterations without merging distinct scenes.
HASH_SIDE = 16
HAMMING_THRESHOLD = 30


def difference_hash(path: Path, side: int = HASH_SIDE) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (side + 1, side))
    return np.packbits(resized[:, 1:] > resized[:, :-1])


def build_index(image_paths: Iterable[str], data_root: Path,
                threshold: int = HAMMING_THRESHOLD) -> Dict[str, str]:
    """Map each image path to a photo id shared by its near-duplicates.

    Hamming distance between every pair is computed as a matrix product rather
    than a Python loop: for bit vectors, `popcount(a ^ b)` equals
    `sum(a) + sum(b) - 2 * (a . b)`. Blocked so the full 13k x 13k product is
    never held at once. Groups are then closed with union-find, so a chain of
    near-matches collapses to one photo id.
    """
    paths: list[str] = []
    bits: list[np.ndarray] = []
    for image_path in image_paths:
        digest = difference_hash(data_root / image_path)
        if digest is not None:
            paths.append(image_path)
            bits.append(np.unpackbits(digest))

    if not paths:
        return {}

    matrix = np.asarray(bits, dtype=np.float32)
    ones = matrix.sum(axis=1)
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    block = 512
    for start in range(0, len(paths), block):
        stop = min(start + block, len(paths))
        products = matrix[start:stop] @ matrix.T
        distances = ones[start:stop, None] + ones[None, :] - 2.0 * products
        close = np.argwhere(distances <= threshold)
        for row, column in close:
            i = start + int(row)
            j = int(column)
            if i < j:
                union(i, j)

    roots = {}
    photo_of: Dict[str, str] = {}
    for index, image_path in enumerate(paths):
        root = find(index)
        if root not in roots:
            roots[root] = f'photo{len(roots):06d}'
        photo_of[image_path] = roots[root]
    return photo_of


def load_index(path: Path = DEFAULT_INDEX) -> Dict[str, str]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def save_index(photo_of: Dict[str, str], path: Path = DEFAULT_INDEX) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(photo_of))
