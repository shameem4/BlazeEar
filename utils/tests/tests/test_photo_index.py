"""
Content-based photo identity.

Filenames do not identify a photo in this dataset: the same scene was ingested
under different sequence numbers, so `1-626-`, `1-1161-`, `1-232-` and
`1-1046-` are one photo that no filename rule groups. Content hashing finds 559
duplicate files where the filename key finds 231.
"""
import cv2
import numpy as np
import pytest

from utils.photo_index import build_index, difference_hash, load_index, save_index


@pytest.fixture
def photos(tmp_path):
    rng = np.random.default_rng(0)
    scene_a = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    scene_b = (rng.random((120, 160, 3)) * 255).astype(np.uint8)

    # Same scene under unrelated names, one of them brightened the way a
    # Roboflow re-export alters it.
    cv2.imwrite(str(tmp_path / '1-626-_jpg.rf.aaa.jpg'), scene_a)
    cv2.imwrite(str(tmp_path / '1-1161-_jpg.rf.bbb.jpg'), scene_a)
    cv2.imwrite(str(tmp_path / '1-232-_jpg.rf.ccc.jpg'),
                np.clip(scene_a.astype(int) + 12, 0, 255).astype(np.uint8))
    cv2.imwrite(str(tmp_path / 'other_jpg.rf.ddd.jpg'), scene_b)
    return tmp_path


class TestDifferenceHash:
    def test_missing_file_returns_none(self, tmp_path):
        assert difference_hash(tmp_path / 'nope.jpg') is None

    def test_hash_is_stable(self, photos):
        a = difference_hash(photos / '1-626-_jpg.rf.aaa.jpg')
        b = difference_hash(photos / '1-626-_jpg.rf.aaa.jpg')
        assert np.array_equal(a, b)


class TestBuildIndex:
    def test_same_scene_under_different_names_groups(self, photos):
        names = ['1-626-_jpg.rf.aaa.jpg', '1-1161-_jpg.rf.bbb.jpg']
        index = build_index(names, photos)
        assert index[names[0]] == index[names[1]]

    def test_photometric_alteration_still_groups(self, photos):
        names = ['1-626-_jpg.rf.aaa.jpg', '1-232-_jpg.rf.ccc.jpg']
        index = build_index(names, photos)
        assert index[names[0]] == index[names[1]]

    def test_different_scenes_stay_apart(self, photos):
        names = ['1-626-_jpg.rf.aaa.jpg', 'other_jpg.rf.ddd.jpg']
        index = build_index(names, photos)
        assert index[names[0]] != index[names[1]]

    def test_grouping_is_transitive(self, photos):
        names = ['1-626-_jpg.rf.aaa.jpg', '1-1161-_jpg.rf.bbb.jpg',
                 '1-232-_jpg.rf.ccc.jpg']
        index = build_index(names, photos)
        assert len(set(index[n] for n in names)) == 1

    def test_unreadable_files_are_skipped(self, photos):
        index = build_index(['1-626-_jpg.rf.aaa.jpg', 'missing.jpg'], photos)
        assert 'missing.jpg' not in index
        assert len(index) == 1

    def test_empty_input(self, photos):
        assert build_index([], photos) == {}


class TestPersistence:
    def test_round_trip(self, photos, tmp_path):
        index = build_index(['1-626-_jpg.rf.aaa.jpg'], photos)
        path = tmp_path / 'idx.json'
        save_index(index, path)
        assert load_index(path) == index

    def test_missing_index_is_empty_not_an_error(self, tmp_path):
        assert load_index(tmp_path / 'absent.json') == {}
