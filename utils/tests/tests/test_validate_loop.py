"""
The validation loop must actually consume the whole split.

A mis-indented edit once left the loop body assigning variables and nothing
else, so everything after it ran once, on the final batch. With 2017 images at
batch size 32 that is a single image, and validation reported metrics for it
while looking entirely normal: no error, plausible-looking numbers, mAP 0.
Counting images is the cheapest guard against that whole class of bug.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from blazeear import BlazeEar
from train_blazeear import BlazeEarTrainer


class _FakeSet(Dataset):
    """Images with one box each, sized so the last batch is a partial one."""

    def __init__(self, n=70):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        targets = torch.zeros(896, 5)
        targets[i % 896, 0] = 1.0
        targets[i % 896, 1:] = torch.tensor([0.3, 0.3, 0.5, 0.45])
        return {
            'image': torch.randint(0, 255, (3, 128, 128)).float(),
            'sample_index': i,
            'anchor_targets': targets,
            'anchor_ignore': torch.zeros(896, dtype=torch.bool),
            'gt_boxes': torch.tensor([[0.3, 0.3, 0.5, 0.45]]),
        }


def _loader(batch_size):
    from dataloader import collate_detector_fn
    return DataLoader(_FakeSet(), batch_size=batch_size, shuffle=False,
                      collate_fn=collate_detector_fn)


def _trainer(tmp_path, batch_size=32):
    loader = _loader(batch_size)
    return BlazeEarTrainer(
        model=BlazeEar(), train_loader=loader, val_loader=loader, device='cpu',
        checkpoint_dir=str(tmp_path / 'ck'), log_dir=str(tmp_path / 'lg'),
        use_amp=False,
    ), loader


def test_validate_sees_every_image(tmp_path):
    trainer, _ = _trainer(tmp_path)
    result = trainer.validate(compute_map=True)
    assert result['num_images'] == 70, (
        f"validation scored {result['num_images']} of 70 images"
    )


def test_partial_final_batch_is_not_the_only_batch(tmp_path):
    """70 images at batch 32 makes a final batch of 6; that must not be all."""
    trainer, _ = _trainer(tmp_path, batch_size=32)
    result = trainer.validate(compute_map=True)
    assert result['num_images'] > 6


def test_ground_truth_count_matches_the_split(tmp_path):
    trainer, _ = _trainer(tmp_path)
    result = trainer.validate(compute_map=True)
    assert result['num_ground_truth'] == 70


def test_max_batches_truncates_deliberately(tmp_path):
    trainer, _ = _trainer(tmp_path, batch_size=10)
    result = trainer.validate(compute_map=True, max_batches=3)
    assert result['num_images'] == 30
