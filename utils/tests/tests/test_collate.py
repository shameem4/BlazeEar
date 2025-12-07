import torch

from dataloader import collate_detector_fn


def _make_sample(num_boxes: int) -> dict:
    image = torch.zeros(3, 128, 128)
    anchor_targets = torch.zeros(896, 5)
    small = torch.zeros(16, 16, 5)
    big = torch.zeros(8, 8, 5)
    gt_boxes = torch.zeros((num_boxes, 4), dtype=torch.float32)
    for idx in range(num_boxes):
        gt_boxes[idx] = torch.tensor([0.1 * idx, 0.1 * idx, 0.2 + 0.1 * idx, 0.2 + 0.1 * idx])
    return {
        "image": image,
        "anchor_targets": anchor_targets,
        "small_anchors": small,
        "big_anchors": big,
        "gt_boxes": gt_boxes,
    }


def test_collate_pads_gt_boxes_to_max_count():
    batch = collate_detector_fn([_make_sample(2), _make_sample(0)])
    assert batch["gt_boxes"].shape == (2, 2, 4)
    assert torch.allclose(batch["gt_boxes"][1], torch.zeros(2, 4))
    assert torch.equal(batch["gt_box_counts"], torch.tensor([2, 0]))


def test_collate_handles_all_zero_gt_boxes():
    batch = collate_detector_fn([_make_sample(0), _make_sample(0)])
    assert batch["gt_boxes"].shape == (2, 1, 4)
    assert torch.allclose(batch["gt_boxes"], torch.zeros(2, 1, 4))
    assert torch.equal(batch["gt_box_counts"], torch.tensor([0, 0]))
