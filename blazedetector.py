import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from blazebase import BlazeBase
from utils.iou import intersect_torch, jaccard_torch, overlap_similarity_torch

class BlazeDetector(BlazeBase):
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    
    Training methodology adapted from vincent1bt/blazeface-tensorflow.
    """
    
    # Type annotations for class attributes
    x_scale: float
    y_scale: float
    w_scale: float
    h_scale: float
    num_keypoints: int
    num_anchors: int
    num_coords: int
    num_classes: int
    anchors: torch.Tensor
    score_clipping_thresh: float
    min_score_thresh: float
    min_suppression_threshold: float
    
    # Training mode flag
    _training_mode: bool = False


    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1] (MediaPipe convention)."""
        return x.float() / 127.5 - 1.0
    
    # =========================================================================
    # Training Methods (following vincent1bt/blazeface-tensorflow)
    # =========================================================================
    
    def get_training_outputs(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns raw outputs for training.
        
        Unlike predict_on_batch which applies post-processing and NMS,
        this returns the raw regression and classification outputs
        needed for loss computation.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) with values in [0, 255]
            
        Returns:
            raw_boxes: (B, 896, num_coords) - raw box regression outputs
            raw_scores: (B, 896, 1) - raw classification logits (before sigmoid)
        """
        x = self._preprocess(x)
        raw_boxes, raw_scores = self.__call__(x)
        return raw_boxes, raw_scores
    def compute_training_metrics(
        self,
        pred_boxes: torch.Tensor,
        true_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        true_classes: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute training metrics for monitoring.
        
        Args:
            pred_boxes: (B, 896, 4) decoded predicted boxes
            true_boxes: (B, 896, 4) ground truth boxes
            pred_scores: (B, 896) predicted scores (after sigmoid)
            true_classes: (B, 896) ground truth classes (0 or 1)
            
        Returns:
            Dictionary with metrics:
            - positive_accuracy: accuracy on positive samples
            - background_accuracy: accuracy on background samples
            - mean_iou: mean IoU on positive samples
        """
        # Create masks
        positive_mask = true_classes > 0.5
        background_mask = ~positive_mask
        
        # Accuracy metrics
        pred_binary = (pred_scores > 0.5).float()
        
        positive_correct = (pred_binary[positive_mask] == 1.0).float()
        positive_accuracy = positive_correct.mean().item() if positive_mask.any() else 0.0
        
        background_correct = (pred_binary[background_mask] == 0.0).float()
        background_accuracy = background_correct.mean().item() if background_mask.any() else 1.0
        
        # IoU on positive samples (simplified)
        mean_iou = 0.0
        if positive_mask.any():
            pos_pred = pred_boxes[positive_mask]  # (N, 4)
            pos_true = true_boxes[positive_mask]  # (N, 4)
            
            # Compute IoU for each pair
            inter_x1 = torch.max(pos_pred[:, 0], pos_true[:, 0])
            inter_y1 = torch.max(pos_pred[:, 1], pos_true[:, 1])
            inter_x2 = torch.min(pos_pred[:, 2], pos_true[:, 2])
            inter_y2 = torch.min(pos_pred[:, 3], pos_true[:, 3])
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
            
            pred_area = (pos_pred[:, 2] - pos_pred[:, 0]) * (pos_pred[:, 3] - pos_pred[:, 1])
            true_area = (pos_true[:, 2] - pos_true[:, 0]) * (pos_true[:, 3] - pos_true[:, 1])
            
            union_area = pred_area + true_area - inter_area
            iou = inter_area / (union_area + 1e-6)
            mean_iou = iou.mean().item()
        
        return {
            'positive_accuracy': positive_accuracy,
            'background_accuracy': background_accuracy,
            'mean_iou': mean_iou
        }

    def predict_on_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def resize_pad(
        self,
        img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        """ resize and pad images to be input to the detectors

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio.

        Returns:
            img1: 256x256
            img2: 128x128
            scale: scale factor between original image and 256x256 image
            pad: pixels of padding in the original image
        """

        size0 = img.shape
        if size0[0]>=size0[1]:
            h1 = 256
            w1 = 256 * size0[1] // size0[0]
            padh = 0
            padw = 256 - w1
            scale = size0[1] / w1
        else:
            h1 = 256 * size0[0] // size0[1]
            w1 = 256
            padh = 256 - h1
            padw = 0
            scale = size0[0] / h1
        padh1 = padh//2
        padh2 = padh//2 + padh%2
        padw1 = padw//2
        padw2 = padw//2 + padw%2
        img1 = cv2.resize(img, (w1,h1))
        img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
        pad = (int(padh1 * scale), int(padw1 * scale))
        img2 = cv2.resize(img1, (128,128))
        return img1, img2, scale, pad


    def denormalize_detections(
        self,
        detections: torch.Tensor,
        scale: float,
        pad: tuple[int, int]
    ) -> torch.Tensor:
        """Map detection coordinates (boxes + keypoints) back to image space.

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio. This function maps the
        normalized coordinates back to the original image coordinates.

        Inputs:
            detections: nx5 tensor. n is the number of detections.
                First 4 values are box coordinates [ymin, xmin, ymax, xmax],
                5th value is confidence score.
            scale: scalar that was used to resize the image
            pad: padding in the x and y dimensions

        """
        if detections.numel() == 0:
            return detections

        detections = detections.clone()

        detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
        detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
        detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
        detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

        # Decode any keypoint coordinates (x, y pairs) if present (vectorized)
        num_coords = detections.shape[1]
        keypoint_coords = max(0, num_coords - 5)
        num_keypoints = keypoint_coords // 2
        if num_keypoints > 0:
            kp_end = 4 + num_keypoints * 2
            # x coordinates (even indices starting at 4): apply x transform
            detections[:, 4:kp_end:2] = detections[:, 4:kp_end:2] * scale * 256 - pad[1]
            # y coordinates (odd indices starting at 5): apply y transform
            detections[:, 5:kp_end:2] = detections[:, 5:kp_end:2] * scale * 256 - pad[0]

        return detections

    def predict_on_batch(self, x: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 5).

        Each detection is a PyTorch tensor consisting of 5 numbers:
            - ymin, xmin, ymax, xmax
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == self.y_scale
        assert x.shape[3] == self.x_scale

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, int(self.num_coords)+1))
            filtered_detections.append(faces)

        return filtered_detections


    def _tensors_to_detections(
        self,
        raw_box_tensor: torch.Tensor,
        raw_score_tensor: torch.Tensor,
        anchors: torch.Tensor
    ) -> list[torch.Tensor]:
        """The output of the neural network is a tensor of shape (b, 896, 4)
        containing the bounding box regressor predictions, as well as a tensor 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 5) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        # Ensure coordinates are ordered (allow values outside [0, 1] for padding offsets)
        ymin = torch.minimum(detection_boxes[..., 0], detection_boxes[..., 2])
        ymax = torch.maximum(detection_boxes[..., 0], detection_boxes[..., 2])
        xmin = torch.minimum(detection_boxes[..., 1], detection_boxes[..., 3])
        xmax = torch.maximum(detection_boxes[..., 1], detection_boxes[..., 3])

        detection_boxes = detection_boxes.clone()
        detection_boxes[..., 0] = ymin
        detection_boxes[..., 1] = xmin
        detection_boxes[..., 2] = ymax
        detection_boxes[..., 3] = xmax

        widths = xmax - xmin
        heights = ymax - ymin
        valid_box_mask = (widths > 1e-5) & (heights > 1e-5)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        finite_mask = torch.isfinite(detection_scores)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = (detection_scores >= self.min_score_thresh) & valid_box_mask & finite_mask

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(
        self,
        raw_boxes: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.

        Following vincent1bt/blazeface-tensorflow decoding (no anchor w/h scaling):
        - x_center = anchor_x + (pred_x / scale)
        - y_center = anchor_y + (pred_y / scale)
        - w = pred_w / scale
        - h = pred_h / scale

        Output format: [ymin, xmin, ymax, xmax] (MediaPipe box convention)
        """
        boxes = torch.zeros_like(raw_boxes)

        # Decode center and size (raw layout = [dx, dy, w, h])
        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        # Decode keypoint coordinates (vectorized)
        # MediaPipe stores x,y pairs after the box coords
        if hasattr(self, "num_keypoints") and self.num_keypoints > 0:
            kp_end = 4 + self.num_keypoints * 2
            # x coordinates are at indices 4, 6, 8, ... (even offsets from 4)
            # y coordinates are at indices 5, 7, 9, ... (odd offsets from 4)
            boxes[..., 4:kp_end:2] = raw_boxes[..., 4:kp_end:2] / self.x_scale * anchors[:, 2:3] + anchors[:, 0:1]
            boxes[..., 5:kp_end:2] = raw_boxes[..., 5:kp_end:2] / self.y_scale * anchors[:, 3:4] + anchors[:, 1:2]

        return boxes

    def _weighted_non_max_suppression(self, detections: torch.Tensor) -> list[torch.Tensor]:
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 5).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, self.num_coords], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = self.overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            ious = torch.nan_to_num(ious, nan=0.0, posinf=0.0, neginf=0.0)
            mask = ious > self.min_suppression_threshold
            if not torch.any(mask):
                mask[0] = True
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


    # IOU functions now use utils.iou module

    def intersect(self, box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        """Compute intersection area between two sets of boxes."""
        return intersect_torch(box_a, box_b)

    def jaccard(self, box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        """Compute Jaccard overlap (IoU) between two sets of boxes."""
        return jaccard_torch(box_a, box_b)

    def overlap_similarity(self, box: torch.Tensor, other_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between a single box and a set of other boxes."""
        return overlap_similarity_torch(box, other_boxes)
