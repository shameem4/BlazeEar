import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blazebase import BlazeBlock_WT
from blazedetector import BlazeDetector




class BlazeFace(BlazeDetector):
    """The BlazeFace face detection model from MediaPipe.
    
    The version from MediaPipe is simpler than the one in the paper; 
    it does not use the "double" BlazeBlocks.

    Architecture notes:
    - Input: 128x128 (front) or 256x256 (back)
    - Output: 896 anchors = 512 (16x16 grid × 2) + 384 (8x8 grid × 6)
    - Classifier: outputs raw logits, sigmoid applied during post-processing
    - Regressor: outputs 16 coords per anchor (4 box + 6×2 keypoints)
    
    For training (following vincent1bt/blazeface-tensorflow):
    - forward() returns [raw_boxes, raw_scores] for loss computation
    - raw_scores are logits (no sigmoid) - apply sigmoid for loss/inference
    - Use get_training_outputs() from BlazeDetector for preprocessed input

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/

    """
    def __init__(self):
        super(BlazeFace, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16  # Box + keypoints (MediaPipe layout)
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 6  # Keep keypoint outputs (untrained without labels)

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        self.detection2roi_method = 'box'
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
        self.kp1 = 1
        self.kp2 = 0
        self.theta0 = 0.
        self.dscale = 1.5
        self.dy = 0.

        self._define_layers()

    def _define_layers(self):
        # Front model architecture (128x128 input)
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock_WT(24, 24),
            BlazeBlock_WT(24, 28),
            BlazeBlock_WT(28, 32, stride=2),
            BlazeBlock_WT(32, 36),
            BlazeBlock_WT(36, 42),
            BlazeBlock_WT(42, 48, stride=2),
            BlazeBlock_WT(48, 56),
            BlazeBlock_WT(56, 64),
            BlazeBlock_WT(64, 72),
            BlazeBlock_WT(72, 80),
            BlazeBlock_WT(80, 88),
        )
    
        self.backbone2 = nn.Sequential(
            BlazeBlock_WT(88, 96, stride=2),
            BlazeBlock_WT(96, 96),
            BlazeBlock_WT(96, 96),
            BlazeBlock_WT(96, 96),
            BlazeBlock_WT(96, 96),
        )

        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        # Regressor outputs split into box (4 coords) and keypoints (remaining coords)
        # 8x8 grid: 2 anchors × (4 box + 12 kp) = (8 + 24) channels
        # 16x16 grid: 6 anchors × (4 box + 12 kp) = (24 + 72) channels
        self.regressor_8_box = nn.Conv2d(88, 8, 1, bias=True)
        self.regressor_8_kp = nn.Conv2d(88, 24, 1, bias=True)
        self.regressor_16_box = nn.Conv2d(96, 24, 1, bias=True)
        self.regressor_16_kp = nn.Conv2d(96, 72, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone1(x)           # (b, 88, 16, 16)
        h = self.backbone2(x)           # (b, 96, 8, 8)
        
        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        box_r1 = self.regressor_8_box(x)        # (b, 8, 16, 16)
        box_r1 = box_r1.permute(0, 2, 3, 1).reshape(b, -1, 4)  # (b, 512, 4)
        kp_r1 = self.regressor_8_kp(x)          # (b, 24, 16, 16)
        kp_r1 = kp_r1.permute(0, 2, 3, 1).reshape(b, -1, 12)   # (b, 512, 12)
        r1 = torch.cat((box_r1, kp_r1), dim=-1)  # (b, 512, 16)

        box_r2 = self.regressor_16_box(h)        # (b, 24, 8, 8)
        box_r2 = box_r2.permute(0, 2, 3, 1).reshape(b, -1, 4)  # (b, 384, 4)
        kp_r2 = self.regressor_16_kp(h)          # (b, 72, 8, 8)
        kp_r2 = kp_r2.permute(0, 2, 3, 1).reshape(b, -1, 12)   # (b, 384, 12)
        r2 = torch.cat((box_r2, kp_r2), dim=-1)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]
    
    def freeze_keypoint_regressors(self) -> None:
        """Disable gradients for keypoint regressors so they stay fixed."""
        for layer in (self.regressor_8_kp, self.regressor_16_kp):
            layer.requires_grad_(False)
    
    def calculate_scale(
        self,
        min_scale: float,
        max_scale: float,
        stride_index: int,
        num_strides: int
    ) -> float:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


    def generate_anchors(self, options: dict) -> None:
        strides_size = len(options["strides"])
        assert options["num_layers"] == strides_size

        anchors = []
        layer_id = 0
        while layer_id < strides_size:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            # For same strides, we merge the anchors in the same order.
            last_same_stride_layer = layer_id
            while (last_same_stride_layer < strides_size) and \
                  (options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
                scale = self.calculate_scale(options["min_scale"],
                                        options["max_scale"],
                                        last_same_stride_layer,
                                        strides_size)

                if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                    # For first layer, it can be specified to use predefined anchors.
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)
                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)                
                else:
                    for aspect_ratio in options["aspect_ratios"]:
                        aspect_ratios.append(aspect_ratio)
                        scales.append(scale)

                    if options["interpolated_scale_aspect_ratio"] > 0.0:
                        scale_next = 1.0 if last_same_stride_layer == strides_size - 1 \
                                         else self.calculate_scale(options["min_scale"],
                                                              options["max_scale"],
                                                              last_same_stride_layer + 1,
                                                              strides_size)
                        scales.append(np.sqrt(scale * scale_next))
                        aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

                last_same_stride_layer += 1

            for i in range(len(aspect_ratios)):
                ratio_sqrts = np.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)            
                
            stride = options["strides"][layer_id]
            feature_map_height = int(np.ceil(options["input_size_height"] / stride))
            feature_map_width = int(np.ceil(options["input_size_width"] / stride))

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + options["anchor_offset_x"]) / feature_map_width
                        y_center = (y + options["anchor_offset_y"]) / feature_map_height

                        # Store [x_center, y_center, width, height]
                        # With fixed_anchor_size=True, w=h=1.0 (predictions scaled by input size only)
                        # With fixed_anchor_size=False, w/h vary per anchor (predictions scaled by anchor size)
                        if options.get("fixed_anchor_size", True):
                            anchor_w = 1.0
                            anchor_h = 1.0
                        else:
                            anchor_w = anchor_width[anchor_id]
                            anchor_h = anchor_height[anchor_id]
                        anchors.append([x_center, y_center, anchor_w, anchor_h])

            layer_id = last_same_stride_layer

        self.anchors = torch.tensor(anchors).to(self._device())
        assert self.anchors.ndimension() == 2
        assert self.anchors.shape[0] == self.num_anchors
        assert self.anchors.shape[1] == 4  # [x, y, w, h]
        assert len(self.anchors) == 896

    def process(self, frame: np.ndarray) -> torch.Tensor:
        img1, img2, scale, pad = self.resize_pad(frame)
        normalized_face_detections = self.predict_on_image(img2)
        face_detections = self.denormalize_detections(normalized_face_detections, scale, pad)
        # xc, yc, scale, theta = self.detection2roi(face_detections.cpu())

        return face_detections
