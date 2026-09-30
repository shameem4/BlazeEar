"""
Configuration constants for BlazeEar detection.
"""

# Anchor configuration
SMALL_GRID_SIZE = 16
BIG_GRID_SIZE = 8
SMALL_ANCHORS_PER_CELL = 2
BIG_ANCHORS_PER_CELL = 6
TOTAL_ANCHORS = 896  # 16*16*2 + 8*8*6

# Default paths
DEFAULT_WEIGHTS_PATH = "model_weights/blazeface.pth"
DEFAULT_DATA_ROOT = "data/raw/"
DEFAULT_TRAIN_CSV = "data/splits/train.csv"
DEFAULT_VAL_CSV = "data/splits/val.csv"
DEFAULT_CHECKPOINT_DIR = "runs/checkpoints"
DEFAULT_BEST_CHECKPOINT = "runs/checkpoints/BlazeEar_best.pth"
DEFAULT_LOG_DIR = "runs/logs"

# Training hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
DEFAULT_SAVE_EVERY = 10

# Model parameters
DEFAULT_INPUT_SIZE = 128

# Detection post-processing. One value each, shared by every path that turns
# raw anchors into detections. These were previously spread across five
# implementations at three different IoU thresholds (0.3 in the model, 0.35
# forced by utils/model_utils, 0.5 in the trainer's evaluation), which meant
# the reported mAP measured post-processing that nothing deployed used.
DETECTION_SCORE_THRESHOLD = 0.70
NMS_IOU_THRESHOLD = 0.3
MAX_DETECTIONS = 100

# Debug/inference defaults
DEFAULT_DEBUG_WEIGHTS = DEFAULT_BEST_CHECKPOINT
DEFAULT_SECONDARY_WEIGHTS = DEFAULT_WEIGHTS_PATH
DEFAULT_COMPARE_THRESHOLD = 0.70
DEFAULT_COMPARE_LABEL = "Mediapipe"
DEFAULT_DETECTOR_THRESHOLD_DEBUG = 0.70
DEFAULT_SCREENSHOT_OUTPUT = "runs/logs/debug_images"
DEFAULT_SCREENSHOT_COUNT = 10
DEFAULT_SCREENSHOT_MIN_FACES = 2
DEFAULT_SCREENSHOT_CANDIDATES = 20
DEFAULT_EVAL_MAX_IMAGES = 500
DEFAULT_EVAL_SCORE_THRESHOLD = 0.5
DEFAULT_EVAL_IOU_THRESHOLD = 0.5

# Annotation provenance (data_prep.py writes these into annotation_source)
# Human-verified boxes; everything else in the CSV is machine-generated.
HUMAN_ANNOTATION_SOURCES = ("GT", "GT+EAR", "GT+REVIEW")

# Boxes a reviewer confirmed hold a real ear that is too blurred, small or
# occluded to learn from. They are neither positives nor negatives: training
# excludes their anchors from both the positive set and hard negative mining,
# and evaluation neither rewards nor punishes a detection there. Including them
# as labels teaches an unlearnable target; excluding them entirely turns a real
# ear into a hard negative, which is the failure this dataset already had.
IGNORE_ANNOTATION_SOURCE = "IGNORE"

# A row that exists only so its image reaches the dataloader, carrying no box
# and no ignore region: a pure background image the model should learn to stay
# silent on. Face crops that contain no ear are the case this was added for --
# without them the crop model never sees a face whose ears are hidden, yet at
# inference roughly a third of the crops it is handed are exactly that.
NEGATIVE_ANNOTATION_SOURCE = "NEGATIVE"

# Two-stage pipeline: MediaPipe BlazeFace on the full frame, then the ear model
# on a square crop around each face. The dataset builder and the evaluator must
# agree on these, or evaluation crops differently than training did and the
# pipeline is scored on inputs it never saw.
FACE_CROP_EXPAND = 1.5        # crop side, in multiples of the face box
# Face score floor. This sets the recall ceiling, and is the only real lever
# on it: an ear whose face is missed never reaches the second stage. Swept on
# the validation split, with no retraining of the crop model:
#
#   threshold   no face   mAP@0.5   mAP@[.5:.95]   det IoU
#      0.30       4.1%     0.6336      0.2783       0.7276
#      0.25       2.8%     0.6381      0.2786       0.7259
#      0.20       1.3%     0.6447      0.2804       0.7236
#      0.15       0.4%     0.6476      0.2763       0.7213
#
# 0.20 takes the ceiling from 4.1% to 1.3% and has the best mAP@[.5:.95].
# 0.15 buys a little more mAP@0.5 but loses localisation and costs more crops
# per frame, so the gain there is weaker detections, not better ones.
FACE_CROP_THRESHOLD = 0.2
FACE_CROP_MAX_FACES = 8

# Duplicate suppression (near-duplicate box filtering)
NEAR_CENTER_DISTANCE_FRAC = 0.55
NEAR_MIN_AREA_RATIO = 0.35
NEAR_MIN_COVERAGE = 0.8

# Geometric post-filters (false positive rejection)
#
# OFF by default. These were compensating for a detector that flooded the frame
# with false positives, and they paid for it in recall: measured against 15630
# geometry-sane human boxes in master.csv, the previous bounds
# (aspect 0.35-1.4, size_frac 0.03-0.55) rejected 25.86% of REAL ears --
# 18.92% from the lower aspect bound alone, because ears are tall (median
# aspect 0.50, 0.5th percentile 0.141).
#
# The bounds below are the 0.5/99.5 percentiles of the real distribution, so
# enabling the filter costs about 1% of true ears rather than a quarter. They
# are correspondingly loose, which is the honest conclusion: there is no
# geometric rule that separates this detector's false positives from real ears.
# Fix precision in the model, not here.
#
# Only BlazeEar.process reads this. The exported ONNX graph cannot express the
# filter and the browser demo does not implement it, so turning it on makes
# the Python path and the deployed graph disagree by construction rather than
# by oversight. Treat it as a diagnostic switch, not a pipeline stage.
EAR_GEOMETRY_FILTER_ENABLED = False
EAR_MIN_ASPECT_RATIO = 0.14
EAR_MAX_ASPECT_RATIO = 2.57
EAR_MIN_SIZE_FRAC = 0.022
EAR_MAX_SIZE_FRAC = 0.733

# Extra duplicate suppression beyond IoU NMS (centre-distance and coverage
# tests). OFF by default, and it must stay consistent across paths: it was
# previously applied only in the trainer's evaluation NMS and in no inference
# path, so reported metrics were computed with suppression that deployment
# never performed.
DUPLICATE_SUPPRESSION_ENABLED = False

# Anchor priors, as (width, height) normalized to the model input.
#
# The MediaPipe defaults are square and span 0.148-0.866, sized for faces. Ears
# are small and tall: median 0.053 wide by 0.108 high in this dataset. Measured
# best-IoU between a ground-truth ear and its closest anchor, over 13477 human
# boxes in train.csv:
#
#   fixed (w=h=1.0, the original)   median 0.006   0.1% reach IoU 0.5
#   MediaPipe variable-size         median 0.234  18.0% reach IoU 0.5
#   the fitted priors below         median 0.372  23.8% reach IoU 0.5
#
# Even fitted priors leave most ears under IoU 0.5, because the binding
# constraint is spatial: stride 8 spaces anchor centres 0.0625 apart for objects
# 0.053 wide. That is why assignment is best-match per box rather than
# IoU-thresholded. A stride-4 head would raise the ceiling substantially
# (median 0.530, 57.6% at IoU 0.5) but changes the exported graph.
#
# Produced by k-means (k=8) over human-verified box (w, h), ordered by area.
# The two smallest go on the 16x16 grid, the rest on the 8x8 grid.
EAR_ANCHOR_PRIORS_SMALL = (
    (0.027, 0.044),
    (0.049, 0.095),
)
EAR_ANCHOR_PRIORS_BIG = (
    (0.069, 0.161),
    (0.136, 0.156),
    (0.114, 0.245),
    (0.228, 0.395),
    (0.507, 0.696),
    (0.748, 1.017),
)

# Hard-negative mining. The negative budget per image is
# (mean positives per image) * this ratio, floored at
# LossFunction.min_negatives_per_image.
#
# This was 1.5, which starves the loss: ANCHOR_TOP_K = 3 gives about 2.3
# positive anchors per image, so 1.5 yielded max(3, 10) = 10 negatives out of
# the 894 available, and the model settled at 0.34 background accuracy against
# 0.42 positive. Every run since has passed --hard-negative-ratio 50 by hand;
# a default that has to be overridden on every invocation to work is just a
# bug with a workaround, so it is now the measured value.
HARD_NEGATIVE_RATIO = 50.0

# Anchor assignment
ANCHOR_TOP_K = 3          # positives per ground-truth box
ANCHOR_IGNORE_IOU = 0.35  # non-positive anchors above this are neither pos nor neg
