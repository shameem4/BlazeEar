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
DEFAULT_DETECTION_THRESHOLD = 0.70
DEFAULT_TRAIN_THRESHOLD = 0.3
DEFAULT_NMS_IOU_THRESHOLD = 0.35

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

# Anchor assignment
ANCHOR_TOP_K = 3          # positives per ground-truth box
ANCHOR_IGNORE_IOU = 0.35  # non-positive anchors above this are neither pos nor neg
