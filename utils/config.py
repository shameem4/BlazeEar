"""
Configuration constants for BlazeEar detection.
"""
import os

# Model input size.
#
# The ear model runs at 256. The network is fully convolutional, so the
# MediaPipe weights load unchanged at any size and only the spatial grids grow;
# the parameter count is identical at 128 and 256. Measured on crops_val
# through the deployed path, 256 is worth +0.0353 mAP@0.5 (95% CI [+0.0210,
# +0.0499]) for 3.0x the ear model's CPU time -- see README. It also beats
# 128-with-4-view-TTA outright while costing less than it.
DEFAULT_INPUT_SIZE = 256

# The face stage is MediaPipe's front BlazeFace and stays at 128: those are
# someone else's weights at their own input size. Both sizes therefore exist in
# one process, which is why input size is a BlazeEar constructor argument and
# not only this constant.
FACE_INPUT_SIZE = 128

# Anchor configuration, derived from the input size rather than restated.
# The stride-8 head is input/8 cells across and the stride-16 head input/16, so
# 128 gives the historical 16/8 grids and 896 anchors, and 256 gives 32/16 and
# 3584. These were literals, which meant changing the input size left the
# anchor count behind -- and a decode built on the wrong anchor count produces
# boxes at the wrong scale rather than an error.
SMALL_ANCHORS_PER_CELL = 2
BIG_ANCHORS_PER_CELL = 6


def grids_for(input_size: int) -> tuple[int, int]:
    """(stride-8 grid, stride-16 grid) for an input size."""
    return input_size // 8, input_size // 16


def total_anchors_for(input_size: int) -> int:
    small, big = grids_for(input_size)
    return (small ** 2 * SMALL_ANCHORS_PER_CELL
            + big ** 2 * BIG_ANCHORS_PER_CELL)


SMALL_GRID_SIZE, BIG_GRID_SIZE = grids_for(DEFAULT_INPUT_SIZE)
TOTAL_ANCHORS = total_anchors_for(DEFAULT_INPUT_SIZE)

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

# Detection post-processing. One value each, shared by every path that turns
# raw anchors into detections. These were previously spread across five
# implementations at three different IoU thresholds (0.3 in the model, 0.35
# forced by utils/model_utils, 0.5 in the trainer's evaluation), which meant
# the reported mAP measured post-processing that nothing deployed used.
#
# The deployed operating point. mAP sweeps the whole score range, so it says
# nothing about the one threshold that actually ships; this was 0.70, inherited
# from MediaPipe's face detector and never fitted to this model. Swept on the
# common held-out set (279 images, 355 ears) at IoU 0.5, precision/recall/F1:
#
#   threshold   plain pipeline        with 4-view TTA
#               prec   rec    F1      prec   rec    F1
#      0.70     0.614  0.628  0.621   0.508  0.682  0.582
#      0.75     0.681  0.614  0.646   0.582  0.668  0.622
#      0.80     0.758  0.592  0.665   0.665  0.642  0.653
#      0.85     0.838  0.541  0.658   0.769  0.620  0.686
#      0.90     0.894  0.428  0.579   0.878  0.566  0.688
#
# Two separate findings. The plain pipeline peaks at 0.80, not 0.70. And TTA
# shifts the optimum UP, because `max` fusion gives a cluster its best member's
# score -- so at 0.70 the shipped TTA was a measured REGRESSION on F1 against
# no TTA at all (-0.0389, 95% CI [-0.0649, -0.0123], paired bootstrap over
# 2000 resampled images), even while it gained +0.0467 mAP@0.5. Recalibrating
# TTA from 0.70 to its own operating point is worth +0.1054 F1
# (95% CI [+0.0715, +0.1404]).
#
# TTA takes 0.85 rather than its argmax 0.90: F1 is flat across 0.85-0.90
# (0.686 vs 0.688) and then falls off a cliff (0.666 at 0.91, 0.634 at 0.92),
# so the plateau centre is the safer choice than the edge of it. Both values
# are fitted on 355 ears; the direction is bootstrap-confirmed, the second
# decimal is not. The table is here so a deployment that wants precision over
# recall can re-pick deliberately rather than by inheritance.
DETECTION_SCORE_THRESHOLD = 0.80
TTA_SCORE_THRESHOLD = 0.85
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
# Crop side, in multiples of the face box.
#
# This is the LARGER of the two levers on the recall ceiling, which is the
# opposite of what the note below this used to say. Measured per ear on the
# common held-out set (355 ears), counting an ear as reachable when its centre
# falls inside some crop window:
#
#   reachable                            315   88.7%
#   no face found in the image             9    2.5%
#   face found, ear outside every crop    31    8.7%
#
# So 11.3% of ears can never reach the second stage, and the dominant cause is
# crop GEOMETRY, not face detection -- ears sit at the side of the head, and
# 1.5x a frontal face box does not always contain them. Those 40 ears score
# exactly 0.0000 mAP@0.5 at both input sizes, which is how the figures above
# were confirmed.
#
# Raising this captures more ears and spends more of the input on background,
# so it trades coverage against the ear's size in model pixels. 1.5 HAS now
# been re-swept at 256 px and it is the optimum; do not re-run this expecting
# a win. Two-stage recall at IoU>=0.3 over 1,200 full frames / 1,400 human
# ear boxes (measured by the Ear_Landmarker project, 2026-10-15):
#
#   expand   recall     vs 1.5                paired
#     1.20   71.93%    -7.93pp  p=6.8e-19     lost 139, gained 28
#     1.50   79.86%    optimal
#     1.90   75.57%    -4.29pp  p=2.6e-09     lost  82, gained 22
#     2.40   70.86%    -9.00pp  p=4.0e-24     lost 148, gained 22
#
# The trade is in the paired counts and breaks even at no setting: each
# widening covers a couple of dozen more ears and loses three to five times
# that many to the ear shrinking in the crop. An earlier version of this note
# argued the optimum should have moved at 256 because there is "resolution to
# spare" -- there is not, and that reasoning was wrong.
#
# So the 8.7% above is still the larger share of the ceiling and this constant
# is still its cause, but it is NOT recoverable by widening: a concentric
# square buys periphery in every direction while paying resolution everywhere,
# and ears sit lateral and posterior to a frontal face box. Anything aimed at
# those 31 ears needs a different crop PLACEMENT or SHAPE, not a larger scalar.
# The figures are final: the ranking and the mechanism reproduced on three
# independent ground-truth bases, and 1.5 scored 79.86% here against 79.55%
# on a separate human-basis run. One caveat stands -- the statistic is not the
# same one as the 8.7%, which is per-ear reachability on common_heldout rather
# than two-stage recall on full frames, so the two are compatible rather than
# directly comparable.
# See README.md for the long form.
FACE_CROP_EXPAND = 1.5
# Face score floor. One of the two levers on the recall ceiling, and the
# smaller one (see above): an ear whose face is missed never reaches the second
# stage, but far more ears are lost to the crop window than to the detector.
# Swept on
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

# Random scale augmentation, as (min, max) multipliers on the crop.
#
# This was (0.75, 1.25), and the low end is the lever. The median training ear
# is 64.9 px in a 256 px crop, so 0.75x is 48.7 px: the augmentation could not
# manufacture an ear in the 0-32 px band at all, which is the band the model is
# worst at -- 18.2% recall against 95.7% at the mode, and only 5.9% of the
# training boxes. It was being asked to generalise to sizes it had never been
# shown. 0.35 puts the median ear at 22.7 px and the p10 ear at 13.5.
#
# Known confound, stated because it is not fixed: `augment_scale`'s shrink
# branch pads with BLACK to restore the original size, so at 0.35 roughly 65%
# of the frame is border where at 0.75 it was 25%. A model could learn "small
# ear implies black surround" rather than "small ear", and that would not
# transfer. Edge-replicate padding is the obvious remedy, and it is
# deliberately NOT bundled in here: it would change the 0.75-1.0 range too, so
# the run would move two variables and explain neither. If the low end pays,
# the padding is the next thing to vary; if it does not, the padding is the
# first suspect.
#
# Overridable from the environment so an A/B can run both arms against one
# checked-out tree. Editing this file between arms would invalidate the as-run
# snapshot the arms are compared under, which is the thing that makes a
# finished run interpretable later.
AUGMENT_SCALE_RANGE = (
    float(os.environ.get('BLAZEEAR_AUGMENT_SCALE_LOW', 0.35)),
    float(os.environ.get('BLAZEEAR_AUGMENT_SCALE_HIGH', 1.25)),
)

# Anchor assignment
ANCHOR_TOP_K = 3          # positives per ground-truth box
ANCHOR_IGNORE_IOU = 0.35  # non-positive anchors above this are neither pos nor neg
