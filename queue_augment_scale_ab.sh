#!/usr/bin/env bash
#
# QUEUED, NOT STARTED. Run this by hand when the GPU is free.
#
# Tests one hypothesis: that stage 2 under-detects small ears partly because
# the augmentation never showed it any. AUGMENT_SCALE_RANGE's low end went
# from 0.75 to 0.35, which moves the 0-32 px band from 5.99% of training
# boxes to 13.34% (measured over 2,500 augmented samples).
#
# WHY TWO ARMS. train_blazeear.py has no --seed, so run-to-run variation is
# not controlled and a single retrain cannot be compared against the shipped
# checkpoint -- that checkpoint also predates other changes, so a difference
# would have several candidate causes. The control arm here is the OLD scale
# range run under today's code, so the only thing differing between the two
# arms is the one constant. Seed noise is still uncontrolled, which is why
# the comparison below is paired per image rather than a difference of
# headline numbers.
#
# COST, measured rather than inferred. On this RTX 5060 Ti 16 GB at batch 32
# with AMP and 4 workers: 29.9 ms/step, 508 steps/epoch = 15.2 s/epoch of
# training, plus 11.4 s/epoch of validation with val mAP on (2.7 s with
# --no-val-map). So 26.6 s/epoch, and at the 100-epoch default:
#
#     0.74 GPU-hours per arm
#     1.48 GPU-hours for two arms   (89 minutes)
#     2.22 GPU-hours for three      (133 minutes)
#
# An earlier version of this comment said 1-3 GPU-hours PER ARM, inferred
# from the shipped checkpoint's file timestamps. That was wrong by roughly
# 2-4x: mtime gaps include whatever else the machine was doing between
# epochs. Timing 60 real steps costs a minute and is not an estimate.
#
# Arms run SEQUENTIALLY: the card has no room for two, and sharing it would
# make the timings useless.
#
# WHAT WOULD COUNT AS A RESULT. The treatment should lift recall in the
# 0-32 px and 32-48 px bands -- currently 18.2% and 58.0% against 95.7% at
# the mode -- without giving back the mode. A lift that comes with an equal
# loss at the mode is the resolution-coverage trade that killed the
# FACE_CROP_EXPAND sweep, reappearing on a different axis.
#
# WHAT WOULD NOT COUNT, pre-registered here so it is decided before the
# numbers exist rather than after. A single band improved by a couple of
# points with the mode flat, on one draw and with no interval, is NOT a
# result -- it is the same shape as trainable_blazeface's one positive bin
# (16-32 px, +1.2 against stock) which they themselves downgraded as "exactly
# the size of thing that does not replicate": one bin, one seed, adjacent to
# a boundary they had chosen, no interval. The honest read of that shape is
# "needs a second draw", and the next step is to re-run both arms rather than
# to write it up.
#
# Note on what "another draw" means here: train_blazeear.py has no --seed, so
# every run is an independent, unreproducible draw. That is adequate for
# REPLICATION -- run the pair again and see whether the effect survives --
# and inadequate for exact reproduction, so a result that matters will need a
# --seed added before it can be pinned. Worth doing if B1 pays; not worth
# touching the trainer for speculatively.
#
# THE KNOWN CONFOUND, which this pair does NOT resolve. augment_scale pads
# with black when it shrinks, so at 0.35 roughly 65% of the frame is border
# against 25% at 0.75. A lift could be the model learning "small ear" or it
# could be learning "black surround". If the treatment wins, vary the padding
# next; if it loses, suspect the padding first. See utils/config.py.
set -euo pipefail

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/home/shameem/.local/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
PY=/home/shameem/.conda/envs/mocap/bin/python

STAMP=$(date +%Y-%m-%d)
SCRATCH="/var/tmp/claude/blazeear/${STAMP}-augment-scale-ab"
mkdir -p "$SCRATCH"

# An as-run snapshot BEFORE anything launches. A finished run whose code can
# no longer be identified is uninterpretable, and the confound can no longer
# be ruled out.
git rev-parse HEAD > "$SCRATCH/commit.txt"
git status --porcelain > "$SCRATCH/dirty.txt"
cp utils/config.py dataloader.py utils/augmentation.py "$SCRATCH/"
sha256sum utils/config.py dataloader.py utils/augmentation.py \
    train_blazeear.py loss_functions.py > "$SCRATCH/sha256.txt"
echo "as-run snapshot in $SCRATCH"

run_arm () {
    local name="$1" low="$2" high="$3" cap="$4"
    local ckpt="runs/checkpoints_${name}" log="runs/logs_${name}"

    if [ -e "$ckpt" ]; then
        echo "REFUSING: $ckpt already exists -- move it aside or pick another name." >&2
        return 1
    fi

    # The range is read from config at import, so the arm sets it in the
    # environment and a one-line sitecustomize-free shim applies it. Editing
    # config.py between arms would break the as-run snapshot above.
    echo "=== arm $name: scale=($low, $high)  prior cap=$cap ==="
    systemd-run --user --unit="blazeear-$name" --same-dir --wait --collect \
        --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        --setenv=BLAZEEAR_AUGMENT_SCALE_LOW="$low" \
        --setenv=BLAZEEAR_AUGMENT_SCALE_HIGH="$high" \
        --setenv=BLAZEEAR_HARD_NEGATIVE_PRIOR_CAP="$cap" \
        "$PY" train_blazeear.py \
            --train-data data/splits/crops_train.csv \
            --val-data data/splits/crops_val.csv \
            --data-root data/face_crops/ \
            --checkpoint-dir "$ckpt" \
            --log-dir "$log" \
            --amp
    # systemd-run puts stdout in the journal, which survives a reboot:
    #   journalctl --user -u blazeear-$name
    # A plain background job's output dies with the crash that killed it.
}

# Three arms, because there are now two changes in flight and a two-arm
# design would leave one of them untested.
#
# HARD_NEGATIVE_PRIOR_CAP is ON by default (0.25), so a plain control/treatment
# pair would carry it in BOTH arms: that isolates B1 cleanly and leaves the cap
# itself never measured. The third arm turns the cap off alongside the old
# scale range, which makes `baseline` the real pre-change configuration and
# gives each lever its own contrast against it:
#
#     baseline   old scale, no cap     <- the configuration before either change
#     cap_only   old scale, cap 0.25   <- baseline + lever 2
#     control    old scale, cap 0.25   == cap_only, so it is not run twice
#     treatment  new scale, cap 0.25   <- cap_only + B1
#
# So: treatment vs cap_only isolates B1, and cap_only vs baseline isolates the
# cap. An extra 44 minutes to avoid confounding two changes is a good trade.
run_arm baseline  0.75 1.25 1.0
run_arm cap_only  0.75 1.25 0.25
run_arm treatment 0.35 1.25 0.25

echo
echo "Both arms done. Compare them PAIRED, per image, and report THREE things:"
echo
echo "  1. Band recalls, not the headline mAP. The hypothesis is about 0-32 px"
echo "     and 32-48 px, which are 5.9% and 10.5% of the data and can move a"
echo "     lot without moving a headline at all."
echo "       $PY evaluate_two_stage.py --help   # for the band-stratified path"
echo
echo "  2. The 64-80 px MODE band, on both arms, alongside (1). 1,402 of 3,790"
echo "     ears sit in that mode. A pure rebalance that lifted 0-48 px by"
echo "     starving the mode would read as a win on the band you aimed at and"
echo "     a loss in aggregate -- so the mode is the guard, not an extra."
echo "     (Guard from Ear_Landmarker, who scoped the lever.)"
echo
echo "  3. Mean positives-per-ear and per-prior positive share, per arm, since"
echo "     the assignment statistics move BEFORE recall does and will say"
echo "     whether the rebalance happened at all:"
echo "       BLAZEEAR_AUGMENT_SCALE_LOW=... $PY measure_anchor_budget.py"
echo
echo "  4. FALSE POSITIVES per image at the LARGE bands, for the cap arms."
echo "     The cap takes the 129.8x178.2 prior from 54.3% of the negative"
echo "     budget to 25.3%, and that prior's job was suppressing large false"
echo "     positives. Fewer negatives there is the cap's specific risk, and"
echo "     recall alone will not show it."
echo
echo "There is also an external scoring offer on the table, recorded here so"
echo "it is not lost: Ear_Landmarker will score both arms on a human"
echo "ground-truth basis -- 3,790 ears delivered in crop, recall per size band"
echo "at matched FP per image, 0-32 and 32-48 never pooled, and the 64-80 mode"
echo "reported alongside. Taking it up is the user's call, not this script's."
