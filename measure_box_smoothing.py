#!/usr/bin/env python3
"""EMA or One Euro for the demo's box smoothing? Measure it on the clips.

The demo shipped an EMA at k=0.6 chosen by eye. `Ear_Landmarker` measured One
Euro cutting jitter 66-80% while also lowering error, on its own pipeline and
with landmark ground truth. This repo has no per-frame box labels for the 25
clips, so the error half of that cannot be reproduced here. What can be
measured is the trade every smoother makes, without labels:

  JITTER  frame-to-frame movement of the EMITTED box while the ear is nearly
          still. The raw detector wobbles on a stationary ear; that wobble is
          the artefact a smoother exists to remove. Lower is better.

  LAG     distance between the emitted box and the raw detection while the
          ear is moving fast. Every smoother buys jitter reduction with lag;
          the question is the exchange rate. Lower is better.

Speed is measured from the RAW box series, so the two arms are banded by the
same frames. Association is also identical across arms by construction: the
tracker matches on the raw box, so track identity does not depend on the
smoother at all. Each track's raw series is extracted once and replayed
through each smoother, which removes association as a source of difference.

A smoother that wins on both is better. A smoother that wins on one is a
trade, and then the honest answer is the exchange rate rather than a verdict.

    python measure_box_smoothing.py

Prediction recorded before running: One Euro should win on jitter at
comparable lag, because its alpha adapts to speed where the EMA's does not,
and should win by more at the high-frame-rate clips, where a fixed k smooths
a fixed fraction per FRAME rather than per SECOND. If it loses on lag at
equal jitter, the EMA stays and the One Euro port is dead.
"""
import argparse
import glob
import math
import pickle
import statistics

from utils.box_tracker import (BoxTracker, EmaBoxSmoother, OneEuroBoxSmoother,
                               CONFIRM_SCORE, CONTINUE_SCORE, MIN_HITS,
                               MAX_MISSES)

DEFAULT_PKL = '/var/tmp/claude/blazeear/clips_frames.pkl'


def to_detections(frame):
    """Cached candidates are [ymin, xmin, ymax, xmax]; the tracker wants
    xmin/ymin/width/height."""
    out = []
    for d in frame:
        ymin, xmin, ymax, xmax = d['box']
        out.append({'xmin': xmin, 'ymin': ymin,
                    'width': xmax - xmin, 'height': ymax - ymin,
                    'confidence': d['score']})
    return out


def centre(b):
    return (b['xmin'] + b['width'] / 2.0, b['ymin'] + b['height'] / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def raw_series(frames, fps):
    """Per-track [(t, raw_box)] from a run with no smoother at all.

    Matching reads the raw box, so this series is what every arm would have
    seen; the smoother cannot change which detection joins which track.
    """
    tracker = BoxTracker(confirm_score=CONFIRM_SCORE,
                         continue_score=CONTINUE_SCORE, min_hits=MIN_HITS,
                         max_misses=MAX_MISSES, smoother=None)
    series = {}
    for i, frame in enumerate(frames):
        t = i / fps
        for tr in tracker.update(to_detections(frame), t):
            if not tr['matched']:
                continue                 # coasting: no new observation
            series.setdefault(tr['id'], []).append(
                (t, {'xmin': tr['xmin'], 'ymin': tr['ymin'],
                     'width': tr['width'], 'height': tr['height']}))
    return [s for s in series.values() if len(s) >= 8]


def replay(series, make):
    """Emitted boxes for one track under one smoother."""
    sm = make() if make is not None else None
    return [(t, sm.smooth(box, t) if sm is not None else dict(box))
            for t, box in series]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pkl', default=DEFAULT_PKL)
    p.add_argument('--clips', default='data/clips')
    p.add_argument('--fps-fallback', type=float, default=30.0)
    args = p.parse_args()

    clips = pickle.load(open(args.pkl, 'rb'))
    paths = sorted(glob.glob(args.clips + '/*.mp4'))
    fpss = []
    if len(paths) == len(clips):
        import cv2
        for path in paths:
            cap = cv2.VideoCapture(path)
            f = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fpss.append(f if f and f > 1 else args.fps_fallback)
    else:
        print(f'  {len(paths)} videos against {len(clips)} cached clips; '
              f'using {args.fps_fallback} fps throughout')
        fpss = [args.fps_fallback] * len(clips)

    arms = (('raw (no smoothing)', None),
            ('EMA k=0.6 (shipped)', EmaBoxSmoother),
            ('One Euro (centre/size)', OneEuroBoxSmoother))
    jit = {name: [] for name, _ in arms}
    lag = {name: [] for name, _ in arms}
    tracks = 0

    for frames, fps in zip(clips, fpss):
        for series in raw_series(frames, fps):
            tracks += 1
            # Speed from the raw series, so every arm is banded identically.
            speeds = [0.0] + [dist(centre(series[i][1]), centre(series[i-1][1]))
                              / max(series[i][0] - series[i-1][0], 1e-6)
                              for i in range(1, len(series))]
            lo = statistics.quantiles(speeds, n=3)[0]
            hi = statistics.quantiles(speeds, n=3)[1]
            for name, make in arms:
                out = replay(series, make)
                for i in range(1, len(out)):
                    size = max(series[i][1]['width'], series[i][1]['height'])
                    if size <= 0:
                        continue
                    if speeds[i] <= lo:
                        jit[name].append(
                            dist(centre(out[i][1]), centre(out[i-1][1])) / size)
                    if speeds[i] >= hi:
                        lag[name].append(
                            dist(centre(out[i][1]), centre(series[i][1])) / size)

    print(f'\n{tracks} tracks of >=8 matched frames across {len(clips)} clips')
    print('jitter: emitted-box movement between frames while nearly still')
    print('lag:    emitted box vs raw detection while moving fast')
    print('both as a fraction of the box\'s longest side; lower is better\n')
    base_j = statistics.median(jit['raw (no smoothing)'])
    print(f'  {"arm":<24} {"jitter":>9} {"vs raw":>9} {"lag":>9} {"n":>8}')
    print('  ' + '-' * 64)
    for name, _ in arms:
        j = statistics.median(jit[name])
        l = statistics.median(lag[name])
        print(f'  {name:<24} {j:9.4f} {100*(j/base_j-1):+8.1f}% '
              f'{l:9.4f} {len(jit[name]):8d}')


if __name__ == '__main__':
    main()
