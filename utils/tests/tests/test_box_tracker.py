"""The box tracker, and its JavaScript mirror.

Constants-only pinning is not enough here and this file exists because of a
specific failure: `docs/index.html` matched IoU against the smoothed box under
a comment saying it matched against the raw one, and the existing test checked
the constants, which were correct. Replaying the clips both ways moved lateral
confirmations 189 -> 212. So the parity test below runs BOTH implementations
over one shared sequence and compares every emitted box, which is the only
form that catches a behavioural divergence.
"""
import json
import pathlib
import shutil
import subprocess
import unittest

from utils.box_tracker import (BoxTracker, EmaBoxSmoother, OneEuroBoxSmoother,
                               CONFIRM_SCORE, CONTINUE_SCORE, MIN_HITS,
                               MAX_MISSES, MATCH_IOU, iou)

ROOT = pathlib.Path(__file__).resolve().parents[3]
JS = ROOT / 'docs' / 'ear_tracker.js'
FIXTURE = ROOT / 'utils' / 'tests' / 'assets' / 'tracker_sequence.json'


def det(xmin, ymin, w, h, conf):
    return {'xmin': float(xmin), 'ymin': float(ymin),
            'width': float(w), 'height': float(h), 'confidence': float(conf)}


class TestHysteresis(unittest.TestCase):
    def test_a_track_is_not_reported_before_min_hits(self):
        tr = BoxTracker()
        d = [det(10, 10, 20, 30, 0.9)]
        self.assertEqual(len(tr.update(d)), 0)
        self.assertEqual(len(tr.update(d)), 0)
        self.assertEqual(len(tr.update(d)), 1)

    def test_a_candidate_below_confirm_never_starts_a_track(self):
        tr = BoxTracker()
        d = [det(10, 10, 20, 30, 0.70)]
        for _ in range(10):
            self.assertEqual(tr.update(d), [])

    def test_a_confirmed_track_continues_below_the_confirm_score(self):
        """The whole point of hysteresis: entry is dear, staying is cheap."""
        tr = BoxTracker()
        for _ in range(3):
            tr.update([det(10, 10, 20, 30, 0.9)])
        out = tr.update([det(10, 10, 20, 30, 0.70)])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]['matched'])

    def test_a_track_is_drawn_through_a_gap_and_flagged_unmatched(self):
        """Dropping the box on the first miss is what put the strobe back."""
        tr = BoxTracker()
        for _ in range(3):
            tr.update([det(10, 10, 20, 30, 0.9)])
        out = tr.update([])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]['matched'])
        self.assertIsNone(out[0]['det_index'])

    def test_a_track_dies_past_max_misses(self):
        tr = BoxTracker()
        for _ in range(3):
            tr.update([det(10, 10, 20, 30, 0.9)])
        for _ in range(MAX_MISSES + 1):
            tr.update([])
        self.assertEqual(tr.update([]), [])

    def test_two_tracks_cannot_absorb_the_same_detection(self):
        tr = BoxTracker()
        both = [det(10, 10, 20, 30, 0.95), det(300, 10, 20, 30, 0.90)]
        for _ in range(3):
            out = tr.update(both)
        self.assertEqual(len(out), 2)
        self.assertEqual({o['id'] for o in out}, {1, 2})
        self.assertEqual(sorted(o['det_index'] for o in out), [0, 1])

    def test_ids_are_stable_and_never_reused(self):
        tr = BoxTracker()
        d = [det(10, 10, 20, 30, 0.9)]
        for _ in range(3):
            out = tr.update(d)
        first = out[0]['id']
        for _ in range(MAX_MISSES + 2):
            tr.update([])
        for _ in range(3):
            out = tr.update(d)
        self.assertNotEqual(out[0]['id'], first,
                            'a new track must not inherit a dead id')

    def test_pass_through_config_reports_every_matched_frame(self):
        """Ear_Landmarker's adoption path: association and smoothing, no
        hysteresis, but a track that survives a dropout."""
        tr = BoxTracker(confirm_score=0.70, continue_score=0.70, min_hits=1,
                        max_misses=5, smoother=OneEuroBoxSmoother, fps=30)
        d = [det(10, 10, 20, 30, 0.75)]
        self.assertEqual(len(tr.update(d)), 1)
        out = tr.update([])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]['matched'])
        again = tr.update(d)
        self.assertEqual(again[0]['id'], 1,
                         'the id, and so any per-track state keyed on it, '
                         'must survive a dropout')

    def test_matching_uses_the_raw_box_not_the_smoothed_one(self):
        """The bug this module was created to make impossible.

        With a smoother that returns a fixed far-away box, a tracker that
        matched on the emitted box would lose the detection immediately. One
        that matches on the raw box keeps it.
        """
        class Elsewhere:
            def smooth(self, box, t=None):
                return {'xmin': 1000.0, 'ymin': 1000.0,
                        'width': 20.0, 'height': 30.0}

        tr = BoxTracker(smoother=Elsewhere)
        d = [det(10, 10, 20, 30, 0.9)]
        for _ in range(4):
            out = tr.update(d)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['hits'], 4, 'the track lost its own detection')
        self.assertEqual(out[0]['xmin'], 1000.0, 'the emitted box is smoothed')


class TestSmoothers(unittest.TestCase):
    def test_one_euro_is_fed_only_on_matched_frames(self):
        """A coasting track must emit its last smoothed box unchanged.

        Feeding the filter a repeated box at small dt reads as "the ear
        stopped", drags the speed estimate toward zero and over-smooths the
        next real motion.
        """
        calls = []

        class Counting(OneEuroBoxSmoother):
            def smooth(self, box, t):
                calls.append(t)
                return super().smooth(box, t)

        tr = BoxTracker(smoother=Counting, fps=30)
        d = [det(10, 10, 20, 30, 0.9)]
        for _ in range(3):
            tr.update(d)
        n = len(calls)
        before = tr.update([])[0]
        self.assertEqual(len(calls), n, 'the smoother ran during a gap')
        after = tr.update([])[0]
        self.assertEqual(before['xmin'], after['xmin'])

    def test_pre_confirmation_frames_are_fed(self):
        """A warm start, not contamination: they are real detections of the
        same track, withheld from display only because it has not confirmed."""
        calls = []

        class Counting(OneEuroBoxSmoother):
            def smooth(self, box, t):
                calls.append(t)
                return super().smooth(box, t)

        tr = BoxTracker(smoother=Counting, fps=30)
        d = [det(10, 10, 20, 30, 0.9)]
        tr.update(d)
        tr.update(d)
        self.assertEqual(len(calls), 2, 'the filter cold-started on display')

    def test_one_euro_catches_up_over_a_longer_gap(self):
        """Larger dt gives larger alpha, so a reappearing ear is not dragged."""
        near, far = {'xmin': 0.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}, \
                    {'xmin': 100.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}
        short, long_ = OneEuroBoxSmoother(), OneEuroBoxSmoother()
        short.smooth(near, 0.0)
        long_.smooth(near, 0.0)
        a = short.smooth(far, 1 / 30.0)['xmin']
        b = long_.smooth(far, 10 / 30.0)['xmin']
        self.assertGreater(b, a)

    def test_one_euro_keeps_the_box_in_shape(self):
        """Centre/size parameterisation: a wobbling box cannot be sheared."""
        sm = OneEuroBoxSmoother()
        for i in range(20):
            out = sm.smooth({'xmin': 10.0 + (i % 2), 'ymin': 10.0,
                             'width': 20.0, 'height': 40.0}, i / 30.0)
        self.assertAlmostEqual(out['width'] / out['height'], 0.5, places=2)

    def test_duplicate_timestamp_returns_the_previous_output(self):
        sm = OneEuroBoxSmoother()
        sm.smooth({'xmin': 0.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}, 1.0)
        a = sm.smooth({'xmin': 50.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}, 2.0)
        b = sm.smooth({'xmin': 90.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}, 2.0)
        self.assertEqual(a['xmin'], b['xmin'])

    def test_ema_has_no_timestep(self):
        """Recorded as the structural complaint against it: the same motion is
        smoothed by a different amount at a different frame rate."""
        a, b = EmaBoxSmoother(), EmaBoxSmoother()
        box0 = {'xmin': 0.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}
        box1 = {'xmin': 100.0, 'ymin': 0.0, 'width': 20.0, 'height': 20.0}
        a.smooth(box0, 0.0); b.smooth(box0, 0.0)
        self.assertEqual(a.smooth(box1, 1 / 60.0)['xmin'],
                         b.smooth(box1, 1 / 24.0)['xmin'])


class TestIou(unittest.TestCase):
    def test_disjoint_boxes(self):
        self.assertEqual(iou(det(0, 0, 10, 10, 1), det(100, 100, 10, 10, 1)), 0.0)

    def test_identical_boxes(self):
        self.assertEqual(iou(det(0, 0, 10, 10, 1), det(0, 0, 10, 10, 1)), 1.0)

    def test_a_sharp_size_change_falls_below_the_match_floor(self):
        """Why IoU and not nearest-centre: concentric boxes 3x apart in size
        are the same centre and a different object."""
        self.assertLess(iou(det(0, 0, 30, 30, 1), det(-30, -30, 90, 90, 1)),
                        MATCH_IOU)


class TestJsParity(unittest.TestCase):
    """Both implementations over one shared sequence, box for box."""

    def setUp(self):
        if not shutil.which('node'):
            self.skipTest('node not available')
        self.seq = json.loads(FIXTURE.read_text())

    def _python(self, smoother):
        tr = BoxTracker(smoother=smoother)
        return [[{k: r[k] for k in ('id', 'matched', 'xmin', 'ymin',
                                    'width', 'height', 'hits', 'misses')}
                 for r in tr.update(f['detections'], f['t'])]
                for f in self.seq['frames']]

    def _js(self, smoother):
        script = f'''
        const {{BoxTracker, OneEuroBoxSmoother, EmaBoxSmoother}} =
            require({str(JS)!r});
        const seq = require({str(FIXTURE)!r});
        const sm = {smoother};
        const tr = new BoxTracker({{smoother: sm}});
        const out = seq.frames.map(f => tr.update(f.detections, f.t).map(r => ({{
            id: r.id, matched: r.matched, xmin: r.xmin, ymin: r.ymin,
            width: r.width, height: r.height, hits: r.hits, misses: r.misses
        }})));
        console.log(JSON.stringify(out));
        '''
        res = subprocess.run(['node', '-e', script], capture_output=True,
                             text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def _compare(self, py, js, label):
        self.assertEqual(len(py), len(js))
        for i, (a, b) in enumerate(zip(py, js)):
            self.assertEqual(len(a), len(b),
                             f'{label}: frame {i} reports {len(a)} tracks in '
                             f'Python and {len(b)} in JS')
            for ra, rb in zip(a, b):
                self.assertEqual(ra['id'], rb['id'], f'{label}: frame {i} id')
                self.assertEqual(ra['matched'], rb['matched'],
                                 f'{label}: frame {i} matched')
                self.assertEqual(ra['hits'], rb['hits'], f'{label}: frame {i} hits')
                for k in ('xmin', 'ymin', 'width', 'height'):
                    self.assertAlmostEqual(
                        ra[k], rb[k], places=9,
                        msg=f'{label}: frame {i} {k} diverges')

    def test_raw_parity(self):
        self._compare(self._python(None), self._js('null'), 'raw')

    def test_one_euro_parity(self):
        self._compare(self._python(OneEuroBoxSmoother),
                      self._js('OneEuroBoxSmoother'), 'one euro')

    def test_ema_parity(self):
        self._compare(self._python(EmaBoxSmoother),
                      self._js('EmaBoxSmoother'), 'ema')

    def test_both_languages_reject_an_unknown_option(self):
        """A misspelled option must fail, not run a constant nobody chose.

        `headsize v2` hit the concrete case from the consumer side: passing
        this module's own Python names to the JS ({confirm_score: 0.7,
        min_hits: 1}), which took none of them, silently ran 0.80 / minHits 3
        and emitted nothing on frame one. Python's signature already rejected
        it; `Object.assign` accepted anything, so the two languages disagreed
        about whether a typo was an error. That is the same shape as the
        slider's value="0.70" bug, and it is likely precisely BECAUSE the
        naming differs between the two files.
        """
        with self.assertRaises(TypeError):
            BoxTracker(confirmScore=0.7)        # the JS spelling, in Python
        script = f"""
        const {{BoxTracker}} = require({str(JS)!r});
        try {{
            new BoxTracker({{confirm_score: 0.7, min_hits: 1}});
            console.log('ACCEPTED');
        }} catch (e) {{
            console.log(e instanceof TypeError ? 'TYPEERROR' : 'OTHER');
        }}
        """
        res = subprocess.run(['node', '-e', script], capture_output=True,
                             text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), 'TYPEERROR',
                         'the Python spelling must not be silently ignored '
                         'by the JS')

    def test_the_js_is_not_an_es_module(self):
        """A bare `export` would break the parity test that guards this file.

        Consumers reach it by <script src> and globals, by require, or by a
        side-effect import plus globalThis. Making it an ES module is a
        syntax error under require(), which would silently disable the only
        check that compares the two implementations' behaviour.
        """
        text = JS.read_text()
        self.assertNotRegex(text, r'(?m)^\s*export\s',
                            'adding a bare export disables the parity test')
        self.assertIn('globalThis.BoxTracker', text)

    def test_field_names_are_pinned_per_language(self):
        """They differ on purpose -- idiomatic in each -- so pin both."""
        tr = BoxTracker()
        d = [det(10, 10, 20, 30, 0.9)]
        for _ in range(3):
            out = tr.update(d)
        self.assertIn('det_index', out[0])
        self.assertNotIn('detIndex', out[0])
        script = f"""
        const {{BoxTracker}} = require({str(JS)!r});
        const tr = new BoxTracker();
        const d = [{{xmin: 10, ymin: 10, width: 20, height: 30,
                    confidence: 0.9}}];
        let out;
        for (let i = 0; i < 3; i++) out = tr.update(d, i / 30);
        console.log(JSON.stringify(Object.keys(out[0])));
        """
        res = subprocess.run(['node', '-e', script], capture_output=True,
                             text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        keys = json.loads(res.stdout)
        self.assertIn('detIndex', keys)
        self.assertNotIn('det_index', keys)

    def test_a_parameterised_factory_works_and_agrees(self):
        """A tuned smoother must be passable the same way in both languages.

        Found by Ear_Landmarker from the consumer side: the JS did
        `new this.smoother()`, so `smoother: () => new OneEuroBoxSmoother(...)`
        -- the only way to pass non-default filter constants -- worked in
        Python and threw "this.smoother is not a constructor" in JS. The
        JSDoc said "factory" and this fixture only ever passed a class, so
        neither the documentation nor the tests covered the difference. They
        subclassed to get around it.

        This runs a factory with a non-default cutoff through both and
        compares the boxes, so the asymmetry cannot come back quietly and
        the parameters are proven to actually reach the filter.
        """
        tr = BoxTracker(smoother=lambda: OneEuroBoxSmoother(min_cutoff=2.0))
        py = [[{k: r[k] for k in ('id', 'matched', 'xmin', 'ymin',
                                  'width', 'height', 'hits', 'misses')}
               for r in tr.update(f['detections'], f['t'])]
              for f in self.seq['frames']]
        js = self._js('() => new OneEuroBoxSmoother({minCutoff: 2.0})')
        self._compare(py, js, 'parameterised factory')

        # ...and a tuned filter must differ from the default, or the test
        # above would pass on a factory whose arguments were ignored.
        default = self._python(OneEuroBoxSmoother)
        self.assertNotEqual([r['xmin'] for f in py for r in f],
                            [r['xmin'] for f in default for r in f],
                            'min_cutoff=2.0 produced the default output, so '
                            'the factory arguments are not reaching the '
                            'filter')

    def test_a_genuine_constructor_error_is_not_swallowed(self):
        """The class-or-factory fallback must not retry a real failure.

        It is narrowed to "is not a constructor" precisely so that a throw
        inside a smoother's constructor propagates rather than being called
        again as a function and failing somewhere less informative.
        """
        class Boom:
            def __init__(self):
                raise RuntimeError('boom')

        tr = BoxTracker(smoother=Boom)
        with self.assertRaises(RuntimeError):
            tr.update([det(10, 10, 20, 30, 0.9)])

        script = f"""
        const {{BoxTracker}} = require({str(JS)!r});
        class Boom {{ constructor() {{ throw new RangeError('boom'); }} }}
        try {{
            new BoxTracker({{smoother: Boom, minHits: 1}}).update(
                [{{xmin: 1, ymin: 1, width: 5, height: 5, confidence: 0.9}}], 0);
            console.log('SWALLOWED');
        }} catch (e) {{
            console.log(e.constructor.name);
        }}
        """
        res = subprocess.run(['node', '-e', script], capture_output=True,
                             text=True, cwd=ROOT)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), 'RangeError')

    def test_js_constants_match_python(self):
        text = JS.read_text()
        import re
        block = re.search(r'const TRACKER_DEFAULTS = \{(.*?)\};', text, re.S)
        self.assertIsNotNone(block)
        got = dict(re.findall(r'(\w+):\s*([\d.]+)', block.group(1)))
        self.assertEqual(float(got['confirmScore']), CONFIRM_SCORE)
        self.assertEqual(float(got['continueScore']), CONTINUE_SCORE)
        self.assertEqual(int(got['minHits']), MIN_HITS)
        self.assertEqual(int(got['maxMisses']), MAX_MISSES)
        self.assertEqual(float(got['matchIou']), MATCH_IOU)


if __name__ == '__main__':
    unittest.main()
