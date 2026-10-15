"""What exactly was lost. For every GT ear and every arm, record the best
candidate BEFORE the score gate, so a loss can be attributed:

  localisation loss -- no candidate reaches IoU 0.5 any more
  scoring loss      -- the candidate is still there, below the 0.80 gate
"""
import csv, os, pathlib, collections
import cv2, numpy as np, torch
os.chdir(pathlib.Path(__file__).resolve().parent)
from blazeear import BlazeEar
from utils.anchor_utils import anchor_options
from utils.config import (DEFAULT_INPUT_SIZE as S, DETECTION_SCORE_THRESHOLD as GATE,
                          NMS_IOU_THRESHOLD as NMS, HUMAN_ANNOTATION_SOURCES)
ARMS = ['baseline', 'cap_only', 'treatment']

def load(a):
    m = BlazeEar(); m.generate_anchors(anchor_options)
    sd = torch.load(f'runs/checkpoints_{a}/BlazeEar_best.pth', map_location='cpu',
                    weights_only=False)
    m.load_state_dict(sd.get('model_state_dict', sd), strict=False)
    m.min_score_thresh = 0.0          # pre-gate: keep every candidate
    m.min_suppression_threshold = NMS
    m = m.eval().to('cuda'); m.anchors = m.anchors.to('cuda'); return m

gt = collections.defaultdict(list)
for r in csv.DictReader(open('data/splits/crops_val.csv')):
    if r['annotation_source'] not in HUMAN_ANNOTATION_SOURCES: continue
    w, h = float(r['w']), float(r['h'])
    if w > 0 and h > 0:
        gt[r['image_path']].append((float(r['x1']), float(r['y1']), w, h))

def iou(a, b):
    if not len(b): return np.zeros(0)
    x0 = np.maximum(a[0], b[:,0]); y0 = np.maximum(a[1], b[:,1])
    x1 = np.minimum(a[2], b[:,2]); y1 = np.minimum(a[3], b[:,3])
    i = np.clip(x1-x0, 0, None) * np.clip(y1-y0, 0, None)
    return i / np.maximum((a[2]-a[0])*(a[3]-a[1])
                          + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - i, 1e-9)

rec = {a: {'iou': [], 'score': []} for a in ARMS}
meta = []
for a in ARMS:
    m = load(a); first = (a == ARMS[0])
    for rel, boxes in gt.items():
        img = cv2.imread(os.path.join('data/face_crops', rel))
        if img is None: continue
        side = max(img.shape[:2])
        with torch.no_grad():
            d = m.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cpu().numpy()
        pr = np.stack([d[:,1], d[:,0], d[:,3], d[:,2]], 1) if len(d) else np.zeros((0,4))
        sc = d[:, -1] if len(d) else np.zeros(0)
        for (x, y, w, h) in boxes:
            ious = iou(np.array([x, y, x+w, y+h]), pr)
            if len(ious):
                j = int(np.argmax(ious)); rec[a]['iou'].append(float(ious[j]))
                rec[a]['score'].append(float(sc[j]))
            else:
                rec[a]['iou'].append(0.0); rec[a]['score'].append(0.0)
            if first:
                meta.append({'band': max(w, h)/side*S, 'img': rel})
out = pathlib.Path('runs/arm_loss_anatomy.npy')
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, {'rec': rec, 'meta': meta}, allow_pickle=True)
print(f'wrote {out}')
print(f'recorded {len(meta)} ears x {len(ARMS)} arms')
