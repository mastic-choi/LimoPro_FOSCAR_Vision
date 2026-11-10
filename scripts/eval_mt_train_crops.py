#!/usr/bin/env python3
"""train 의 sign_A/sign_B/parking 박스를 잘라 몽타주 (Klein 다듬은 사진 / 원본 따로). 라벨이 맞는지 눈으로 확인용.
사용: python -I scripts/eval_mt_train_crops.py . <출력 폴더>
"""
import os, sys, csv, random, glob
import cv2, numpy as np
root, out = sys.argv[1], sys.argv[2]
d = os.path.join(root, "dataset/synth_signs_v2/train")
kept = set()
for f in glob.glob(os.path.join(d, "klein_refine*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["status"] == "kept":
            kept.add(r["image"])
NAMES = {1: "A", 2: "B", 5: "P"}
rows = {c: [] for c in NAMES}
for lab in sorted(glob.glob(os.path.join(d, "labels", "*.txt"))):
    stem = os.path.basename(lab)[:-4]
    for l in open(lab):
        p = l.split()
        if p and int(p[0]) in NAMES:
            rows[int(p[0])].append((stem, [float(v) for v in p[1:5]]))
random.seed(0)
S = 120
for klein in (True, False):
    lines = []
    for c, rr in rows.items():
        pool = [r for r in rr if (r[0] in kept) == klein]
        sel = random.sample(pool, min(16, len(pool)))
        tiles = []
        for stem, (cx, cy, w, h) in sel:
            img = cv2.imread(glob.glob(os.path.join(d, "images", stem + ".*"))[0])
            H, W = img.shape[:2]
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            s = max(w, h) * 2 + 6
            X1, Y1, X2, Y2 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
            cr = img[Y1:Y2, X1:X2]
            sc = S / max(cr.shape[:2])
            cr = cv2.resize(cr, (max(1, round(cr.shape[1] * sc)), max(1, round(cr.shape[0] * sc))), interpolation=cv2.INTER_NEAREST)
            t = np.full((S, S, 3), 25, np.uint8); t[:cr.shape[0], :cr.shape[1]] = cr
            cv2.putText(t, f"{NAMES[c]} {np.sqrt(w*h):.0f}px", (3, S - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        while len(tiles) < 16:
            tiles.append(np.full((S, S, 3), 25, np.uint8))
        lines.append(np.hstack(tiles))
        print(("klein" if klein else "orig"), NAMES[c], "n", len(pool))
    g = np.vstack(lines)
    head = np.full((34, g.shape[1], 3), 30, np.uint8)
    cv2.putText(head, f"train crops, rows A / B / P ({'Klein refined' if klein else 'not refined'})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, f"ab_train_crops_{'klein' if klein else 'orig'}.jpg"), np.vstack([head, g]), [cv2.IMWRITE_JPEG_QUALITY, 90])
