#!/usr/bin/env python3
"""synth_signs.py 결과를 박스와 함께 몽타주로 본다.
사용: .venv/Scripts/python scripts/synth_preview.py <데이터 폴더> <출력.jpg> [--n 24] [--seed 0] [--pool venue|general]
"""
import argparse, csv, os, random
import cv2
import numpy as np

COL = {"traffic_light": (0, 255, 255), "light_red": (0, 0, 255), "light_yellow": (0, 200, 255),
       "light_green": (0, 255, 0)}

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pool", default=None)
a = ap.parse_args()
classes = open(os.path.join(a.data, "classes.txt")).read().split()
rows = list(csv.DictReader(open(os.path.join(a.data, "meta.csv"), encoding="utf-8")))
rows = [r for r in rows if r["objects"] != "[]" and (a.pool is None or r["pool"] == a.pool)]
random.seed(a.seed)
pick = random.sample(rows, min(a.n, len(rows)))
tiles = []
for r in pick:
    im = cv2.imread(os.path.join(a.data, "images", r["image"] + ".jpg"))
    H, W = im.shape[:2]
    im = cv2.resize(im, (640, 360))
    for line in open(os.path.join(a.data, "labels", r["image"] + ".txt")):
        c, cx, cy, w, h = line.split()
        c = classes[int(c)]
        cx, cy, w, h = float(cx) * 640, float(cy) * 360, float(w) * 640, float(h) * 360
        col = COL.get(c, (255, 0, 255))
        cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), col, 1)
        if not c.startswith("light_"):
            cv2.putText(im, c, (int(cx - w / 2), max(10, int(cy - h / 2) - 3)), 0, 0.4, col, 1)
    cv2.putText(im, f"{r['image']} {r['pool']}", (5, 15), 0, 0.45, (255, 255, 255), 1)
    tiles.append(im)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
