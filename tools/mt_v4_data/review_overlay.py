#!/usr/bin/env python3
"""pseudo-label 확인용 오버레이 격자. da = 파랑, ll = 빨강, 박스 = 클래스 이름(무시 박스 -2 는 회색 점선 느낌의 얇은 선).

사용: python review_overlay.py <dataset 폴더> <출력.jpg> [--names a,b,...] [--flag 문자열] [--bag 이름] [--n 24] [--cols 4] [--seed 0]
"""
import argparse, json, os, random

import cv2
import numpy as np

CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking", "traffic_light",
           "light_red", "light_yellow", "light_green", "limo", "cone"]
COL = {"sign_A": (0, 0, 255), "sign_B": (255, 128, 0), "traffic_light": (0, 255, 255), "light_red": (0, 0, 255),
       "light_yellow": (0, 220, 255), "light_green": (0, 255, 0), "limo": (255, 0, 255)}

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--names", default=""); ap.add_argument("--flag", default=""); ap.add_argument("--bag", default="")
ap.add_argument("--n", type=int, default=24); ap.add_argument("--cols", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
meta = [json.loads(l) for l in open(os.path.join(a.root, "pseudo_meta.jsonl"))]
if a.names:
    want = set(a.names.split(",")); meta = [m for m in meta if m["name"] in want]
if a.flag:
    meta = [m for m in meta if any(a.flag in f for f in m["flags"])]
if a.bag:
    meta = [m for m in meta if a.bag in m["bag"]]
random.Random(a.seed).shuffle(meta)
meta = sorted(meta[:a.n], key=lambda m: (m["bag"], m["t"]))
tiles = []
for m in meta:
    d = os.path.join(a.root, m["split"])
    img = cv2.imread(os.path.join(d, "images", m["name"] + ".png"))
    da = cv2.imread(os.path.join(d, "da_masks", m["name"] + ".png"), 0) > 0
    ll = cv2.imread(os.path.join(d, "ll_masks", m["name"] + ".png"), 0) > 0
    o = img.copy()
    o[da] = (0.55 * o[da] + 0.45 * np.array([255, 120, 0])).astype(np.uint8)
    o[ll] = (0, 0, 255)
    for line in open(os.path.join(d, "labels", m["name"] + ".txt")):
        c, cx, cy, w, h = line.split(); c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
        x1, y1, x2, y2 = int((cx - w / 2) * 640), int((cy - h / 2) * 360), int((cx + w / 2) * 640), int((cy + h / 2) * 360)
        if c == -2:
            cv2.rectangle(o, (x1, y1), (x2, y2), (160, 160, 160), 1); cv2.putText(o, "ign", (x1, max(10, y1 - 3)), 0, 0.4, (160, 160, 160), 1)
        else:
            col = COL.get(CLASSES[c], (255, 255, 255))
            cv2.rectangle(o, (x1, y1), (x2, y2), col, 2); cv2.putText(o, CLASSES[c], (x1, max(10, y1 - 3)), 0, 0.45, col, 1)
    cv2.rectangle(o, (0, 0), (640, 20), (0, 0, 0), -1)
    cv2.putText(o, f"{m['name']}  M{m['mission']}  {' '.join(m['flags'])[:60]}", (4, 15), 0, 0.45, (255, 255, 255), 1)
    tiles.append(o)
if not tiles:
    raise SystemExit("고른 프레임 없음")
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % a.cols)
g = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
cv2.imwrite(a.out, g, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(a.out, len(meta), "장")
