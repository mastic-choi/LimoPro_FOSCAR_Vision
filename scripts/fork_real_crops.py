#!/usr/bin/env python3
"""실물 표지판 M7 주행(원본 프레임 + sign_track.json + det.json) → 갈림길 분류기 크롭 (A / B / none).

  표지판: SAM2 박스(sign_track.json) → 로봇과 같은 크롭 규칙(fork_crop_rule.crop_rect, 640x360 검출 좌표 기준)으로 원본에서 자름.
          라벨 = 주행 글자(사람 확인). 버림: 화면 끝에 닿음(3 px), 가로/세로 비 0.75~1.33 밖(가려짐·잘림),
          지름 < --min-px (먼 구간은 흐려서 화살표가 안 보임, HANDOFF 규칙 "모양이 안 보이면 버림"). 2 프레임에 1장.
  없음:   det.json 의 일반(640x360) 검출 표지판류 후보(점수 >= 0.1) 중 그 프레임 표지판 박스와 거의 안 겹치는 것
          (교집합 / 후보 넓이 < 0.1). 3 프레임에 1장.
출력: <out>/{A|B}/, <out>/none/, <out>/crops.csv (버린 것도 이유와 함께)
사용 (fork venv): python scripts/fork_real_crops.py <run 폴더> <A|B> <out> [--min-px 60]
"""
import argparse, csv, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # fork_crop_rule
import fork_crop_rule as rule   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("sign", choices=["A", "B"]); ap.add_argument("out")
ap.add_argument("--min-px", type=int, default=60)
a = ap.parse_args()
F = json.load(open(os.path.join(a.run, "sign_track.json")))["frames"]
D = json.load(open(os.path.join(a.run, "det.json")))
for d in (a.sign, "none"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
rows, n = [], {"sign": 0, "none": 0, "drop_small": 0, "drop_partial": 0}


def crop(im, b):
    x, y, s, _ = rule.crop_rect((b[0] / 2, b[1] / 2, b[2] / 2, b[3] / 2), 640, 360, 1280, 720)
    return im[y:y + s, x:x + s]


for r, d in zip(F, D):
    i = r["i"]; sb = r["sam_box"]; im = None
    if sb:
        w, h = sb[2] - sb[0], sb[3] - sb[1]
        why = ""
        if sb[0] <= 3 or sb[1] <= 3 or sb[2] >= 1277 or sb[3] >= 717 or not (0.75 <= w / max(h, 1) <= 1.33):
            why = "drop_partial"
        elif max(w, h) < a.min_px:
            why = "drop_small"
        if not why and i % 2 == 0:
            im = cv2.imread(os.path.join(a.run, "frames", r["frame"]))
            fn = f"{os.path.basename(a.run)}_{i:05d}_sign.png"
            cv2.imwrite(os.path.join(a.out, a.sign, fn), crop(im, sb)); n["sign"] += 1
            rows.append([fn, i, a.sign, max(w, h), r.get("pass") or "near", "", ""])
        elif why:
            n[why] += 1; rows.append(["", i, why, max(w, h), r.get("pass") or "near", "", ""])
    if i % 3 == 0:
        for k, c in enumerate(x for x in d["cand"] if x["src"] == "full" and x["score"] >= 0.1):
            b = c["box"]
            if sb:
                ix = max(0, min(b[2], sb[2]) - max(b[0], sb[0])); iy = max(0, min(b[3], sb[3]) - max(b[1], sb[1]))
                if ix * iy / max(1.0, (b[2] - b[0]) * (b[3] - b[1])) >= 0.1: continue
            if im is None: im = cv2.imread(os.path.join(a.run, "frames", r["frame"]))
            fn = f"{os.path.basename(a.run)}_{i:05d}_{k}_{c['cls']}_{c['score']:.2f}.png"
            cv2.imwrite(os.path.join(a.out, "none", fn), crop(im, b)); n["none"] += 1
            rows.append([fn, i, "none", round(max(b[2] - b[0], b[3] - b[1])), "", c["cls"], c["score"]])
with open(os.path.join(a.out, "crops.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "frame", "label", "size_px_1280", "pass", "det_cls", "det_score"]); w.writerows(rows)
print(f"{os.path.basename(a.run)} ({a.sign}): {n}")
