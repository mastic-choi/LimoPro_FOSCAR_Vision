#!/usr/bin/env python3
"""정지선 검수 후보가 아니었던 bag 프레임에도 multitask-v3real 정지선을 pseudo-label 로 넣는다 (사용자 요청 11-16).

  sl = 확률 > 0.5 (30 px 미만 조각은 '모름'), sl_valid = 확률 < 0.3 또는 > 0.7 (sl_apply.py 의 '모델 맞음'과 같은 규칙)
  이미 sl 라벨이 있는 프레임(정지선 검수 반영)과 정지선 검수에서 '애매·틀림'으로 판정한 프레임은 건드리지 않는다.
  넣은 프레임 목록 → <dataset>/sl_pseudo_unreviewed.txt (검수 화면에 'stop line: pseudo' 로 표시, 틀리면 5 사용 + 정지선 빼기)

사용: python sl_fill_pseudo.py <dataset bag_mt_v1> <sl_review>
"""
import csv, os, sys

import cv2
import numpy as np

root, slr = sys.argv[1], sys.argv[2]
skip = {r["rep"] for r in csv.DictReader(open(os.path.join(slr, "sl_decisions.csv"))) if r["ds"] == "bag" and r["choice"] == "skip"}
done = []
for sp in ("train", "test_bag1116"):
    for sub in ("sl_masks", "sl_valid"): os.makedirs(os.path.join(root, sp, sub), exist_ok=True)
    for f in sorted(os.listdir(os.path.join(root, sp, "images"))):
        stem = f[:-4]
        if os.path.exists(os.path.join(root, sp, "sl_masks", f)) or stem in skip:
            continue
        p = cv2.imread(os.path.join(slr, "bag", sp, "sl_prob", f), 0) / 255.0
        pos = p > 0.5
        n, lab, st, _ = cv2.connectedComponentsWithStats(pos.astype(np.uint8), 8)
        small = np.zeros(n, bool); small[1:] = st[1:, cv2.CC_STAT_AREA] < 30
        tiny = small[lab]
        cv2.imwrite(os.path.join(root, sp, "sl_masks", f), ((pos & ~tiny) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(root, sp, "sl_valid", f), ((((p < 0.3) | (p > 0.7)) & ~tiny) * 255).astype(np.uint8))
        done.append(f"{sp}/{stem}")
open(os.path.join(root, "sl_pseudo_unreviewed.txt"), "w").write("\n".join(done) + "\n")
print(f"pseudo 정지선 {len(done)}장 (검수 '애매' 제외 {len(skip)}장)")
