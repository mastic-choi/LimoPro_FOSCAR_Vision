#!/usr/bin/env python3
"""정지선 검수 판정(sl_decisions.csv)을 라벨로 쓴다. 검수한 프레임에만 sl 라벨이 생긴다(나머지는 학습에서 sl 손실 제외).

  model : sl = 모델 확률 > 0.5 (30 px 미만 조각은 '모름'), 믿을 픽셀 sl_valid = 확률 < 0.3 또는 > 0.7
  none  : sl = 0, sl_valid = 전부 (정지선 없음을 확실히)
  exist : 기존 sl_masks·sl_valid 그대로 복사
  skip  : sl 라벨 없음. 기존 라벨이 있던 프레임이면 remove_existing.txt 에 적어 둠 (합칠 때 그 프레임 sl 을 뺀다)
묶음(같은 배경 변형)은 대표 판정을 모든 변형에 쓴다. 모델 라벨은 변형마다 그 변형의 확률로 만든다.

출력: <out>/<ds>/<split>/sl_masks|sl_valid/<stem>.png, <out>/<ds>/remove_existing.txt, <out>/sl_apply_summary.json
  bag 데이터는 --bag-root 를 주면 그 데이터셋 폴더(<split>/sl_masks, sl_valid)에 바로 쓴다.
사용: python sl_apply.py <sl_review> <out> [--bag-root bag_mt_v1]
"""
import argparse, csv, json, os, shutil
from collections import Counter

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("slr"); ap.add_argument("out"); ap.add_argument("--bag-root", default="")
a = ap.parse_args()
items = {it["id"]: it for it in json.load(open(os.path.join(a.slr, "candidates.json")))}
dec = {r["id"]: r["choice"] for r in csv.DictReader(open(os.path.join(a.slr, "sl_decisions.csv"), encoding="utf-8"))}
cnt, frames, removed = Counter(), Counter(), {}

for iid, c in dec.items():
    it = items[iid]; ds, sp = it["ds"], it["split"]
    base = os.path.join(a.bag_root, sp) if (ds == "bag" and a.bag_root) else os.path.join(a.out, ds, sp)
    cnt[c] += 1
    for stem in it["members"]:
        if c == "skip":
            if os.path.exists(os.path.join(it["root"], sp, "sl_masks", stem + ".png")):
                removed.setdefault(ds, []).append(f"{sp}/{stem}")
            for sub in ("sl_masks", "sl_valid"):   # 다시 반영할 때 예전 판정의 라벨이 남지 않게
                p = os.path.join(base, sub, stem + ".png")
                if os.path.exists(p): os.remove(p)
            frames[c] += 1; continue
        for sub in ("sl_masks", "sl_valid"): os.makedirs(os.path.join(base, sub), exist_ok=True)
        if c == "exist":
            for sub in ("sl_masks", "sl_valid"):
                shutil.copy2(os.path.join(it["root"], sp, sub, stem + ".png"), os.path.join(base, sub, stem + ".png"))
        elif c == "none":
            cv2.imwrite(os.path.join(base, "sl_masks", stem + ".png"), np.zeros((360, 640), np.uint8))
            cv2.imwrite(os.path.join(base, "sl_valid", stem + ".png"), np.full((360, 640), 255, np.uint8))
        elif c == "model":
            p = cv2.imread(os.path.join(a.slr, ds, sp, "sl_prob", stem + ".png"), 0) / 255.0
            pos = p > 0.5
            n, lab, st, _ = cv2.connectedComponentsWithStats(pos.astype(np.uint8), 8)
            small = np.zeros(n, bool); small[1:] = st[1:, cv2.CC_STAT_AREA] < 30
            tiny = small[lab]
            sl = pos & ~tiny
            valid = ((p < 0.3) | (p > 0.7)) & ~tiny
            cv2.imwrite(os.path.join(base, "sl_masks", stem + ".png"), (sl * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(base, "sl_valid", stem + ".png"), (valid * 255).astype(np.uint8))
        frames[c] += 1
for ds, lst in removed.items():
    os.makedirs(os.path.join(a.out, ds), exist_ok=True)
    open(os.path.join(a.out, ds, "remove_existing.txt"), "w").write("\n".join(sorted(lst)) + "\n")
summ = {"groups": dict(cnt), "frames": dict(frames), "removed_existing": {k: len(v) for k, v in removed.items()},
        "undecided_groups": len(items) - len(dec)}
json.dump(summ, open(os.path.join(a.out, "sl_apply_summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, ensure_ascii=False))
