#!/usr/bin/env python3
"""3차 학습 데이터셋: 2차(ego_v2_clean) + 11/06 대회장 조명 사진(venue_20261106).

- 라벨 기준: da는 모든 구간에서 지금 달리는 차선만 (2026-11-06 결정). 2차 데이터에서 da가 화면의 --wide 이상인
  라벨(주로 3차선 전체를 칠한 것)은 이 기준과 달라서 뺀다. 목록: --wide-list
- 대회장 사진: train은 train에 더하고, test는 따로 test_venue로 둔다(대회장 성능을 따로 재려고).
  대회장 사진은 연속 프레임이라 test 구간 앞뒤 10프레임을 이미 빼 두었다(venue_split.json).
출력: <out>/{train,val,test,test_venue}/{images,da_masks,ll_masks}/ + manifest.csv

사용 예:
  .venv/Scripts/python -I scripts/build_train_dataset_v3.py dataset/ego_v3
"""
import argparse
import csv
import os
import shutil

import cv2

SUBDIRS = ("images", "da_masks", "ll_masks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="dataset/ego_v2_clean")
    ap.add_argument("--venue", default="dataset/venue_20261106")
    ap.add_argument("--wide", type=float, default=0.40, help="이 비율 이상 da를 칠한 2차 라벨은 뺌")
    args = ap.parse_args()

    with open(os.path.join(args.base, "manifest.csv"), encoding="utf-8", newline="") as f:
        base = [r for r in csv.DictReader(f) if r["use"] == "1"]
    with open(os.path.join(args.venue, "manifest.csv"), encoding="utf-8", newline="") as f:
        venue = list(csv.DictReader(f))
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    rows, wide = [], []
    for r in base:
        da = cv2.imread(os.path.join(args.base, r["split"], "da_masks", f"{r['frame']}.png"), 0)
        area = float((da > 0).mean())
        if area >= args.wide and r["split"] != "test":
            wide.append(dict(frame=r["frame"], split=r["split"], reviewer=r.get("reviewer", ""), da_area=round(area, 3)))
            continue
        rows.append(dict(frame=r["frame"], split=r["split"], curve=r["curve"], source="ego_v2_clean", label=r.get("decision", ""),
                         reviewer=r.get("reviewer", ""), section="", split_old=r.get("split_old", "")))
        for sub in SUBDIRS:
            d = os.path.join(args.out, r["split"], sub); os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(args.base, r["split"], sub, f"{r['frame']}.png"), d)
    for r in venue:
        split = "train" if r["split"] == "train" else "test_venue"
        rows.append(dict(frame=r["frame"], split=split, curve=r["curve"], source="venue_20261106", label=r["label"],
                         reviewer="", section=r["section"], split_old=""))
        for sub in SUBDIRS:
            d = os.path.join(args.out, split, sub); os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(args.venue, r["split"], sub, f"{r['frame']}.png"), d)
    for r in rows:
        r["use"] = 1
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "excluded_wide_da.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "split", "reviewer", "da_area"]); w.writeheader(); w.writerows(wide)
    count = lambda s, src=None: sum(r["split"] == s and (src is None or r["source"] == src) for r in rows)
    print(f"train {count('train')} (2차 {count('train', 'ego_v2_clean')} + 대회장 {count('train', 'venue_20261106')}), "
          f"val {count('val')}, test {count('test')}, test_venue {count('test_venue')} | 뺀 넓은 da 라벨 {len(wide)}장")


if __name__ == "__main__":
    main()
