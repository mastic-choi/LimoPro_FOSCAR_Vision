#!/usr/bin/env python3
"""사람이 검수한 ego lane 라벨(ok/)로 학습용 train / val / test 데이터셋을 만든다.

분할은 시간 구간 단위(연속 프레임이라 무작위로 나누면 거의 같은 장면이 train/test 양쪽에 들어가 점수가 부풀려짐):
  test : --test-range 구간 전체 (자연 비율 그대로, 줄이지 않음)
  val  : --val-range 구간 전체 (best 체크포인트 선택용)
  train: 나머지 검수 범위. 각 구간 앞뒤 --buffer 프레임은 어디에도 안 넣음
train만 줄인다:
  1. 비슷한 프레임 거르기: 마지막으로 남긴 프레임과 dHash 해밍거리 < --dhash 이면 버림
  2. 커브/직진 균형: ego da 중심선의 휨(2차 계수)·기울기로 커브성 판정, 직진을 커브 수 x --straight-ratio 까지 무작위 다운샘플

출력: <out>/{train,val,test}/{images,da_masks,ll_masks}/<frame>.png + manifest.csv

사용 예:
  .venv/Scripts/python -I scripts/build_train_dataset.py outputs/ego_labels_20261027_183234 dataset/ego_v1 --reviewed-max 4929
"""
import argparse
import csv
import os
import random
import shutil

import cv2
import numpy as np

SUBDIRS = ("images", "da_masks", "ll_masks")


def dhash(img, size=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def bend(da):
    """ego da 행별 중심 x를 아래->위로 2차 피팅. (|휨|, |기울기|), 화면 크기로 정규화."""
    h, w = da.shape
    ys = np.flatnonzero(da.any(1))
    if ys.size < 20:
        return 0.0, 0.0
    cx = np.array([np.flatnonzero(da[y]).mean() for y in ys])
    a2, a1, _ = np.polyfit((ys - ys.max()) / h, (cx - w / 2) / w, 2)
    return abs(a2), abs(a1)


def in_range(i, r, pad=0):
    return r[0] - pad <= i <= r[1] + pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="split_ego_labels.py / review_ok_labels.py 를 거친 폴더 (ok/ 사용)")
    ap.add_argument("out")
    ap.add_argument("--reviewed-max", type=int, required=True, help="사람 검수가 끝난 마지막 프레임 번호")
    ap.add_argument("--test-range", type=int, nargs=2, default=[4300, 4929])
    ap.add_argument("--val-range", type=int, nargs=2, default=[2000, 2299])
    ap.add_argument("--buffer", type=int, default=30)
    ap.add_argument("--dhash", type=int, default=6, help="train 비슷한 프레임 거르기 해밍거리 기준 (64비트 중)")
    ap.add_argument("--curve-bend", type=float, default=1.0)
    ap.add_argument("--curve-slope", type=float, default=0.4)
    ap.add_argument("--straight-ratio", type=float, default=1.0, help="직진 수 상한 = 커브 수 x 이 값")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.labels, "ok", "status.csv"), encoding="utf-8", newline="") as f:
        stems = sorted(os.path.splitext(r["frame"])[0] for r in csv.DictReader(f))
    idx = lambda s: int(s[6:12])
    stems = [s for s in stems if idx(s) <= args.reviewed_max]

    rows, last = [], None
    for s in stems:
        i = idx(s)
        if in_range(i, args.test_range):
            split = "test"
        elif in_range(i, args.val_range):
            split = "val"
        elif in_range(i, args.test_range, args.buffer) or in_range(i, args.val_range, args.buffer):
            continue
        else:
            split = "train"
        img = cv2.imread(os.path.join(args.labels, "ok", "images", f"{s}.png"))
        da = cv2.imread(os.path.join(args.labels, "ok", "da_masks", f"{s}.png"), 0) > 0
        b2, b1 = bend(da)
        curve = b2 >= args.curve_bend or b1 >= args.curve_slope
        h = dhash(img)
        dup = split == "train" and last is not None and int((h != last).sum()) < args.dhash
        if split == "train" and not dup:
            last = h
        rows.append(dict(frame=s, split=split, curve=int(curve), bend=round(b2, 3), slope=round(b1, 3), dup=int(dup)))

    train = [r for r in rows if r["split"] == "train" and not r["dup"]]
    cur = [r for r in train if r["curve"]]
    st = [r for r in train if not r["curve"]]
    n_st = min(len(st), int(round(len(cur) * args.straight_ratio)))
    keep_st = set(r["frame"] for r in random.Random(args.seed).sample(st, n_st))
    for r in rows:
        r["use"] = int(r["split"] != "train" or (not r["dup"] and (r["curve"] or r["frame"] in keep_st)))

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for r in rows:
        if not r["use"]:
            continue
        for sub in SUBDIRS:
            d = os.path.join(args.out, r["split"], sub)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(args.labels, "ok", sub, f"{r['frame']}.png"), d)
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tr_all = [r for r in rows if r["split"] == "train"]
    print(f"검수 OK {len(stems)}장 (≤ frame {args.reviewed_max}), 버퍼로 제외 {len(stems) - len(rows)}")
    print(f"train 후보 {len(tr_all)} -> 비슷한 프레임 제거 후 {len(train)} (커브 {len(cur)} / 직진 {len(st)})"
          f" -> 사용 커브 {len(cur)} + 직진 {n_st} = {len(cur) + n_st}")
    for sp in ("val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        print(f"{sp}: {len(rs)} (커브 {sum(r['curve'] for r in rs)})")


if __name__ == "__main__":
    main()
