#!/usr/bin/env python3
"""manifest의 학습/평가 라벨을 정리해 새 데이터셋으로 저장한다. 원본 라벨은 바꾸지 않는다.

da만 정리하고 ll은 그대로 둔다. 모델 출력은 쓰지 않는다(평가 점수가 부풀려지지 않게 라벨 모양과 이미지 색만 사용):
  1. 흰 선 포함: 양옆 또는 위아래가 큰 da 덩어리로 막힌 곳의 흰색(저채도, 고명도) 픽셀을 da에 넣는다. 흰 선 위도 달릴 수 있는 곳이다.
  2. 작은 구멍 메우기: SAM 구멍이 많은 프레임만 5x5 closing, 그 뒤 모든 프레임에서 --max-hole 이하 구멍을 메운다.
     A/B 갈림길 가운데 섬처럼 큰 구멍은 원래 비어 있어야 해서 남긴다.
  3. 작은 조각 지우기: --max-frag 미만이고 ll과 닿지 않는 조각을 지운다.
마지막에 라벨 관례대로 da = da | ll.

출력: <out>/{train,val,test}/{images,da_masks,ll_masks}/<frame>.png, manifest.csv(정리 전후 수치 추가)

사용 예:
  .venv/Scripts/python -I scripts/clean_labels.py outputs/train_v2_foscar/manifest_ego_v2_ys_split.csv dataset/ego_v2_clean \
      --sources dataset/ego_v1/test dataset/ego_v1/val dataset/ego_v1/train outputs/pseudo_ft_v1_split/pseudo_ft_v1_허재_review ...
"""
import argparse
import csv
import os
import shutil

import cv2
import numpy as np

KER3, KER5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
MIN_BASE = 2000  # 내 차로로 보는 da 덩어리의 최소 넓이 px


def hole_labels(u):
    """u(0/1)의 안쪽 구멍을 표시한 (bool 마스크, 라벨, 통계). 화면 가장자리와 이어진 빈 곳은 구멍이 아니다."""
    h, w = u.shape
    pad = cv2.copyMakeBorder(1 - u, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    inner = (pad[1:-1, 1:-1] == 1).astype(np.uint8)
    return cv2.connectedComponentsWithStats(inner, connectivity=4)


def noise(da):
    """(구멍 비율, 가장 큰 조각을 뺀 조각 비율)"""
    u = da.astype(np.uint8)
    n, _, st, _ = hole_labels(u)
    hole = st[1:, cv2.CC_STAT_AREA].sum() if n > 1 else 0
    n, _, st, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    area = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    return hole / max(da.sum(), 1), 1 - area.max() / max(area.sum(), 1)


def enclosed(da):
    """같은 행의 양옆 또는 같은 열의 위아래에 da가 있는 곳"""
    row = np.maximum.accumulate(da, 1) & np.maximum.accumulate(da[:, ::-1], 1)[:, ::-1]
    col = np.maximum.accumulate(da, 0) & np.maximum.accumulate(da[::-1], 0)[::-1]
    return row | col


def white_pixels(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 70) & (hsv[..., 2] > 150)


def clean(img, da, ll, max_hole, max_frag):
    """-> (정리한 da, 수치 dict)"""
    u = da | ll
    # 울타리는 ll을 뺀 큰 da 덩어리만: 옆 차로 ll까지 넣으면 옆 차로 횡단보도와 트랙 밖 흰 바닥이 들어온다
    n, lab, st, _ = cv2.connectedComponentsWithStats((da & ~ll).astype(np.uint8), connectivity=8)
    base = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_BASE])
    fence = enclosed(base)
    white = cv2.dilate((white_pixels(img) & fence).astype(np.uint8), KER3) > 0  # 흰 선 가장자리까지
    add = white & ~u & fence
    u = u | add
    hole0, frag0 = noise(u)
    x = u.astype(np.uint8)
    if hole0 > .005 or frag0 > .01:
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, KER5)
    n, lab, st, _ = hole_labels(x)
    small = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] <= max_hole]
    filled = int(st[small, cv2.CC_STAT_AREA].sum()) if small else 0
    x[np.isin(lab, small)] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
    touching = set(np.unique(lab[ll]).tolist())
    biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
    drop = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < max_frag and i not in touching and i != biggest]
    removed = int(st[drop, cv2.CC_STAT_AREA].sum()) if drop else 0
    x[np.isin(lab, drop)] = 0
    out = (x > 0) | ll
    hole1, frag1 = noise(out)
    return out, dict(white_added=int(add.sum()), holes_filled=filled, frags_removed=removed, hole_before=round(float(hole0), 5),
                     frag_before=round(float(frag0), 5), hole_after=round(float(hole1), 5), frag_after=round(float(frag1), 5),
                     da_before=int(da.sum()), da_after=int(out.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="frame, split, use 열이 있는 csv")
    ap.add_argument("out")
    ap.add_argument("--sources", nargs="+", required=True, help="라벨 폴더. 앞쪽 폴더를 먼저 찾되, 검수자 이름이 든 폴더를 우선")
    ap.add_argument("--max-hole", type=int, default=300, help="메울 구멍의 최대 넓이 px")
    ap.add_argument("--max-frag", type=int, default=300, help="지울 조각의 최대 넓이 px")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for r in rows:
        if r["use"] != "1":
            continue
        s = r["frame"]
        src = [d for d in args.sources if r["reviewer"] and r["reviewer"] in os.path.basename(os.path.normpath(d))] or args.sources
        d = next((d for d in src if os.path.exists(os.path.join(d, "da_masks", f"{s}.png"))), None)
        assert d, f"{s}: 라벨을 찾을 수 없음"
        img = cv2.imread(os.path.join(d, "images", f"{s}.png"))
        da = cv2.imread(os.path.join(d, "da_masks", f"{s}.png"), 0) > 0
        ll = cv2.imread(os.path.join(d, "ll_masks", f"{s}.png"), 0) > 0
        out, info = clean(img, da, ll, args.max_hole, args.max_frag)
        r.update(info, label_dir=d)
        base = os.path.join(args.out, r["split"])
        for sub in ("images", "da_masks", "ll_masks"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        shutil.copy(os.path.join(d, "images", f"{s}.png"), os.path.join(base, "images"))
        shutil.copy(os.path.join(d, "ll_masks", f"{s}.png"), os.path.join(base, "ll_masks"))
        cv2.imwrite(os.path.join(base, "da_masks", f"{s}.png"), out.astype(np.uint8) * 255)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    used = [r for r in rows if r["use"] == "1"]
    for sp in ("train", "val", "test"):
        x = [r for r in used if r["split"] == sp]
        print(f"{sp} {len(x)}장: 흰 선 추가 {sum(r['white_added'] > 0 for r in x)}장, 구멍 메움 {sum(r['holes_filled'] > 0 for r in x)}장, "
              f"조각 지움 {sum(r['frags_removed'] > 0 for r in x)}장, 구멍 1% 초과 {sum(r['hole_before'] > .01 for r in x)} -> {sum(r['hole_after'] > .01 for r in x)}장")


if __name__ == "__main__":
    main()
