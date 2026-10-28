#!/usr/bin/env python3
"""make_finetuned_pseudo_labels.py 결과를 6x6(기본) 몽타주 한 장으로. 파랑 = da, 빨강 = ll.

쉬운 장면만 고르지 않도록 meta.csv의 da_agree(모델 간 일치도) 순으로 3등분해서
어려운 / 중간 / 쉬운 구간에서 같은 수씩 고르게 뽑는다(행 순서도 이 순서: 위 = 어려움).
칸마다 프레임 번호, 원래 판정(FAIL 사유 / 미검수 OK), 일치도를 적는다.

사용 예:
  .venv/Scripts/python -I scripts/make_pseudo_montage.py outputs/pseudo_ft_v1 outputs/montages/pseudo_ft_v1_6x6.jpg
"""
import argparse
import csv
import os

import cv2
import numpy as np

SHORT = {"FAIL: no lane pair": "no pair", "FAIL: lane lost": "lost", "REJECT: human review": "rejected",
         "OK (unreviewed)": "OK unrev"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pseudo", help="pseudo-label 폴더 (images/ da_masks/ ll_masks/ meta.csv)")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, nargs=2, default=[320, 180])
    args = ap.parse_args()

    with open(os.path.join(args.pseudo, "meta.csv"), encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["da_agree"]))
    n = args.rows * args.cols
    bands = np.array_split(np.arange(len(rows)), 3)  # 어려움 / 중간 / 쉬움
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    pick = []
    for band, k in zip(bands, per):
        if len(band):
            pick += [rows[i] for i in np.unique(band[np.linspace(0, len(band) - 1, k).astype(int)])]  # 장수가 적으면 중복 없이

    tw, th = args.tile
    tiles = []
    for r in pick:
        s = r["frame"]
        img = cv2.imread(os.path.join(args.pseudo, "images", f"{s}.png"))
        da = cv2.imread(os.path.join(args.pseudo, "da_masks", f"{s}.png"), 0) > 0
        ll = cv2.imread(os.path.join(args.pseudo, "ll_masks", f"{s}.png"), 0) > 0
        img[da] = (0.5 * img[da].astype(np.float32) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
        img[ll] = (0, 0, 255)
        t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        src = r["source"].split(" (")[0] if r["source"].startswith("FAIL: few rows") else r["source"]
        text = f"{s[6:]} {SHORT.get(src, src.replace('FAIL: ', ''))} agree {float(r['da_agree']):.2f}"
        cv2.rectangle(t, (0, 0), (tw, 20), (0, 0, 0), -1)
        cv2.putText(t, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) < n:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i * args.cols:(i + 1) * args.cols]) for i in range(args.rows)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(pick)}장 -> {os.path.abspath(args.out)} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
