#!/usr/bin/env python3
"""자동 판정 OK 라벨을 사람이 빠르게 검수한다 (split_ego_labels.py 다음 단계).

  make : <labels>/ok/ 의 프레임마다 [원본 | 오버레이(파랑 = da, 빨강 = ll)] 검수 이미지를
         <labels>/ok_review/<frame>.jpg 로 저장
  (사람) 이미지 뷰어로 넘겨 보며 OK가 아닌데 OK 판정된 이미지를 ok_review/ 에서 삭제
  apply: ok_review/ 에서 지워진 프레임을 ok/ -> fail/ 로 옮김
         (fail/status.csv 에 "REJECT: human review", 목록은 <labels>/rejected.csv 에 누적)

사용 예:
  .venv/Scripts/python scripts/review_ok_labels.py make  outputs/ego_labels_20261027_183234
  .venv/Scripts/python scripts/review_ok_labels.py apply outputs/ego_labels_20261027_183234
"""
import argparse
import csv
import os
import shutil

import cv2
import numpy as np

SUBDIRS = ("images", "da_masks", "ll_masks")


def overlay(img, da, ll):
    """make_model_compare_gif.overlay_masks 와 같은 색 (torch import 없이 쓰려고 복사)."""
    out = img.copy()
    out[da] = (0.5 * out[da].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    out[ll] = (0, 0, 255)
    return out


def read_status(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_status(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make(labels):
    rows = read_status(os.path.join(labels, "ok", "status.csv"))
    out_dir = os.path.join(labels, "ok_review")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(rows):
        stem = os.path.splitext(r["frame"])[0]
        img = cv2.imread(os.path.join(labels, "ok", "images", f"{stem}.png"))
        da = cv2.imread(os.path.join(labels, "ok", "da_masks", f"{stem}.png"), 0) > 0
        ll = cv2.imread(os.path.join(labels, "ok", "ll_masks", f"{stem}.png"), 0) > 0
        cv2.imwrite(os.path.join(out_dir, f"{stem}.jpg"), np.hstack([img, overlay(img, da, ll)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        if i % 500 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    print(f"검수 이미지 {len(rows)}장: {os.path.abspath(out_dir)}")


def apply(labels):
    ok_rows = read_status(os.path.join(labels, "ok", "status.csv"))
    fail_rows = read_status(os.path.join(labels, "fail", "status.csv"))
    fields = list(ok_rows[0].keys())
    review_dir = os.path.join(labels, "ok_review")
    keep, rejected = [], []
    for r in ok_rows:
        stem = os.path.splitext(r["frame"])[0]
        (keep if os.path.exists(os.path.join(review_dir, f"{stem}.jpg")) else rejected).append(r)
    for r in rejected:
        stem = os.path.splitext(r["frame"])[0]
        for sub in SUBDIRS:
            os.makedirs(os.path.join(labels, "fail", sub), exist_ok=True)
            shutil.move(os.path.join(labels, "ok", sub, f"{stem}.png"), os.path.join(labels, "fail", sub, f"{stem}.png"))
        fail_rows.append({**r, "status": "REJECT: human review"})
    write_status(os.path.join(labels, "ok", "status.csv"), keep, fields)
    write_status(os.path.join(labels, "fail", "status.csv"), sorted(fail_rows, key=lambda r: r["frame"]), fields)
    rej_path = os.path.join(labels, "rejected.csv")
    write_status(rej_path, sorted(read_status(rej_path) + rejected, key=lambda r: r["frame"]), fields)
    print(f"반려 {len(rejected)}장 ok -> fail. 남은 OK {len(keep)}, FAIL {len(fail_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "apply"])
    ap.add_argument("labels", help="split_ego_labels.py 로 ok/ fail/ 이 나뉜 폴더")
    args = ap.parse_args()
    (make if args.mode == "make" else apply)(args.labels)


if __name__ == "__main__":
    main()
