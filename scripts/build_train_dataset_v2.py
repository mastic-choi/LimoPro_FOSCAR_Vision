#!/usr/bin/env python3
"""ego_v1에 사람 검수한 pseudo-label(tools/ego_label_review_*.ipynb 결과)을 더해 2차 학습 데이터셋을 만든다.

val/test는 ego_v1 그대로 복사한다(1차 모델과 같은 기준으로 비교하려고). train만 늘린다:
  1. 후보: ego_v1 train 전부(이미 걸러짐) + 검수 결과의 trust/corrected (discard 제외)
  2. 비슷한 프레임 거르기: 프레임 순서대로 마지막으로 남긴 프레임과 dHash 해밍거리 < --dhash 이면 버림 (ego_v1 프레임은 항상 남김)
  3. 커브/직진 균형: 쉬운 직진만 커브 + 어려운 프레임 수 x --straight-ratio 까지 무작위 다운샘플
     어려운 프레임(기존 파이프라인 FAIL/사람 반려 또는 사람이 보정함)은 끊긴 차선 같은 약점 장면일 수 있어서 직진이어도 남김

출력: <out>/{train,val,test}/{images,da_masks,ll_masks}/<frame>.png + manifest.csv

사용 예:
  .venv/Scripts/python -I scripts/build_train_dataset_v2.py dataset/ego_v2 --base dataset/ego_v1 \
      --reviews outputs/pseudo_ft_v1_split/pseudo_ft_v1_허재_review outputs/pseudo_ft_v1_split/pseudo_ft_v1_최준수_review
"""
import argparse
import csv
import os
import random
import shutil

import cv2

from build_train_dataset import SUBDIRS, bend, dhash, in_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="dataset/ego_v1")
    ap.add_argument("--reviews", nargs="+", required=True, help="검수 결과 폴더 (review.csv images/ da_masks/ ll_masks/)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[1970, 2329, 4270, 4959], help="val/test 구간 + 버퍼 (섞이면 중단)")
    ap.add_argument("--dhash", type=int, default=6)
    ap.add_argument("--curve-bend", type=float, default=1.0)
    ap.add_argument("--curve-slope", type=float, default=0.4)
    ap.add_argument("--straight-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.base, "manifest.csv"), encoding="utf-8", newline="") as f:
        base = [r for r in csv.DictReader(f) if r["use"] == "1"]
    items = {}  # frame -> (폴더, 출처, 판정, 검수자)
    for r in base:
        if r["split"] == "train":
            items[r["frame"]] = (os.path.join(args.base, "train"), "v1", "v1", "")
    ranges = list(zip(args.exclude[::2], args.exclude[1::2]))
    for rev in args.reviews:
        who = os.path.basename(os.path.normpath(rev)).replace("pseudo_ft_v1_", "").replace("_review", "")
        with open(os.path.join(rev, "review.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["decision"] == "discard":
                    continue
                s = r["frame"]
                assert s not in items, f"{s}: 프레임 중복 ({who})"
                assert not any(in_range(int(s[6:12]), x) for x in ranges), f"{s}: val/test 구간 프레임이 섞임 ({who})"
                items[s] = (rev, r["source"], r["decision"], who)

    rows, last = [], None
    for s in sorted(items):
        d, source, decision, who = items[s]
        img = cv2.imread(os.path.join(d, "images", f"{s}.png"))
        da = cv2.imread(os.path.join(d, "da_masks", f"{s}.png"), 0) > 0
        b2, b1 = bend(da)
        h = dhash(img)
        dup = decision != "v1" and last is not None and int((h != last).sum()) < args.dhash
        if not dup:
            last = h
        hard = decision == "corrected" or source.startswith(("FAIL", "REJECT"))
        rows.append(dict(frame=s, split="train", curve=int(b2 >= args.curve_bend or b1 >= args.curve_slope),
                         bend=round(b2, 3), slope=round(b1, 3), dup=int(dup), use=0, source=source, decision=decision,
                         reviewer=who, hard=int(hard)))

    alive = [r for r in rows if not r["dup"]]
    must = [r for r in alive if r["curve"] or r["hard"] or r["decision"] == "v1"]
    easy = [r for r in alive if not (r["curve"] or r["hard"] or r["decision"] == "v1")]
    keep = random.Random(args.seed).sample(easy, min(len(easy), int(round(len(must) * args.straight_ratio))))
    for r in must + keep:
        r["use"] = 1

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for r in rows:
        if r["use"]:
            for sub in SUBDIRS:
                d = os.path.join(args.out, "train", sub)
                os.makedirs(d, exist_ok=True)
                shutil.copy(os.path.join(items[r["frame"]][0], sub, f"{r['frame']}.png"), d)
    for sp in ("val", "test"):
        shutil.copytree(os.path.join(args.base, sp), os.path.join(args.out, sp))
    extra = [dict(frame=r["frame"], split=r["split"], curve=r["curve"], bend=r["bend"], slope=r["slope"], dup=r["dup"],
                  use=r["use"], source="v1", decision="v1", reviewer="", hard=0) for r in base if r["split"] != "train"]
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + extra)

    use = [r for r in rows if r["use"]]
    print(f"후보 {len(rows)} (v1 {sum(r['decision'] == 'v1' for r in rows)}) -> 비슷한 프레임 제거 후 {len(alive)} -> 사용 {len(use)}")
    print(f"  v1 {sum(r['decision'] == 'v1' for r in use)}, 커브 {sum(r['curve'] for r in use)}, "
          f"어려운 프레임 {sum(r['hard'] for r in use)}, 쉬운 직진 {len(keep)}/{len(easy)}")
    for who in sorted({r["reviewer"] for r in use if r["reviewer"]}):
        print(f"  {who}: {sum(r['reviewer'] == who for r in use)}장")
    print(f"val {sum(r['split'] == 'val' for r in extra)}, test {sum(r['split'] == 'test' for r in extra)} (ego_v1 그대로)")


if __name__ == "__main__":
    main()
