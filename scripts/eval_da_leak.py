#!/usr/bin/env python3
"""앙상블의 da가 차선 밖으로 얼마나 새는지 재고, 라벨 없는 다른 촬영 프레임을 몽타주로 확인한다.

--split: 라벨 있는 test에서 프레임별 침범 비율 = (GT da를 --tol px 넓힌 영역 밖의 예측 da) / (GT da 면적)
         near = 화면 아래쪽 절반(차 바로 앞)만. 결과: <out>/<split>_leak.csv, 가장 많이 샌 프레임 몽타주
         (흰 선 = GT 경계, 빨강 = 침범, 노랑 = 놓침). 전체 침범 비율 순으로 정렬.
--frames: 라벨 없음 -> 모델 간 일치도(da_agree)만 재고 몽타주(파랑 = da, 빨강 = ll). 위 = 고르게 뽑은 프레임, 아래 = 일치도 최저.
         입력이 640x360이 아니면 가로 640으로 줄이고 가운데를 360으로 자름(학습 입력과 같은 크기).

주의: test는 기존 파이프라인을 통과한 OK 프레임만 있어서 차선이 심하게 끊긴 장면은 적을 수 있음(침범이 낮게 나올 수 있음).

사용 예:
  .venv/Scripts/python scripts/eval_da_leak.py --split test --frames D:/fine-tune/LimoPro_frames_20261023 --out outputs/leak_v1
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_finetuned_pseudo_labels as P  # noqa: E402

RUNS = ["tln_large_seed0", "yolop_chbce_seed0", "hybridnets_seed0", "yolopv2_seed0"]
TILE = (320, 180)


def label(t, text):
    t = cv2.resize(t, TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TILE[0], 20), (0, 0, 0), -1)
    cv2.putText(t, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def grid(tiles, cols=6):
    while len(tiles) % cols:
        tiles.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def to_input(img):
    h, w = img.shape[:2]
    if w != 640:
        img = cv2.resize(img, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
    top = (img.shape[0] - 360) // 2
    return img[top:top + 360]


def run_split(members, m, limiter, args):
    d = os.path.join(args.data, args.split)
    names = sorted(os.listdir(os.path.join(d, "images")))
    kernel = np.ones((2 * args.tol + 1, 2 * args.tol + 1), np.uint8)
    rows, keep = [], {}
    for k in range(0, len(names), args.batch):
        chunk = names[k:k + args.batch]
        imgs = [cv2.imread(os.path.join(d, "images", n)) for n in chunk]
        with limiter:
            da_p, ll_p = P.predict(members, imgs, m.device)
        da, _ = P.ensemble(da_p, ll_p)
        for j, n in enumerate(chunk):
            gt = cv2.imread(os.path.join(d, "da_masks", n), 0) > 0
            leak = da[j] & (cv2.dilate(gt.astype(np.uint8), kernel) == 0)
            area = max(int(gt.sum()), 1)
            rows.append(dict(frame=n[:-4], leak=round(leak.sum() / area, 4),
                             leak_near=round(leak[180:].sum() / area, 4),
                             da_agree=round(P.mean_pair_iou(da_p[:, j] >= 0.5), 4)))
            keep[n[:-4]] = (leak, da[j] & ~leak)  # 몽타주용
    rows.sort(key=lambda r: -r["leak"])
    with open(os.path.join(args.out, f"{args.split}_leak.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["leak"] for r in rows])
    summary = dict(frames=len(rows), tol_px=args.tol, mean=round(float(a.mean()), 4), median=round(float(np.median(a)), 4),
                   p90=round(float(np.percentile(a, 90)), 4), max=round(float(a.max()), 4),
                   over_1pct=int((a > 0.01).sum()), over_5pct=int((a > 0.05).sum()),
                   near_mean=round(float(np.mean([r["leak_near"] for r in rows])), 4),
                   da_agree_median=round(float(np.median([r["da_agree"] for r in rows])), 4))
    tiles = []
    for r in rows[:args.worst]:
        n = r["frame"]
        t = cv2.imread(os.path.join(d, "images", f"{n}.png"))
        gt = (cv2.imread(os.path.join(d, "da_masks", f"{n}.png"), 0) > 0).astype(np.uint8)
        leak, inside = keep[n]
        t[gt.astype(bool) & ~inside] = (0, 255, 255)
        t[leak] = (0, 0, 255)
        cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (255, 255, 255), 1)
        tiles.append(label(t, f"{n[6:]} leak {r['leak'] * 100:.1f}%"))
    cv2.imwrite(os.path.join(args.out, f"{args.split}_worst.jpg"), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return summary


def run_frames(members, m, limiter, args):
    ps = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")) + glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    rows, tiles = [], {}
    even = set(np.linspace(0, len(ps) - 1, args.montage // 2).astype(int).tolist())
    for k in range(0, len(ps), args.batch):
        chunk = ps[k:k + args.batch]
        imgs = [to_input(cv2.imread(p)) for p in chunk]
        with limiter:
            da_p, ll_p = P.predict(members, imgs, m.device)
        da, ll = P.ensemble(da_p, ll_p)
        for j, p in enumerate(chunk):
            agree = P.mean_pair_iou(da_p[:, j] >= 0.5)
            t = imgs[j].copy()
            t[da[j]] = (0.5 * t[da[j]].astype(np.float32) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
            t[ll[j]] = (0, 0, 255)
            s = os.path.splitext(os.path.basename(p))[0]
            rows.append(dict(frame=s, da_agree=round(agree, 4), da_area=round(float(da[j].mean()), 4)))
            tiles[s] = label(t, f"{s[6:]} agree {agree:.2f}")
    with open(os.path.join(args.out, "frames_agree.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["da_agree"] for r in rows])
    pick = [rows[i]["frame"] for i in sorted(even)]
    pick += [r["frame"] for r in sorted(rows, key=lambda r: r["da_agree"]) if r["frame"] not in pick][:args.montage - len(pick)]
    cv2.imwrite(os.path.join(args.out, "frames_montage.jpg"), grid([tiles[s] for s in pick]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dict(frames=len(rows), da_agree_median=round(float(np.median(a)), 4), under_0_9=int((a < 0.9).sum()),
                under_0_8=int((a < 0.8).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--split", help="라벨 있는 split (예: test)")
    ap.add_argument("--data", default="dataset/ego_v1")
    ap.add_argument("--frames", help="라벨 없는 프레임 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=int, default=5, help="경계 허용 오차 px (라벨 경계 흔들림은 침범으로 안 침)")
    ap.add_argument("--worst", type=int, default=18)
    ap.add_argument("--montage", type=int, default=36)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import make_model_compare_gif as m  # ZLUDA 설정 포함 (여기서 import해야 CUDA 초기화)
    os.makedirs(args.out, exist_ok=True)
    members = [P.load_member(m, os.path.join(m.ROOT, "outputs", "train", r)) for r in args.runs]
    limiter = m.GpuDutyLimiter()
    res = {"runs": args.runs}
    if args.split:
        res[args.split] = run_split(members, m, limiter, args)
        print(args.split, json.dumps(res[args.split]), flush=True)
    if args.frames:
        res["frames"] = dict(run_frames(members, m, limiter, args), path=args.frames)
        print("frames", json.dumps(res["frames"]), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
