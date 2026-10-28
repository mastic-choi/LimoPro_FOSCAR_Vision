#!/usr/bin/env python3
"""파인튜닝한 앙상블 멤버(train_seg.py 결과)로 아직 사람 라벨이 없는 프레임에 pseudo-label을 만든다.

대상 (--labels 폴더, split_ego_labels.py / review_ok_labels.py 를 거친 것):
  - fail/  : 기하 ego lane이 못 잡은 프레임(주로 커브) + 사람이 반려한 OK
  - ok/    : --reviewed-max 이후 아직 사람 검수 안 한 OK
  --exclude 구간(기본: dataset/ego_v1의 val/test 구간 + 버퍼)은 제외 -> 나중에 사람 GT로 평가에 써야 해서 학습에 섞으면 안 됨

앙상블: da = 멤버 확률 평균 >= 0.5, ll = 과반 투표(확률 >= 0.5), 저장 da = da | ll (라벨 관례).
검수 우선순위: 멤버끼리 da 마스크가 얼마나 다른지(쌍별 IoU 평균, 낮을수록 의견이 갈림) -> agree 오름차순.

출력 <out>/: images/ da_masks/ ll_masks/ (640x360 png), meta.csv(frame, source, da_agree, ll_agree, da_area)
  -> tools/ego_label_review_*.ipynb 로 사람 검수/SAM 보정

--eval-test: 같은 앙상블을 dataset/ego_v1 test 에서 평가만 하고 끝 (단일 모델 test.json 과 비교용).

사용 예:
  .venv/Scripts/python scripts/make_finetuned_pseudo_labels.py --eval-test
  .venv/Scripts/python scripts/make_finetuned_pseudo_labels.py --labels outputs/ego_labels_20261027_183234 \
      --reviewed-max 4929 --out outputs/pseudo_ft_v1
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as T  # noqa: E402

DEFAULT_RUNS = ["tln_large_seed0", "yolop_seed0", "hybridnets_seed0", "yolopv2_seed0"]


def load_member(m, run_dir):
    """run 폴더 이름(<model>[suffix]_seed<n>)에서 모델 종류를 찾아 best_da.pt 로드. -> (model, rgb, norm, fwd)"""
    name = os.path.basename(run_dir)
    key = next(k for k in sorted(T.ADAPTERS, key=len, reverse=True) if name.startswith(k))
    model, rgb, norm, fwd, _ = T.ADAPTERS[key](m)
    w = os.path.join(run_dir, "best_da.pt")
    if key == "yolopv2":
        model = T.load_yolopv2_trainable(w, m.device)
    else:
        model.load_state_dict(torch.load(w, map_location="cpu"))
    return model.to(m.device).eval(), rgb, norm, fwd


def prep(img, rgb, norm):
    """EgoDataset(valid)와 같은 전처리: 640x360 -> 위아래 12px 패딩."""
    img = cv2.copyMakeBorder(img, T.PAD, T.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    if norm:
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


@torch.no_grad()
def predict(members, imgs, device):
    """imgs: BGR 640x360 리스트 -> (da_probs[M,B,H,W], ll_probs[M,B,H,W]) numpy"""
    das, lls = [], []
    for model, rgb, norm, fwd in members:
        x = torch.stack([prep(im, rgb, norm) for im in imgs]).to(device)
        pda, pll = fwd(model, x)
        das.append(T.crop(pda).float().cpu().numpy())
        lls.append(T.crop(pll).float().cpu().numpy())
    return np.stack(das), np.stack(lls)


def ensemble(da_p, ll_p):
    da = da_p.mean(0) >= 0.5
    ll = (ll_p >= 0.5).sum(0) >= len(ll_p) // 2 + 1
    return da | ll, ll


def mean_pair_iou(masks):
    """masks[M,H,W] bool -> 멤버 쌍별 IoU 평균 (둘 다 비어 있으면 1)."""
    v = []
    for a, b in itertools.combinations(range(len(masks)), 2):
        u = (masks[a] | masks[b]).sum()
        v.append(1.0 if u == 0 else (masks[a] & masks[b]).sum() / u)
    return float(np.mean(v))


def in_ranges(i, ranges):
    return any(a <= i <= b for a, b in ranges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="outputs/train 아래 run 폴더 이름 (best_da.pt 사용)")
    ap.add_argument("--train-dir", default=None, help="기본: outputs/train")
    ap.add_argument("--labels", help="ego lane 라벨 폴더 (ok/ fail/)")
    ap.add_argument("--reviewed-max", type=int, help="사람 검수가 끝난 마지막 프레임 번호 (그 뒤 OK만 대상)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[1970, 2329, 4270, 4959],
                    help="제외 구간 쌍 (기본: ego_v1 val 2000~2299 / test 4300~4929 + 버퍼 30)")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-test", action="store_true", help="dataset/ego_v1 test에서 앙상블 평가만")
    ap.add_argument("--data", default=None, help="--eval-test 데이터 (기본: dataset/ego_v1)")
    args = ap.parse_args()

    import make_model_compare_gif as m  # ZLUDA 설정 포함 (여기서 import해야 CUDA 초기화)
    train_dir = args.train_dir or os.path.join(m.ROOT, "outputs", "train")
    members = [load_member(m, os.path.join(train_dir, r)) for r in args.runs]
    print(f"앙상블 멤버 {len(members)}: {args.runs}", flush=True)
    limiter = m.GpuDutyLimiter()

    if args.eval_test:
        data = args.data or os.path.join(m.ROOT, "dataset", "ego_v1")
        with open(os.path.join(data, "manifest.csv"), encoding="utf-8") as f:
            curve = {r["frame"]: r["curve"] == "1" for r in csv.DictReader(f) if r["split"] == "test"}
        res = {g: [T.IoU(), T.IoU()] for g in ("all", "curve", "straight")}
        names = sorted(curve)
        for k in range(0, len(names), args.batch):
            chunk = names[k:k + args.batch]
            imgs = [cv2.imread(os.path.join(data, "test", "images", f"{n}.png")) for n in chunk]
            with limiter:
                da_p, ll_p = predict(members, imgs, m.device)
            da, ll = ensemble(da_p, ll_p)
            for j, n in enumerate(chunk):
                gda = torch.from_numpy((cv2.imread(os.path.join(data, "test", "da_masks", f"{n}.png"), 0) > 0).astype(np.float32))
                gll = torch.from_numpy((cv2.imread(os.path.join(data, "test", "ll_masks", f"{n}.png"), 0) > 0).astype(np.float32))
                for g in ("all", "curve" if curve[n] else "straight"):
                    res[g][0].add(torch.from_numpy(da[j].astype(np.float32)), gda)
                    res[g][1].add(torch.from_numpy(ll[j].astype(np.float32)), gll)
        out = {g: {"da_iou": round(a.value(), 4), "ll_iou": round(b.value(), 4)} for g, (a, b) in res.items()}
        print("앙상블 test:", json.dumps(out), flush=True)
        with open(os.path.join(train_dir, "ensemble_test.json"), "w", encoding="utf-8") as f:
            json.dump({"runs": args.runs, "test": out}, f, indent=2)
        return

    assert args.labels and args.reviewed_max is not None and args.out, "--labels --reviewed-max --out 필요"
    ranges = list(zip(args.exclude[::2], args.exclude[1::2]))
    idx = lambda s: int(s[6:12])
    items = []  # (stem, source, src_dir)
    for sub in ("fail", "ok"):
        with open(os.path.join(args.labels, sub, "status.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                s = os.path.splitext(r["frame"])[0]
                if in_ranges(idx(s), ranges) or (sub == "ok" and idx(s) <= args.reviewed_max):
                    continue
                items.append((s, r["status"] if sub == "fail" else "OK (unreviewed)", os.path.join(args.labels, sub)))
    items.sort()
    print(f"대상 {len(items)}장 (fail {sum(1 for i in items if i[1] != 'OK (unreviewed)')}, "
          f"미검수 OK {sum(1 for i in items if i[1] == 'OK (unreviewed)')}), 제외 구간 {ranges}", flush=True)

    for sub in ("images", "da_masks", "ll_masks"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    rows, t0 = [], time.time()
    for k in range(0, len(items), args.batch):
        chunk = items[k:k + args.batch]
        imgs = [cv2.imread(os.path.join(d, "images", f"{s}.png")) for s, _, d in chunk]
        with limiter:
            da_p, ll_p = predict(members, imgs, m.device)
        da, ll = ensemble(da_p, ll_p)
        for j, (s, src, d) in enumerate(chunk):
            shutil.copy(os.path.join(d, "images", f"{s}.png"), os.path.join(args.out, "images"))
            cv2.imwrite(os.path.join(args.out, "da_masks", f"{s}.png"), da[j].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(args.out, "ll_masks", f"{s}.png"), ll[j].astype(np.uint8) * 255)
            rows.append(dict(frame=s, source=src, da_agree=round(mean_pair_iou(da_p[:, j] >= 0.5), 4),
                             ll_agree=round(mean_pair_iou(ll_p[:, j] >= 0.5), 4), da_area=round(float(da[j].mean()), 4)))
        if (k // args.batch) % 50 == 0:
            print(f"{k + len(chunk)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: r["da_agree"])  # 의견이 갈리는 프레임부터
    with open(os.path.join(args.out, "meta.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    a = np.array([r["da_agree"] for r in rows])
    print(f"완료 {len(rows)}장 -> {os.path.abspath(args.out)}  da_agree 중앙값 {np.median(a):.3f}, "
          f"< 0.8: {(a < 0.8).sum()}장, < 0.9: {(a < 0.9).sum()}장", flush=True)


if __name__ == "__main__":
    main()
