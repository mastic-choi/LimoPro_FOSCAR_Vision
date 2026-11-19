#!/usr/bin/env python3
"""릴리즈 전 주행 검사: 대회장 bag 프레임을 모델에 그대로 넣어 DA·차선이 안정적으로 나오는지 (사용자 11-18).

배경: multitask-v5 는 시험셋 점수는 v4 보다 좋았지만, 대회장 주행(신호등 진입 등)에서 DA·LL 이 통째로 0 이 되어
      차가 멈췄다. 시험셋은 고른 사진이라 이런 '전부 꺼짐'을 못 잡는다 → bag 전체 프레임으로 검사한다.

항목 (bag 마다, 후보 vs 기준 모델):
  1. 통째로 꺼짐: 화면 아래 절반 DA < 3 % 이고 LL < 0.3 % 인데 기준 모델은 DA > 30 % (= 도로가 보이는 장면)
  2. 가장 긴 연속 꺼짐 (초)
  3. 차 앞 비어 보임: 화면 맨 아래 띠(아래 1/6) DA < 20 % 인 프레임 (기준 모델은 50 % 넘음)
  4. 흔들림: 이웃 프레임 사이 아래 절반 DA IoU 의 5 % 분위 (낮을수록 프레임마다 출렁임)
통과 규칙 (모두 만족): 1 이 0, 2 가 0.3 s 미만, 3 이 기준 모델보다 bag 마다 5 프레임 넘게 많지 않음, 4 가 기준보다 0.05 넘게 낮지 않음.

사용: python bag_da_gate.py --cand <후보 best.pt> --base <기준 best.pt> --bundle <LimoPro_mt_v4 폴더> <bag 프레임 폴더>...
      bag 프레임 폴더 = bag_fork_frames.py --all 결과 (<폴더>/frames/*.jpg, frames.csv)
      --stride 2 이면 프레임 2장에 1장 (빠르게). 결과 표는 화면과 --out JSON 에.
"""
import argparse, csv, glob, json, os, sys

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--bundle", required=True, help="train_multitask.py 가 있는 LimoPro_mt_v4 폴더")
ap.add_argument("--stride", type=int, default=1); ap.add_argument("--out", default="")
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
runs = [os.path.abspath(r) for r in a.runs]
sys.path.insert(0, os.path.join(os.path.abspath(a.bundle), "scripts"))
import train_multitask as tm, make_model_compare_gif as m  # noqa: E402
os.chdir(m.ROOT)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
M = {"cand": tm.build_model(m, os.path.abspath(a.cand)).cuda().eval(), "base": tm.build_model(m, os.path.abspath(a.base)).cuda().eval()}


def prep(f):
    im = cv2.resize(cv2.imread(f), (640, 360), interpolation=cv2.INTER_AREA)
    im = cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114,) * 3)
    return ((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def run(fs):
    out = {k: {"da": [], "ll": [], "near": [], "mask": []} for k in M}
    for i in range(0, len(fs), 64):
        x = torch.from_numpy(np.stack([prep(f) for f in fs[i:i + 64]])).cuda()
        for k in M:
            *_, da, ll, _ = tm.heads(M[k], x)
            d = da.float()[:, tm.PAD:-tm.PAD] >= 0.5; l = ll.float()[:, tm.PAD:-tm.PAD] >= 0.5
            out[k]["da"] += d[:, 180:].float().mean((1, 2)).tolist(); out[k]["ll"] += l[:, 180:].float().mean((1, 2)).tolist()
            out[k]["near"] += d[:, 300:].float().mean((1, 2)).tolist()
            out[k]["mask"] += list(d[:, 180::4, ::4].cpu().numpy())   # 흔들림용 (1/4 크기)
    return out


def longest(mask, dt):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best * dt


report, ok_all = {}, True
for r in runs:
    fs = sorted(glob.glob(os.path.join(r, "frames", "*.jpg")))[::a.stride]
    rows = list(csv.DictReader(open(os.path.join(r, "frames.csv"))))
    dt = (float(rows[-1]["t"]) - float(rows[0]["t"])) / max(1, len(rows) - 1) * a.stride
    o = run(fs); res = {}
    sees_road = {k: np.array(o["base" if k == "cand" else "cand"]["da"]) > 0.3 for k in M}
    for k in M:
        da, ll, near = (np.array(o[k][n]) for n in ("da", "ll", "near"))
        off = (da < 0.03) & (ll < 0.003) & sees_road[k]
        other_near = np.array(o["base" if k == "cand" else "cand"]["near"])
        ms = o[k]["mask"]
        ious = [((p & q).sum() / max(1, (p | q).sum())) for p, q in zip(ms[:-1], ms[1:]) if (p | q).sum() > 200]
        res[k] = {"frames": len(fs), "off": int(off.sum()), "off_longest_s": round(longest(off, dt), 2),
                  "near_empty": int(((near < 0.2) & (other_near > 0.5)).sum()),
                  "iou_p5": round(float(np.percentile(ious, 5)), 3) if ious else None,
                  "first_off_t": round(float(rows[::a.stride][int(np.argmax(off))]["t"]), 2) if off.any() else None}
    c, b = res["cand"], res["base"]
    ok = c["off"] == 0 and c["off_longest_s"] < 0.3 and c["near_empty"] <= b["near_empty"] + 5 and (c["iou_p5"] or 0) >= (b["iou_p5"] or 0) - 0.05
    ok_all &= ok; res["pass"] = ok; report[os.path.basename(r)] = res
    print(f"{os.path.basename(r):22s} {'통과' if ok else '불합격'} | 꺼짐 후보 {c['off']} (최장 {c['off_longest_s']}s, 처음 {c['first_off_t']}s) / 기준 {b['off']}"
          f" | 차 앞 비어 보임 후보 {c['near_empty']} / 기준 {b['near_empty']} | 흔들림(IoU 5%) 후보 {c['iou_p5']} / 기준 {b['iou_p5']}", flush=True)
print("전체:", "통과" if ok_all else "불합격 (릴리즈하지 말 것)")
if a.out:
    json.dump({"cand": a.cand, "base": a.base, "pass": ok_all, "runs": report}, open(a.out, "w"), indent=1, ensure_ascii=False)
