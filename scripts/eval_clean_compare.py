#!/usr/bin/env python3
"""1차와 2차 모델을 같은 test에서 원본 라벨, 정리 라벨(scripts/clean_labels.py)로 함께 채점하고 비교 몽타주를 만든다.

앙상블 규칙은 1차와 같다: da 확률 평균 >= 0.5, ll 과반 투표, 최종 da = da | ll. 점수는 그룹 전체 픽셀로 낸 IoU.
결과: <out>/summary.json, per_frame.csv, result_compare_clean.jpg (파랑 = da, 빨강 = ll)

GPU 사용률은 GPU_DUTY 환경변수(기본 90%)로 제한.

사용 예:
  .venv/Scripts/python scripts/eval_clean_compare.py --data dataset/ego_v2_clean --out outputs/eval_clean
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as T  # noqa: E402

warnings.filterwarnings("ignore")
RUNS = {"v1": {"hybridnets": "hybridnets_seed0", "tln_large": "tln_large_seed0", "yolop": "yolop_chbce_seed0", "yolopv2": "yolopv2_seed0"},
        "v2": {"hybridnets": "hybridnets_v2_seed0", "tln_large": "tln_large_v2_seed0", "yolop": "yolop_v2_seed0", "yolopv2": "yolopv2_v2_seed0"}}
FONT = "C:/Windows/Fonts/malgun.ttf"


def load(m, key, run_dir):
    model, rgb, norm, fwd, _ = T.ADAPTERS[key](m)
    w = os.path.join(run_dir, "best_da.pt")
    if key == "yolopv2":
        model = T.load_yolopv2_trainable(w, m.device)
    else:
        model.load_state_dict(torch.load(w, map_location="cpu"))
    return model.to(m.device).eval(), rgb, norm, fwd


def prep(img, rgb, norm):
    im = cv2.copyMakeBorder(img, T.PAD, T.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x = im.astype(np.float32) / 255
    if norm:
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None]


def predict(members, img, device):
    """-> {모델: (da, ll)} + ensemble"""
    out, das, lls = {}, [], []
    with torch.no_grad():
        for k, (model, rgb, norm, fwd) in members.items():
            a, b = fwd(model, prep(img, rgb, norm).to(device))
            a, b = T.crop(a)[0].float().cpu().numpy(), T.crop(b)[0].float().cpu().numpy()
            out[k] = (a >= .5, b >= .5)
            das.append(a)
            lls.append(b)
    ll = (np.stack(lls) >= .5).sum(0) >= len(lls) // 2 + 1
    out["ensemble"] = ((np.mean(das, 0) >= .5) | ll, ll)
    return out


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


class Meter:
    def __init__(self):
        self.i, self.u = 0, 0

    def add(self, a, b):
        self.i += int((a & b).sum())
        self.u += int((a | b).sum())

    def value(self):
        return round(self.i / max(self.u, 1), 4)


def groups_of(r):
    g = ["all", "v1" if r["old"] else "new"]
    if r["curve"] == "1":
        g.append("v1_curve" if r["old"] else "new_curve")
    if not r["old"] and r["decision"] in ("corrected", "trust"):
        g.append("new_" + r["decision"])
    return g


def montage(args, sets, m, rows_data, path):
    used, rows = set(), []

    def take(label, cond, q):
        fs = sorted((r for r in rows_data if cond(r)), key=lambda r: r["v2_ensemble_da_clean"])
        for r in fs[int(len(fs) * q):] + fs[:int(len(fs) * q)][::-1]:
            if r["frame"] not in used:
                used.add(r["frame"])
                rows.append((label, r))
                return
    old, new, cv = (lambda r: r["old"]), (lambda r: not r["old"]), (lambda r: r["curve"] == "1")
    take("기존 test · 직진 · 중앙값", lambda r: old(r) and not cv(r), .5)
    take("기존 test · 커브 · 중앙값", lambda r: old(r) and cv(r), .5)
    take("기존 test · 커브 · 하위 10%", lambda r: old(r) and cv(r), .1)
    take("추가 test · 직진 · 중앙값", lambda r: new(r) and not cv(r), .5)
    take("추가 test · 커브 · 중앙값", lambda r: new(r) and cv(r), .5)
    take("추가 test · 커브 · 하위 10%", lambda r: new(r) and cv(r), .1)
    take("추가 test · 보정 라벨 · 중앙값", lambda r: new(r) and r["decision"] == "corrected", .5)
    take("추가 test · 보정 라벨 · 하위 10%", lambda r: new(r) and r["decision"] == "corrected", .1)
    for i in range(3):
        take(f"기존 test · 가장 낮은 점수 {i + 1}", old, 0)
    for i in range(3):
        take(f"추가 test · 가장 낮은 점수 {i + 1}", new, 0)

    cols = [("원본", None), ("정리한 정답", None), ("1차 앙상블", ("v1", "ensemble")), ("2차 앙상블", ("v2", "ensemble")), ("2차 HybridNets", ("v2", "hybridnets")),
            ("2차 TwinLiteNet+ L", ("v2", "tln_large")), ("2차 YOLOP", ("v2", "yolop")), ("2차 YOLOPv2", ("v2", "yolopv2"))]
    tw, th, strip, head = 240, 135, 24, 62
    canvas = Image.new("RGB", (tw * len(cols), head + (th + strip) * len(rows)), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    ft, fc, fr = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 13)
    dr.text((8, 4), "LimoPro 1차 vs 2차 (정리한 라벨 기준, 학습에 안 쓴 프레임)   파랑 = da, 빨강 = ll, 칸 아래 = 프레임 IoU", font=ft, fill=(255, 255, 255))
    for c, (n, _) in enumerate(cols):
        dr.text((c * tw + 8, 36), n, font=fc, fill=(255, 255, 255))

    def ov(img, da, ll):
        v = img.astype(np.float32)
        v[da] = v[da] * .55 + np.array([255, 80, 0]) * .45
        v[ll] = (0, 0, 255)
        return cv2.resize(v.astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)
    d = os.path.join(args.data, "test")
    for i, (label, r) in enumerate(rows):
        n = r["frame"]
        img = cv2.imread(f"{d}/images/{n}.png")
        gd = cv2.imread(f"{d}/da_masks/{n}.png", 0) > 0
        gl = cv2.imread(f"{d}/ll_masks/{n}.png", 0) > 0
        with m.GpuDutyLimiter():
            p = {s: predict(sets[s], img, m.device) for s in sets}
        y = head + i * (th + strip)
        dr.rectangle((0, y, canvas.width, y + strip), fill=(45, 45, 45))
        src = {"v1": "1차 라벨", "trust": "승인 라벨", "corrected": "보정 라벨"}.get(r["decision"], r["decision"])
        dr.text((8, y + 3), f"{label}   frame {n[6:]}   {src}{' · ' + r['reviewer'] if r['reviewer'] else ''}", font=fr, fill=(255, 255, 255))
        for c, (_, key) in enumerate(cols):
            if key is None:
                t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA) if c == 0 else ov(img, gd, gl)
            else:
                t = ov(img, *p[key[0]][key[1]])
            canvas.paste(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)), (c * tw, y + strip))
            if key is not None:
                a, b = p[key[0]][key[1]]
                x0, y0 = c * tw, y + strip + th - 19
                dr.rectangle((x0, y0, x0 + 124, y0 + 19), fill=(0, 0, 0))
                dr.text((x0 + 4, y0 + 1), f"da {iou(a, gd):.2f} ll {iou(b, gl):.2f}", font=fr, fill=(255, 255, 255))
    canvas.save(path, quality=90)
    print(path, canvas.size, [(lab, r["frame"][6:]) for lab, r in rows], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/ego_v2_clean")
    ap.add_argument("--train-dir", default="outputs/train")
    ap.add_argument("--out", default="outputs/eval_clean")
    args = ap.parse_args()
    import make_model_compare_gif as m
    if not torch.cuda.is_available():
        torch.set_num_threads(8)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.data, "manifest.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["use"] == "1"]
    sets = {s: {k: load(m, k, os.path.join(args.train_dir, run)) for k, run in runs.items()} for s, runs in RUNS.items()}
    print(f"모델 로드 완료, test {len(rows)}장, device {m.device}", flush=True)

    limiter = m.GpuDutyLimiter()  # GPU_DUTY 기본 90%
    meters, out, t0 = {}, [], time.time()
    for idx, r in enumerate(rows):
        n = r["frame"]
        r["old"] = r["split_old"] == "test"
        img = cv2.imread(os.path.join(args.data, "test", "images", f"{n}.png"))
        clean_da = cv2.imread(os.path.join(args.data, "test", "da_masks", f"{n}.png"), 0) > 0
        orig_da = cv2.imread(os.path.join(r["label_dir"], "da_masks", f"{n}.png"), 0) > 0
        gl = cv2.imread(os.path.join(args.data, "test", "ll_masks", f"{n}.png"), 0) > 0
        row = dict(frame=n, old=r["old"], curve=r["curve"], decision=r["decision"], reviewer=r["reviewer"])
        for s, members in sets.items():
            with limiter:
                p = predict(members, img, m.device)
            for k, (da, ll) in p.items():
                for g in groups_of(r):
                    for lab, gt in (("orig", orig_da), ("clean", clean_da)):
                        meters.setdefault((s, k, g, lab, "da"), Meter()).add(da, gt)
                    meters.setdefault((s, k, g, "orig", "ll"), Meter()).add(ll, gl)
                if k == "ensemble":
                    row[f"{s}_ensemble_da_orig"] = round(iou(da, orig_da), 4)
                    row[f"{s}_ensemble_da_clean"] = round(iou(da, clean_da), 4)
                    row[f"{s}_ensemble_ll"] = round(iou(ll, gl), 4)
        out.append(row)
        if idx % 50 == 0:
            print(f"{idx + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for (s, k, g, lab, what), v in meters.items():
        res.setdefault(s, {}).setdefault(k, {}).setdefault(g, {})[f"{what}_{lab}"] = v.value()
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "per_frame.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("채점 완료", flush=True)
    montage(args, sets, m, out, os.path.join(args.out, "result_compare_clean.jpg"))


if __name__ == "__main__":
    main()
