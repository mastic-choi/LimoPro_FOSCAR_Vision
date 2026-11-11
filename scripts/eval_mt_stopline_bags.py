#!/usr/bin/env python3
"""11/07 실차 rosbag 프레임에서 다중 작업 모델의 흰 가로선(정지선·미션 시작선) 출력을 뽑아 본다.

입력: dataset/bags_20261107_analysis/<세션>/frames/*.jpg (640x360, bag에서 뽑은 프레임)
프레임마다: 가로선 확률 > --thr 픽셀 수, 최대 확률, 가장 가까운(화면 아래) 가로선까지 거리(m, 카메라 IPM, 기울기 4°)
출력 <out>/
  stopline_bags.csv          프레임별 값
  <세션>.mp4                 가로선(초록)·차선(빨강) 오버레이 영상 (--video)
  <세션>_timeline.png        프레임별 가로선 픽셀 수
  montage_top.jpg            가로선이 가장 크게 잡힌 장면들(세션마다 고르게, 서로 떨어진 프레임)
  montage_random.jpg         무작위 장면 (오탐 확인용)
사용: python scripts/eval_mt_stopline_bags.py <best.pt> [--video] [--out outputs/train_mt_v2/stopline_bags]
"""
import argparse, csv, glob, os, random, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_multitask as tm
import make_model_compare_gif as m

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--bags", default="dataset/bags_20261107_analysis")
ap.add_argument("--out", default="outputs/train_mt_v2/stopline_bags")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--video", action="store_true")
a = ap.parse_args()
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.ROOT, "dataset", "track_label_20261106", "code"))
os.environ.setdefault("TRACK_LABEL_WORK", os.path.join(m.ROOT, "dataset", "track_label_20261106"))
import cam  # 카메라 IPM (limo.yaml 내부 보정, 높이 0.19m)

os.makedirs(a.out, exist_ok=True)
model = tm.build_model(m, a.ckpt).to(m.device).eval()
amp = m.device.type == "cuda" and "ZLUDA" not in torch.cuda.get_device_name(0)
PITCH = 4.0


def prep(img):
    x = cv2.copyMakeBorder(img, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def nearest_dist(sl):
    """화면에서 가장 아래(가까운) 가로선 행의 가운데 점 -> 바닥 거리(m, 뒷바퀴 축 기준 전방)."""
    ys, xs = np.nonzero(sl)
    if len(ys) < 30:
        return None
    y = np.percentile(ys, 97)
    row = xs[ys >= y - 2]
    g = cam.ipm(np.array([[float(np.median(row)), float(y)]]), PITCH)[0]
    return round(float(g[0]), 2) if np.isfinite(g[0]) else None


def overlay(img, da, ll, sl, txt):
    o = img.copy()
    o[da] = (o[da] * 0.75 + np.array([60, 60, 0]) * 0.25).astype(np.uint8)
    o[ll] = (0, 0, 255)
    o[sl] = (o[sl] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
    cv2.putText(o, txt, (6, 18), 0, 0.5, (0, 0, 0), 3)
    cv2.putText(o, txt, (6, 18), 0, 0.5, (255, 255, 255), 1)
    return o


rows, keep = [], {}
sessions = sorted(d for d in os.listdir(a.bags) if os.path.isdir(os.path.join(a.bags, d, "frames")))
with torch.no_grad():
    for ses in sessions:
        fs = sorted(glob.glob(os.path.join(a.bags, ses, "frames", "*.jpg")))
        vw = None
        if a.video:
            vw = cv2.VideoWriter(os.path.join(a.out, f"{ses}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
        for i in range(0, len(fs), a.batch):
            chunk = fs[i:i + a.batch]
            imgs = [cv2.imread(f) for f in chunk]
            x = torch.stack([prep(im) for im in imgs]).to(m.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _, _, _, pda, pll, psl = tm.heads(model, x)
            pda, pll, psl = (tm.ts.crop(p).float().cpu().numpy() for p in (pda, pll, psl))
            for k, f in enumerate(chunk):
                sl = psl[k] > a.thr
                d = nearest_dist(sl)
                r = dict(session=ses, frame=os.path.basename(f)[:-4], sl_px=int(sl.sum()),
                         sl_max=round(float(psl[k].max()), 3), dist_m=d)
                rows.append(r)
                if vw is not None or sl.sum() > 0:
                    ov = overlay(imgs[k], pda[k] > 0.5, pll[k] > 0.5, sl,
                                 f"{ses[:8]} {r['frame']} sl {r['sl_px']}px max {r['sl_max']:.2f}" + (f" {d:.2f}m" if d else ""))
                    if vw is not None:
                        vw.write(ov)
                    if sl.sum() > 0:
                        keep[(ses, r["frame"])] = cv2.imencode(".jpg", cv2.resize(ov, (480, 270)))[1]
                    elif random.random() < 0.02:
                        keep[(ses, r["frame"])] = cv2.imencode(".jpg", cv2.resize(ov, (480, 270)))[1]
                elif random.random() < 0.02:
                    keep[(ses, r["frame"])] = cv2.imencode(".jpg", cv2.resize(overlay(imgs[k], pda[k] > 0.5, pll[k] > 0.5, sl, f"{ses[:8]} {r['frame']} sl 0"), (480, 270)))[1]
        if vw is not None:
            vw.release()
        rs = [r for r in rows if r["session"] == ses]
        print(f"{ses}: {len(rs)} frames, with stop line {sum(r['sl_px'] > 0 for r in rs)} "
              f"(>=200px {sum(r['sl_px'] >= 200 for r in rs)})", flush=True)
        # 타임라인
        h, w = 160, max(400, len(rs))
        tl = np.full((h, w, 3), 255, np.uint8)
        mx = max(1, max(r["sl_px"] for r in rs))
        for j, r in enumerate(rs):
            hh = int(r["sl_px"] / mx * (h - 20))
            if hh:
                cv2.line(tl, (j * w // len(rs), h - 1), (j * w // len(rs), h - 1 - hh), (0, 160, 0), 1)
        cv2.putText(tl, f"{ses} stop-line px per frame (max {mx})", (5, 15), 0, 0.45, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(a.out, f"{ses}_timeline.png"), tl)

with open(os.path.join(a.out, "stopline_bags.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)


def montage(sel, path, cols=4):
    tiles = [cv2.imdecode(keep[s], 1) for s in sel if s in keep]
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


# 가장 크게 잡힌 장면: 세션마다 서로 15프레임 이상 떨어진 상위 장면
top = []
for ses in sessions:
    rs = sorted([r for r in rows if r["session"] == ses and r["sl_px"] > 0], key=lambda r: -r["sl_px"])
    picked = []
    for r in rs:
        fi = int(r["frame"])
        if all(abs(fi - p) > 45 for p in picked):
            picked.append(fi); top.append((ses, r["frame"]))
        if len(picked) >= 4:
            break
montage(top, os.path.join(a.out, "montage_top.jpg"))
random.seed(0)
rand = random.sample([k for k in keep], min(24, len(keep)))
montage(rand, os.path.join(a.out, "montage_random.jpg"))
print("saved", a.out)
