#!/usr/bin/env python3
"""fork_real_sam2_track.py 다음: 옆·뒤 표지판과 한 덩어리로 잡힌 구간을 고쳐서 다시 추적한다 (사용자 11-18).

대회장 갈림길은 표지판 기둥이 둘이라(앞 = 이 주행 글자, 바로 뒤 = 반대쪽을 향한 표지판 뒷면, 흰 원판),
흔들린 프레임에서 SAM 2.1 이 둘을 한 마스크로 묶어 박스가 가로로 넓어진다 (가로/세로 > 1.6).
  1) 기준 프레임: 기존 sign_track.json 에서 가로/세로가 넓은 프레임을 시간에 고르게 고르고 + 원래 seed.
     표지판 글자 쪽(--side left|right, 기본 left)의 정사각형(높이 x 높이)을 정답 박스로, 정사각형 밖으로 남는 부분 가운데에 '아님' 점(label 0, 남는 폭이 높이의 0.25 미만이면 점 없음).
  2) 같은 구간을 다시 앞뒤로 전파 → 프레임마다 가장 큰 덩어리의 박스.
  3) 파란 표지판 확인: 박스 안 파란 픽셀(HSV H 95~130, S >= 60, V >= 50) 비율이 --min-blue 미만이면 버림
     (표지판이 화면 밖으로 나갈 때 흰 물체·벽을 따라간 박스).
기존 결과는 sign_track_before_refine.json 으로 남긴다.

사용 (label venv): python scripts/fork_real_sam2_refine.py <sam2 ckpt> <run 폴더> [--side left] [--n-ref 5] [--min-blue 0.06]
"""
import argparse, json, os, shutil

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("run")
ap.add_argument("--side", choices=("left", "right"), default="left", help="이 주행 글자 표지판이 덩어리의 어느 쪽인가")
ap.add_argument("--n-ref", type=int, default=5, help="넓은 박스 중 기준으로 쓸 프레임 수")
ap.add_argument("--wide", type=float, default=1.6, help="가로/세로가 이보다 크면 두 표지판이 묶인 것으로 봄")
ap.add_argument("--min-blue", type=float, default=0.06)
a = ap.parse_args()
run = os.path.abspath(a.run)
src = os.path.join(run, "sign_track_before_refine.json")   # 다시 돌려도 항상 처음 추적 결과에서 시작
if not os.path.exists(src):
    shutil.copy(os.path.join(run, "sign_track.json"), src)
T = json.load(open(src))
F = T["frames"]; lo, hi = T["window"]
ar = lambda b: (b[2] - b[0]) / max(1, b[3] - b[1])


def split(b):
    """넓은 박스 → (글자 쪽 정사각형 박스, 반대쪽 '아님' 점)."""
    x1, y1, x2, y2 = b; h = y2 - y1; cy = (y1 + y2) / 2
    rest = (x2 - x1) - h                     # 정사각형 밖으로 남는 폭 = 뒤 표지판이 보이는 부분
    if a.side == "left":
        sq = [x1, y1, min(x2, x1 + h), y2]; neg = [x1 + h + rest / 2, cy]
    else:
        sq = [max(x1, x2 - h), y1, x2, y2]; neg = [x1 + rest / 2, cy]
    return sq, (neg if rest >= 0.25 * h else None)   # 남는 폭이 작으면 '아님' 점을 안 찍음 (표지판 안에 찍히지 않게)


wide = [r["i"] for r in F if r["sam_box"] and ar(r["sam_box"]) > a.wide]
refs = [wide[int(k)] for k in np.linspace(0, len(wide) - 1, min(a.n_ref, len(wide))).round()] if wide else []
prompts = {}
for i in refs:
    prompts[i] = split(F[i]["sam_box"])
for s in T["seeds"]:   # 원래 seed: 뒤 표지판이 조금 걸쳐 있으면(가로/세로 > 1.1) 같이 고침
    b = F[s]["det_box"] if F[s].get("det_box") else F[s]["sam_box"]
    prompts[s] = split(b) if ar(b) > 1.1 else (list(b), None)
print("기준 프레임:", {k: ([round(v) for v in p[0]], None if p[1] is None else [round(v) for v in p[1]]) for k, p in sorted(prompts.items())}, flush=True)

pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_b+.yaml", os.path.abspath(a.ckpt), device=torch.device("cuda"))
jd = os.path.join(run, "_sam2_jpg"); shutil.rmtree(jd, ignore_errors=True); os.makedirs(jd)
for k, i in enumerate(range(lo, hi + 1)):
    cv2.imwrite(os.path.join(jd, f"{k:05d}.jpg"), cv2.imread(os.path.join(run, "frames", F[i]["frame"])), [cv2.IMWRITE_JPEG_QUALITY, 95])
masks = {}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    st = pred.init_state(video_path=jd, offload_video_to_cpu=True, offload_state_to_cpu=True)
    for i, (box, neg) in sorted(prompts.items()):
        kw = dict(box=np.array(box, np.float32))
        if neg is not None:
            kw.update(points=np.array([neg], np.float32), labels=np.array([0], np.int32))
        pred.add_new_points_or_box(st, frame_idx=i - lo, obj_id=1, **kw)
    first, last = min(prompts) - lo, max(prompts) - lo
    for rev, start in ((False, first), (True, last)):
        for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=start, reverse=rev):
            masks[fi + lo] = (logits[0, 0] > 0).cpu().numpy()
    pred.reset_state(st)
shutil.rmtree(jd, ignore_errors=True)


def mask_box(m):
    n, lab, stt, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return None, 0
    k = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA])); x, y, w, h, ar_ = stt[k]
    return [int(x), int(y), int(x + w), int(y + h)], int(ar_)


def blue_ratio(img, b):
    x1, y1, x2, y2 = b
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    if hsv.size == 0: return 0.0
    m = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 50)
    return float(m.mean())


dropped, n_box, n_wide = [], 0, 0
for r in F:
    i = r["i"]
    b, area = mask_box(masks[i]) if i in masks else (None, 0)
    if b is not None:
        br = blue_ratio(cv2.imread(os.path.join(run, "frames", r["frame"])), b)
        r["blue"] = round(br, 3)
        if br < a.min_blue:
            dropped.append(i); b, area = None, 0
    r["sam_box"], r["sam_area"] = b, area
    r["seed"] = i in prompts
    if b:
        n_box += 1; n_wide += ar(b) > 1.67
T["seeds"] = sorted(prompts); T["refine"] = {"side": a.side, "refs": sorted(prompts), "min_blue": a.min_blue, "dropped": dropped}
json.dump(T, open(os.path.join(run, "sign_track.json"), "w"))
sz = [((r["sam_box"][2] - r["sam_box"][0]) * (r["sam_box"][3] - r["sam_box"][1])) ** 0.5 / 2 for r in F if r["sam_box"]]
print(f"{os.path.basename(run)}: 다시 추적 SAM 박스 {n_box}장, 그중 가로/세로 > 1.67: {n_wide}장, 파란색 부족으로 버림 {len(dropped)}장 {dropped} | "
      f"가장 작은 박스 {min(sz):.1f}px", flush=True)
