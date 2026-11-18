#!/usr/bin/env python3
"""M7 원본 프레임에서 갈림길 표지판을 SAM 2.1 영상 추적으로 따라가 모든 프레임의 박스를 만든다 (멀어서 검출이 못 잡는 프레임 포함).

1) 시작 박스(seed): fork_real_detect.py 의 det.json 에서 분류기가 표지판(max(pA,pB) >= 0.9)이라 한 후보 중, 화면 안에 온전히 있는 것.
   가장 큰 박스 + 시간을 나눠 최대 3개 (같은 표지판 하나라 obj_id 하나).
2) SAM 2.1 (hiera-base+) 로 seed 들에서 앞·뒤로 전파 → 프레임마다 마스크 → 가장 큰 덩어리의 박스.
3) 검출(분류기 확인) 박스가 있는 프레임은 IoU 를 적어 둔다 (서로 맞는지 확인용).
작업 구간 = 첫 확인 프레임 - BACK(기본 240) ~ 마지막 확인 프레임 + 30 (그 밖은 표지판이 안 보이는 3차선 앞부분).
  환경변수 BACK 으로 더 거슬러 올라갈 수 있다 (30 fps 에서 240 = 8 s).
  환경변수 LETTER=A|B 를 주면 분류기가 그 글자라고 한 박스만 확인 박스로 쓴다 (먼 구간 분류기 오답, 옆 표지판 조각 제외).
끝에 SAM 박스 가장 작은 크기(√(w·h), 640x360 기준)를 출력한다. 14 px 까지 못 따라갔으면 fork_real_sam2_far.py 로 먼 구간을 더 붙인다 (사용자 11-17: 14 px 까지 추적).

사용 (fork venv, sam2 설치): python scripts/fork_real_sam2_track.py <ckpt> <run 폴더>...   → <run>/sign_track.json, <run>/track_check.jpg
"""
import json, os, shutil, sys

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ckpt = os.path.abspath(sys.argv[1]); runs = [os.path.abspath(r) for r in sys.argv[2:]]
dev = torch.device("cuda")
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt, device=dev)
LETTER = os.environ.get("LETTER", "")   # A | B: 주행 글자를 알면 그 글자로 확인된 박스만 seed·확인으로 씀 (사람 확인, 11-17)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def mask_box(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return None, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, h, a = st[k]
    return [int(x), int(y), int(x + w), int(y + h)], int(a)


for run in runs:
    det = json.load(open(os.path.join(run, "det.json")))
    fs = [r["frame"] for r in det]
    conf = {}   # 프레임 → 분류기 확인 박스 (가장 확신 높은 것)
    for i, r in enumerate(det):
        best = None
        for c in r["cand"]:
            p = c["cnn"]
            if LETTER and ("A" if p[0] > p[1] else "B") != LETTER:
                continue   # 글자를 아는 주행: 그 글자로 확인된 박스만 (다른 표지판·뒷면 조각을 seed 로 잡지 않게)
            if max(p[0], p[1]) >= 0.6 and abs(p[0] - p[1]) >= 0.2 and (best is None or max(p[:2]) > max(best["cnn"][:2])):
                best = c
        if best: conf[i] = best
    ci = sorted(conf)
    lo, hi = max(0, ci[0] - int(os.environ.get("BACK", 240))), min(len(fs) - 1, ci[-1] + 30)
    ok = [i for i in ci if max(conf[i]["cnn"][:2]) >= 0.9 and conf[i]["box"][0] > 5 and conf[i]["box"][2] < 1275 and conf[i]["box"][3] < 715]
    area = lambda i: (conf[i]["box"][2] - conf[i]["box"][0]) * (conf[i]["box"][3] - conf[i]["box"][1])
    seeds = [max(ok, key=area)]
    for part in np.array_split(np.array(ok), 3):
        if len(part):
            s = int(max(part, key=area))
            if all(abs(s - x) > 15 for x in seeds): seeds.append(s)
    seeds = sorted(seeds)[:3]
    # SAM2 는 정수 이름 JPEG 폴더를 읽는다
    jd = os.path.join(run, "_sam2_jpg"); shutil.rmtree(jd, ignore_errors=True); os.makedirs(jd)
    for k, i in enumerate(range(lo, hi + 1)):
        cv2.imwrite(os.path.join(jd, f"{k:05d}.jpg"), cv2.imread(os.path.join(run, "frames", fs[i])), [cv2.IMWRITE_JPEG_QUALITY, 95])
    masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=jd, offload_video_to_cpu=True, offload_state_to_cpu=True)
        for s in seeds:
            pred.add_new_points_or_box(st, frame_idx=s - lo, obj_id=1, box=np.array(conf[s]["box"], np.float32))
        for rev in (False, True):
            for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=(seeds[0] if not rev else seeds[-1]) - lo, reverse=rev):
                masks[fi + lo] = (logits[0, 0] > 0).cpu().numpy()
        pred.reset_state(st)
    shutil.rmtree(jd, ignore_errors=True)
    out, ious = [], []
    for i in range(len(fs)):
        b, a = mask_box(masks[i]) if i in masks else (None, 0)
        d = conf.get(i)
        rec = {"i": i, "frame": fs[i], "sam_box": b, "sam_area": a, "seed": i in seeds}
        if d:
            rec["det_box"] = d["box"]; rec["det_cls"] = d["cls"]; rec["det_src"] = d["src"]; rec["cnn"] = d["cnn"]
            if b: rec["iou"] = round(iou(b, d["box"]), 3); ious.append(rec["iou"])
        out.append(rec)
    json.dump({"run": os.path.basename(run), "window": [lo, hi], "seeds": seeds, "frames": out}, open(os.path.join(run, "sign_track.json"), "w"))
    n_sam = sum(1 for r in out if r["sam_box"])
    ious = np.array(ious)
    sz = [((r["sam_box"][2] - r["sam_box"][0]) * (r["sam_box"][3] - r["sam_box"][1])) ** 0.5 / 2 for r in out if r["sam_box"]]
    first = next((r["i"] for r in out if r["sam_box"]), None)
    print(f"{os.path.basename(run)}: 구간 {lo}-{hi}, seed {seeds}, SAM2 박스 {n_sam}장 (검출 확인 {len(ci)}장) | "
          f"검출과 IoU 중앙값 {np.median(ious):.2f}, <0.5 인 프레임 {int((ious < 0.5).sum())}/{len(ious)} | "
          f"가장 작은 박스 {min(sz):.1f}px (첫 박스 프레임 {first})"
          + ("" if min(sz) <= 14.5 else "  ← 14px 까지 못 감: fork_real_sam2_far.py 로 먼 구간 보충"), flush=True)
