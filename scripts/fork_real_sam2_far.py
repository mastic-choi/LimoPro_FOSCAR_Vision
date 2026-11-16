#!/usr/bin/env python3
"""fork_real_sam2_track.py 다음: 먼 구간(장애물 사이로 표지판이 멀리 한 번 보이는 곳)을 SAM 2.1 로 따로 추적해 sign_track.json 에 합친다.

가까운 구간 seed 에서 거꾸로 따라가면, 그 사이 약 100 프레임 동안 차가 돌며 표지판이 화면 밖으로 나가서 SAM2 가 놓친다 (11-16 실물 주행 5개 모두).
먼 구간에서는 분류기가 흐린 작은 크롭을 확인 못 하지만 HybridNets 는 sign_A/sign_B 를 0.8~0.9 로 잡고 글자도 맞았다(사람 확인).
  seed: 가까운 구간 첫 프레임 - 200 ~ -60 안의 sign_A/sign_B 검출 중 점수 >= 0.75, 점수 높은 순으로 시간 간격 15 프레임 이상 최대 3개
  작업 구간: seed 앞뒤 60 프레임. 이미 박스가 있는 프레임은 덮어쓰지 않는다. 합친 프레임에는 "pass": "far".
아주 먼 구간(3차선 장애물 구간에서 차로 끝에 표지판이 아주 작게 보이는 곳, 약 75~200 프레임)은 --name vfar 로 한 번 더:
  seed: 이미 박스가 있는 가장 이른 프레임 - 60 이전의 sign_A/sign_B 검출 중 점수 >= 0.7, 폭 < 60 px (큰 오탐 제외)
사용 (fork venv): python scripts/fork_real_sam2_far.py <ckpt> <run 폴더>... [--name far|vfar]
"""
import json, os, shutil, sys

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

args = [a for a in sys.argv[1:] if not a.startswith("--name")]
NAME = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--name=")), "far")
ckpt = os.path.abspath(args[0]); runs = [os.path.abspath(r) for r in args[1:]]
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt, device=torch.device("cuda"))


def mask_box(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return None, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, h, a = st[k]
    return [int(x), int(y), int(x + w), int(y + h)], int(a)


for run in runs:
    D = json.load(open(os.path.join(run, "det.json"))); T = json.load(open(os.path.join(run, "sign_track.json"))); F = T["frames"]
    if NAME == "far":
        first = next(r["i"] for r in F if r["sam_box"] and r.get("pass") not in ("far", "vfar"))
        cand = [(i, c) for i in range(max(0, first - 200), first - 60) for c in D[i]["cand"]
                if c["cls"] in ("sign_A", "sign_B") and c["score"] >= 0.75]
    else:
        first = next(r["i"] for r in F if r["sam_box"])
        cand = [(i, c) for i in range(0, first - 60) for c in D[i]["cand"]
                if c["cls"] in ("sign_A", "sign_B") and c["score"] >= 0.7 and c["box"][2] - c["box"][0] < 60]
    cand.sort(key=lambda t: -t[1]["score"])
    seeds = []
    for i, c in cand:
        if all(abs(i - s) >= 15 for s, _ in seeds): seeds.append((i, c))
        if len(seeds) == 3: break
    if not seeds:
        print(os.path.basename(run), ": 먼 구간 seed 없음"); continue
    seeds.sort()
    lo, hi = max(0, seeds[0][0] - 60), min(first - 1, seeds[-1][0] + 60)
    jd = os.path.join(run, f"_sam2_{NAME}"); shutil.rmtree(jd, ignore_errors=True); os.makedirs(jd)
    for k, i in enumerate(range(lo, hi + 1)):
        cv2.imwrite(os.path.join(jd, f"{k:05d}.jpg"), cv2.imread(os.path.join(run, "frames", F[i]["frame"])), [cv2.IMWRITE_JPEG_QUALITY, 95])
    masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=jd, offload_video_to_cpu=True, offload_state_to_cpu=True)
        for i, c in seeds:
            pred.add_new_points_or_box(st, frame_idx=i - lo, obj_id=1, box=np.array(c["box"], np.float32))
        for rev in (False, True):
            for fi, _, lg in pred.propagate_in_video(st, start_frame_idx=(seeds[0][0] if not rev else seeds[-1][0]) - lo, reverse=rev):
                masks[fi + lo] = (lg[0, 0] > 0).cpu().numpy()
        pred.reset_state(st)
    shutil.rmtree(jd, ignore_errors=True)
    added = 0
    for i, m in masks.items():
        if F[i]["sam_box"]: continue
        b, a = mask_box(m)
        if b:
            F[i].update({"sam_box": b, "sam_area": a, "pass": NAME, "seed": i in [s for s, _ in seeds]}); added += 1
    T[f"{NAME}_seeds"] = [[i, c["cls"], c["score"], c["box"]] for i, c in seeds]; T[f"{NAME}_window"] = [lo, hi]
    json.dump(T, open(os.path.join(run, "sign_track.json"), "w"))
    far = [r["i"] for r in F if r.get("pass") == NAME]
    print(f"{os.path.basename(run)}: {NAME} seed {[(i, c['cls'], c['score']) for i, c in seeds]}, 박스 +{added}장 ({far[0] if far else '-'}~{far[-1] if far else '-'})", flush=True)
