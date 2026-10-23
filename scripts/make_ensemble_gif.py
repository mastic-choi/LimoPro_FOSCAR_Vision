#!/usr/bin/env python3
"""models/의 공개 da/ll 모델 7개를 앙상블해서 Voting vs Averaging 을 나란히 보여주는 GIF.

나중에 pseudo-label 생성용 앙상블 방식을 고르기 위한 비교. 배치: 원본 | Voting | Averaging.
  - Voting   : 모델별로 확률 >= 0.5 를 1표로 보고, 7표 중 VOTE_MIN(기본 4, 과반) 이상이면 채택
  - Averaging: 7개 모델 확률을 픽셀별 평균(soft-vote) 후 >= 0.5 채택
               (TwinLiteNet-KMU 딥앙상블과 같은 방식)
da/ll 각각 따로 적용. 오버레이 색은 make_model_compare_gif.py와 동일(파랑=da, 빨강=ll).
모델 로더/전처리도 그 스크립트의 것(infer.probs)을 그대로 쓴다.

사용 예:
  .venv/Scripts/python scripts/make_ensemble_gif.py --frames D:/fine-tune/LimoPro_frames_20261023 \
      --start 85 --step 2 --fps 5
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_model_compare_gif as m  # noqa: E402

PANEL_W, PANEL_H = 400, 300


def label_panel(img, text):
    panel = cv2.resize(img, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frame_NNNNNN.png 가 들어있는 폴더")
    ap.add_argument("--out", default=os.path.join(m.ROOT, "outputs", "montages", "ensemble_vote_vs_avg.gif"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--vote-min", type=int, default=None, help="Voting 채택 최소 표 수 (기본: 과반)")
    ap.add_argument("--models", nargs="*", default=list(m.MODELS), help="앙상블 멤버 (기본: 7개 전부)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))[args.start:args.end:args.step]
    if not paths:
        sys.exit(f"프레임 없음: {args.frames}")
    n = len(args.models)
    vote_min = args.vote_min or n // 2 + 1
    print(f"device={m.torch.cuda.get_device_name(0) if m.device.type == 'cuda' else 'cpu'}, "
          f"{len(paths)}프레임, 멤버 {n}개, voting >= {vote_min}/{n}", flush=True)

    members = []
    for key in args.models:
        t = time.time()
        members.append(m.MODELS[key][1]())
        print(f"로드: {key} ({time.time() - t:.1f}s)", flush=True)

    frames = []
    limiter = m.GpuDutyLimiter()
    t0 = time.time()
    for i, path in enumerate(paths):
        img0 = cv2.imread(path)
        with limiter:
            probs = [f.probs(img0) for f in members]
        da = np.stack([p[0] for p in probs])
        ll = np.stack([p[1] for p in probs])
        vote_da = (da >= 0.5).sum(0) >= vote_min
        vote_ll = (ll >= 0.5).sum(0) >= vote_min
        avg_da = da.mean(0) >= 0.5
        avg_ll = ll.mean(0) >= 0.5

        panels = [
            label_panel(img0, f"Raw  {os.path.basename(path)}"),
            label_panel(m.overlay_masks(img0, vote_da, vote_ll), f"Voting (>= {vote_min}/{n} models)"),
            label_panel(m.overlay_masks(img0, avg_da, avg_ll), f"Averaging (mean prob >= 0.5, {n} models)"),
        ]
        frames.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))
        if i % 25 == 0:
            print(f"{i}/{len(paths)} ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"총 {len(frames)}프레임, GIF 저장 중...", flush=True)
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True
    )
    print(f"저장: {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MB, "
          f"{len(frames)}프레임 @ {args.fps}fps)")


if __name__ == "__main__":
    main()
