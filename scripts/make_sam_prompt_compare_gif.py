#!/usr/bin/env python3
"""SAM2 프롬프트 방식 4가지를 2x2 GIF로 비교한다 (make_ego_lane_labels.py와 같은 기하 ego 기준).

  | 점만 (sparse)   | 마스크만        |
  | 마스크 + 점     | 마스크 + 박스   |

각 칸은 SAM2 **원본** 결과(기하 범위로 자르기 전). 칸 제목에 기하 ego 대비 IoU / ego 밖 번짐(%)을 표시.
점 = 포함 3(ego 중심선) + 제외 최대 12(좌/우 경계 바깥), 마스크 = 기하 ego를 256x256 logit으로,
박스 = 기하 ego의 외곽 사각형. ego 추적이 FAIL인 프레임은 프롬프트가 없어 원본만 표시.

사용 예:
  .venv/Scripts/python scripts/make_sam_prompt_compare_gif.py --frames D:/fine-tune/LimoPro_frames_20261023 \
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
import make_ego_lane_labels as E  # noqa: E402

m = E.m
MODES = ["points", "mask", "mask+points", "mask+box"]
TITLES = {"points": "Points only", "mask": "Mask only", "mask+points": "Mask + points", "mask+box": "Mask + box"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=os.path.join(m.ROOT, "outputs", "montages", "sam_prompt_compare.gif"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    from ultralytics import SAM
    from ultralytics.engine.model import Model

    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))[args.start:args.end:args.step]
    if not paths:
        sys.exit(f"프레임 없음: {args.frames}")
    members = [m.MODELS[k][1]() for k in m.MODELS]
    vote_min = len(members) // 2 + 1
    sam = SAM(E.SAM_WEIGHT)
    limiter = m.GpuDutyLimiter()

    def run_sam(img0, prompts):
        with limiter:
            r = Model.predict(sam, img0, False, prompts=prompts, conf=0.25, task="segment",
                              mode="predict", imgsz=1024, verbose=False)[0]
        if r.masks is None:
            return np.zeros(img0.shape[:2], bool)
        sm = r.masks.data[0].cpu().numpy().astype(bool)
        if sm.shape != img0.shape[:2]:
            sm = cv2.resize(sm.astype(np.uint8), (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        return sm

    stats = {k: [] for k in MODES}
    frames = []
    t0 = time.time()
    for i, path in enumerate(paths):
        img0 = cv2.imread(path)
        fname = os.path.basename(path)
        with limiter:
            probs = [f.probs(img0) for f in members]
        road = np.mean([p[0] for p in probs], 0) >= 0.5
        ll_vote = (np.stack([p[1] for p in probs]) >= 0.5).sum(0) >= vote_min
        ll = E.clean_ll(ll_vote | E.hsv_lanes(img0, road))
        left, right, _, reason = E.track_ego(ll, road)
        ego = E.ego_mask(left, right, road, ll.shape)
        pts, lbl = E.sam_prompts(left, right, ll.shape) if not reason else (None, None)

        panels = []
        if pts is None or not ego.any():
            for k in MODES:
                panels.append(E.label_panel(img0, f"{TITLES[k]}  (FAIL: {reason or 'no prompt'})", (80, 80, 255)))
        else:
            ys, xs = np.nonzero(ego)
            mask = E.sam_mask_logits(ego)
            box = [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]
            prompts = {
                "points": dict(points=pts, labels=lbl),
                "mask": dict(masks=mask),
                "mask+points": dict(points=pts, labels=lbl, masks=mask),
                "mask+box": dict(bboxes=box, masks=mask),
            }
            for k in MODES:
                sm = run_sam(img0, prompts[k])
                iou = (sm & ego).sum() / max((sm | ego).sum(), 1)
                leak = (sm & ~ego).sum() / max(ego.sum(), 1)
                stats[k].append((iou, leak))
                vis = m.overlay_masks(img0, sm, ll)
                if k in ("points", "mask+points"):
                    vis = E.draw(vis, np.zeros_like(sm), np.zeros_like(sm), pts=pts, lbl=lbl)
                if k == "mask+box":
                    cv2.rectangle(vis, tuple(box[0][:2]), tuple(box[0][2:]), (0, 255, 255), 2)
                panels.append(E.label_panel(vis, f"{TITLES[k]}  IoU {iou:.2f}  leak {leak * 100:.0f}%"))
        # 2x2 + 원본 파일명은 왼쪽 위 칸 아래쪽에 작게
        cv2.putText(panels[0], fname, (4, E.PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        frames.append(Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)))
        if i % 25 == 0:
            print(f"{i}/{len(paths)} ({time.time() - t0:.1f}s)", flush=True)

    for k in MODES:
        a = np.array(stats[k]) if stats[k] else np.zeros((0, 2))
        if len(a):
            print(f"{TITLES[k]:<14} 기하 ego 대비 평균 IoU {a[:, 0].mean():.3f}, 번짐 {a[:, 1].mean() * 100:.0f}% (n={len(a)})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"저장: {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MB, {len(frames)}프레임)")


if __name__ == "__main__":
    main()
