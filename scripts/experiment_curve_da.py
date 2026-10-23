#!/usr/bin/env python3
"""실험: 차선을 '벽'으로 삼아 앙상블 da를 자르고, 차 위치(화면 아래 가운데)에서 이어진 da만 ego로 남긴다.

배경: 앙상블 da는 커브를 잘 잡는데, ego 기하(행 단위 좌/우 차선 쌍 추적)는 커브에서 차선이 한 개만 보이거나
화면과 거의 평행(가로)하게 누우면 FAIL. 행 단위 좌/우 채우기는 가로 차선에서 방향이 틀린다.
-> 방향에 무관한 방법: ego = flood fill(앙상블 da AND NOT 차선 벽, 시작점 = 차 바로 앞).
   세로 차선 두 개면 그 사이(기하와 같음), 누운 커브 차선이면 그 아래(차 쪽), 한쪽만 보이면 차선 ~ da 끝.

차선 벽: 앙상블 ll ∪ 노란색 HSV에서 화면 맨 아래 가짜 가로줄/작은 조각만 제거. 기하용 clean_ll과 달리
'납작한 가로선' 필터는 빼야 함(누운 커브 차선까지 지워짐). 벽 틈은 팽창으로 메움.

출력: OK 프레임 검증(벽 flood fill vs 기하 ego IoU) + FAIL 프레임 적용 GIF
  (원본 | 앙상블 da/ll | 벽 flood fill 결과: 노란 점 = 시작점)
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
WALL_DILATE = 7     # 차선 벽 팽창(px) — 점선/검출 틈 메우기
MIN_AREA = 8000


def lane_walls(ll_raw):
    """flood fill용 차선 벽: 맨 아래 가짜 가로줄과 작은 조각만 제거(가로 차선은 유지)."""
    h, w = ll_raw.shape
    ll = ll_raw.copy()
    for y in range(h - 15, h):
        xs = np.flatnonzero(ll[y])
        if xs.size:
            for seg in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
                if seg.size > 40:
                    ll[y, seg] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ll.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(ll)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            keep |= lab == i
    return cv2.dilate(keep.astype(np.uint8), np.ones((WALL_DILATE, WALL_DILATE), np.uint8)).astype(bool)


def seed_point(free):
    """화면 아래 가운데에서 가장 가까운 free 픽셀(아래 20행 안에서)."""
    h, w = free.shape
    for y in range(h - 5, h - 25, -1):
        xs = np.flatnonzero(free[y])
        if xs.size:
            return int(xs[np.argmin(np.abs(xs - w / 2))]), y
    return None


def wall_flood(road, walls):
    free = road & ~walls
    seed = seed_point(free)
    if seed is None:
        return np.zeros_like(road), None
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    region = lab == lab[seed[1], seed[0]]
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool) & road
    return region, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(m.ROOT, "outputs", "montages", "curve_da_experiment.gif"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))[args.start:args.end:args.step]
    members = [m.MODELS[k][1]() for k in m.MODELS]
    vote_min = len(members) // 2 + 1
    limiter = m.GpuDutyLimiter()
    t0 = time.time()
    ok_ious, n_fail, n_saved, frames = [], 0, 0, []
    for i, path in enumerate(paths):
        img0 = cv2.imread(path)
        with limiter:
            probs = [f.probs(img0) for f in members]
        road = np.mean([p[0] for p in probs], 0) >= 0.5
        ll_vote = (np.stack([p[1] for p in probs]) >= 0.5).sum(0) >= vote_min
        ll_raw = ll_vote | E.hsv_lanes(img0, road)
        ll = E.clean_ll(ll_raw)
        left, right, _, reason = E.track_ego(ll, road)
        walls = lane_walls(ll_raw)
        region, seed = wall_flood(road, walls)

        if not reason:  # OK: 검증만
            geo = E.ego_mask(left, right, road, ll.shape)
            ok_ious.append((region & geo).sum() / max((region | geo).sum(), 1))
            continue
        n_fail += 1
        area = int(region.sum())
        frac = area / road.sum() if road.any() else 0
        if area < MIN_AREA:
            tag, color = f"too small ({area}px)", (80, 80, 255)
        else:
            n_saved += 1
            tag, color = f"da kept ({frac * 100:.0f}% of ensemble da)", (120, 255, 120)
        vis = m.overlay_masks(img0, region, walls)
        if seed:
            cv2.circle(vis, seed, 7, (0, 255, 255), -1)
        panels = [
            E.label_panel(img0, f"Raw  {os.path.basename(path)}"),
            E.label_panel(m.overlay_masks(img0, road, ll_raw), f"Ensemble (ego was {reason})", (80, 80, 255)),
            E.label_panel(vis, f"Wall flood: {tag}", color),
        ]
        frames.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))
        if i % 25 == 0:
            print(f"{i}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)

    v = np.array(ok_ious)
    print(f"검증(OK {len(v)}장): 벽 flood fill vs 기하 ego IoU 평균 {v.mean():.3f}, 중앙값 {np.median(v):.3f}, "
          f"0.8 이상 {np.mean(v >= 0.8) * 100:.0f}%, 0.5 미만 {np.mean(v < 0.5) * 100:.0f}%")
    print(f"FAIL {n_fail}장 중 da 영역이 나온 프레임 {n_saved}장 (면적 >= {MIN_AREA}px)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"저장: {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MB, {len(frames)}프레임)")


if __name__ == "__main__":
    main()
