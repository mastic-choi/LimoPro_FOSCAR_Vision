#!/usr/bin/env python3
"""공개 모델 7개 앙상블 차선(ll)으로 '지금 달리는 차선(ego lane)' da pseudo-label을 만든다.

흐름 (프레임마다):
  1. 7개 모델 확률 -> ll Voting(과반) / da Averaging
  2. ll 정리: 작은 조각, 화면 맨 아래 가로줄(모델 공통 아티팩트), 납작한 가로선(정지선/횡단보도) 제거
  3. 화면 아래 -> 위로 행 단위 추적: 중심선 기준 좌/우 가장 가까운 차선을 ego lane 경계로 선택
     (이전 행 위치 근처만 허용, 점선 끊김은 선형 외삽으로 메움, 화면 아래로 빠진 구간도 외삽)
  4. ego da = 행마다 좌/우 경계 사이 채우기 (앙상블 da 범위로 클리핑)
  5. (옵션) SAM2: ego lane 중심선에 포함 점, 좌/우 차선 바깥에 제외 점을 찍어 비교용 마스크
  6. 품질 판정(OK / FAIL 사유) -> GIF 라벨에 표시, --save-labels 시 PNG로 저장

GIF 배치: 원본 | 앙상블(도로 전체) | ego lane(기하) | ego lane + SAM2
초록 선 = 선택된 좌/우 경계, 파랑 = da, 빨강 = (정리된) ll.

사용 예:
  .venv/Scripts/python scripts/make_ego_lane_labels.py --frames D:/fine-tune/LimoPro_frames_20261023 \
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
import make_model_compare_gif as m  # noqa: E402  (ZLUDA면 여기서 cuDNN 꺼짐)

PANEL_W, PANEL_H = 320, 240
SAM_WEIGHT = os.path.join(m.MODELS_DIR, "sam2", "sam2.1_b.pt")

# 추적 파라미터 (640x480 기준)
ROW_TOL = 14          # 한 행 사이 경계가 움직일 수 있는 최대 px
MAX_GAP = 40          # 점선 끊김 등으로 경계를 못 찾아도 외삽으로 버티는 최대 행 수
MIN_WIDTH = 8         # 좌/우 경계 간격이 이보다 좁아지면(소실점 근처) 추적 종료
MIN_TRACKED_ROWS = 60  # 실제로 양쪽 다 찾은 행이 이보다 적으면 FAIL


def label_panel(img, text, color=(255, 255, 255)):
    panel = cv2.resize(img, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return panel


def clean_ll(ll):
    """앙상블 ll에서 차선이 아닌 덩어리 제거."""
    h, w = ll.shape
    ll = ll.copy()
    # 화면 맨 아래 가로줄이 바닥까지 내려온 차선과 한 덩어리로 붙으면 컴포넌트 필터로 안 걸러진다
    # -> 아래 15행에서 가로로 긴 구간(> 40px)은 행 단위로 직접 지움
    for y in range(h - 15, h):
        xs = np.flatnonzero(ll[y])
        if xs.size:
            for seg in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
                if seg.size > 40:
                    ll[y, seg] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ll.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(ll, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 150:
            continue
        if y + bh >= h - 3 and bh <= 15:  # 화면 맨 아래 가로줄 (TwinLiteNet+ 등 공통 아티팩트)
            continue
        if bh < 25 and bh < 0.15 * bw:  # 납작한 가로선: 정지선/횡단보도
            continue
        keep |= lab == i
    return keep


def row_runs(row):
    """한 행에서 True 구간들의 중심 x 목록."""
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return np.empty(0)
    splits = np.flatnonzero(np.diff(xs) > 1) + 1
    return np.array([seg.mean() for seg in np.split(xs, splits)])


def predict(hist):
    """최근 추적 점들로 다음 행 위치 선형 외삽. hist = [(y, x), ...] (y 감소 순)."""
    if len(hist) < 2:
        return hist[-1][1]
    pts = np.array(hist[-10:])
    slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
    return hist[-1][1] + slope * -1  # 다음 행 = y - 1


def track_ego(ll, road):
    """행 단위로 좌/우 경계를 추적. 반환: left[y], right[y] (없으면 nan), real_rows, reason."""
    h, w = ll.shape
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    real = np.zeros(h, dtype=bool)
    hist_l, hist_r = [], []
    gap_l = gap_r = 0
    center = w / 2
    started = False
    reason = ""

    for y in range(h - 1, -1, -1):
        runs = row_runs(ll[y])
        found = {}
        for side, hist in (("L", hist_l), ("R", hist_r)):
            if hist:
                pred = predict(hist)
                cand = runs[np.abs(runs - pred) <= ROW_TOL] if runs.size else runs
                found[side] = cand[np.argmin(np.abs(cand - pred))] if cand.size else None
            else:
                cand = runs[runs < center] if side == "L" else runs[runs > center]
                found[side] = (cand.max() if side == "L" else cand.min()) if cand.size else None

        if not started:
            if found["L"] is None or found["R"] is None:
                continue
            started = True

        for side, hist in (("L", hist_l), ("R", hist_r)):
            x = found[side]
            if x is not None:
                hist.append((y, x))
                if side == "L":
                    gap_l = 0
                else:
                    gap_r = 0
            elif hist:
                hist.append((y, predict(hist)))  # 외삽으로 버팀
                if side == "L":
                    gap_l += 1
                else:
                    gap_r += 1

        xl, xr = hist_l[-1][1], hist_r[-1][1]
        if gap_l > MAX_GAP or gap_r > MAX_GAP:
            reason = "lane lost"
            break
        if xr - xl < MIN_WIDTH:
            break  # 소실점 근처 수렴 -> 정상 종료
        if not road[y, int(np.clip((xl + xr) / 2, 0, w - 1))]:
            break  # 앙상블 da 밖(매트 끝/벽) -> 정상 종료
        left[y], right[y] = xl, xr
        real[y] = found["L"] is not None and found["R"] is not None
        center = (xl + xr) / 2

    # 화면 아래로 빠진 구간: 처음 찾은 30행으로 아래쪽 외삽
    ys = np.flatnonzero(real)
    if ys.size >= 10:
        y0 = ys.max()
        fit_rows = ys[ys >= y0 - 60][-30:] if ys.size > 30 else ys
        for arr in (left, right):
            k, b = np.polyfit(fit_rows, arr[fit_rows], 1)
            for y in range(y0 + 1, h):
                arr[y] = np.clip(k * y + b, 0, w - 1)

    n_real = int(real.sum())
    if not reason and n_real < MIN_TRACKED_ROWS:
        reason = f"few rows ({n_real})" if n_real else "no lane pair"
    return left, right, n_real, reason


def ego_mask(left, right, road, shape):
    h, w = shape
    ego = np.zeros(shape, dtype=bool)
    for y in range(h):
        if np.isnan(left[y]) or np.isnan(right[y]):
            continue
        ego[y, int(max(0, left[y])):int(min(w - 1, right[y])) + 1] = True
    road_d = cv2.dilate(road.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return ego & road_d


def sam_prompts(left, right, shape):
    """ego 중심선에 포함 점 3개, 같은 행의 좌/우 차선 바깥에 제외 점 2개씩(가까운/먼 바깥, 최대 12개)."""
    h, w = shape
    ys = np.flatnonzero(~np.isnan(left) & ~np.isnan(right))
    if ys.size < 20:
        return None, None
    pts, lbl = [], []
    for q in (0.25, 0.5, 0.8):
        y = int(ys.min() + q * (ys.max() - ys.min()))
        xl, xr = left[y], right[y]
        off = max(20, 0.25 * (xr - xl))
        pts.append([int((xl + xr) / 2), y]); lbl.append(1)
        for k in (1.0, 2.5):  # 차선 바로 바깥 + 더 먼 바깥
            if xl - k * off > 0:
                pts.append([int(xl - k * off), y]); lbl.append(0)
            if xr + k * off < w - 1:
                pts.append([int(xr + k * off), y]); lbl.append(0)
    return [pts], [lbl]  # (1, N, 2): ultralytics SAM이 점들을 한 오브젝트로 묶도록


def hsv_lanes(img0, road):
    """HSV로 노란 차선 검출(앙상블이 놓친 노란 선 보완용). 도로(앙상블 da) 근처만 남김.
    흰색은 안 씀: 횡단보도/마름모/노면 글자/화살표까지 잡혀서 ego 추적을 망침(흰 점선은 앙상블이 잘 잡음)."""
    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 70, 90), (40, 255, 255))
    mask = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    road_d = cv2.dilate(road.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    return mask.astype(bool) & road_d


def draw(img0, da, ll, left=None, right=None, pts=None, lbl=None):
    out = m.overlay_masks(img0, da, ll)
    if left is not None:
        for arr in (left, right):
            ys = np.flatnonzero(~np.isnan(arr))
            if ys.size:
                poly = np.stack([arr[ys], ys], 1).astype(np.int32)
                cv2.polylines(out, [poly], False, (0, 255, 0), 2)
    if pts is not None:
        for (x, y), l in zip(pts[0], lbl[0]):
            cv2.circle(out, (x, y), 6, (0, 255, 0) if l else (0, 0, 255), -1)
            cv2.circle(out, (x, y), 6, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=os.path.join(m.ROOT, "outputs", "montages", "ego_lane_final.gif"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-sam", action="store_true", help="SAM2 생략(최종 = 기하 결과)")
    ap.add_argument("--no-hsv", action="store_true", help="HSV 차선 보완 끄기")
    ap.add_argument("--save-labels", help="이 폴더에 images/ da_masks/ ll_masks/ (+status.csv) 저장")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))[args.start:args.end:args.step]
    if not paths:
        sys.exit(f"프레임 없음: {args.frames}")
    members = [m.MODELS[k][1]() for k in m.MODELS]
    vote_min = len(members) // 2 + 1
    sam = None
    if not args.no_sam:
        from ultralytics import SAM
        sam = SAM(SAM_WEIGHT)
    if args.save_labels:
        for sub in ("images", "da_masks", "ll_masks"):
            os.makedirs(os.path.join(args.save_labels, sub), exist_ok=True)
        status_f = open(os.path.join(args.save_labels, "status.csv"), "w", encoding="utf-8")
        status_f.write("frame,status,tracked_rows,final_source\n")

    frames, n_ok = [], 0
    limiter = m.GpuDutyLimiter()
    t0 = time.time()
    for i, path in enumerate(paths):
        img0 = cv2.imread(path)
        fname = os.path.basename(path)
        with limiter:
            probs = [f.probs(img0) for f in members]
        road = np.mean([p[0] for p in probs], 0) >= 0.5
        ll_vote = (np.stack([p[1] for p in probs]) >= 0.5).sum(0) >= vote_min
        hsv = np.zeros_like(ll_vote) if args.no_hsv else hsv_lanes(img0, road)
        ll = clean_ll(ll_vote | hsv)  # 앙상블 차선 + HSV 보완

        left, right, n_real, reason = track_ego(ll, road)
        ego = ego_mask(left, right, road, ll.shape)
        ok = not reason
        n_ok += ok
        tag = "OK" if ok else f"FAIL: {reason}"
        tag_color = (120, 255, 120) if ok else (80, 80, 255)

        # 최종: SAM2 결과를 기하 ego 범위(9px 여유) 안으로 클리핑. SAM은 매트 전체로 번지므로 단독 사용 X,
        # 차선 안의 사람/장애물 등을 잘라내는 역할만. SAM이 ego의 절반도 못 덮으면 기하 결과로 폴백.
        final, how = ego, "geometric"
        pts = lbl = None
        if sam is not None and ego.any():
            pts, lbl = sam_prompts(left, right, ll.shape)
            if pts is not None:
                with limiter:
                    r = sam(img0, points=pts, labels=lbl, verbose=False)[0]
                sm = r.masks.data[0].cpu().numpy().astype(bool) if r.masks is not None else np.zeros_like(ll)
                if sm.shape != ll.shape:
                    sm = cv2.resize(sm.astype(np.uint8), (ll.shape[1], ll.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                sm = cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(bool)  # 구멍 메우기
                clipped = sm & cv2.dilate(ego.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                if clipped.sum() >= 0.5 * ego.sum():
                    final, how = clipped, "SAM2 in ego"
                else:
                    how = "SAM2 weak -> geometric"

        panels = [
            label_panel(img0, f"Raw  {fname}"),
            label_panel(m.overlay_masks(img0, road, ll_vote), "1. Ensemble 7 (vote ll / avg da)"),
            label_panel(draw(img0, ego, ll, left, right), f"2. +HSV, ego lane  {tag}", tag_color),
            label_panel(draw(img0, final, ll, left, right, pts, lbl), f"3. Final ({how})", tag_color),
        ]
        frames.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))

        if args.save_labels:
            stem = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(args.save_labels, "images", f"{stem}.png"), img0)
            cv2.imwrite(os.path.join(args.save_labels, "da_masks", f"{stem}.png"), (final | ll).astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(args.save_labels, "ll_masks", f"{stem}.png"), ll.astype(np.uint8) * 255)
            status_f.write(f"{fname},{tag},{n_real},{how}\n")
        if i % 25 == 0:
            print(f"{i}/{len(paths)} ({time.time() - t0:.1f}s)  OK {n_ok}/{i + 1}", flush=True)

    if args.save_labels:
        status_f.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"OK {n_ok}/{len(paths)}. GIF 저장 중...", flush=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"저장: {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MB, {len(frames)}프레임)")


if __name__ == "__main__":
    main()
