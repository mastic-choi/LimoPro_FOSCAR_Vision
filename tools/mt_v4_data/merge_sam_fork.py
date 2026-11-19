#!/usr/bin/env python3
"""pseudo_label_mt.py 결과(bag 프레임 da·ll·박스)의 갈림길 표지판 박스를 SAM 2.1 추적 박스로 바꾼다 (사용자 11-18).

pseudo_label_mt.py 는 표지판을 HybridNets 검출 + 분류기로 정해서, 분류기가 확인 못 하는 먼 표지판은 무시 박스이거나 빠진다.
같은 bag 의 갈림길 구간을 SAM 2.1 로 추적했으면(sign_track.json), 시각이 맞는 프레임은 그 박스를 정답으로 쓴다.
  - 프레임 짝: pseudo 프레임의 stamp_ns(frames.csv) ↔ 추적 run 의 frames.csv stamp_ns (가장 가까운 것, 20 ms 안)
  - 그 프레임의 기존 sign_A/sign_B 박스와, SAM 박스와 IoU >= 0.3 인 무시 박스(-2)를 지우고
  - SAM 박스를 넣는다: 크기 √(w·h) (640x360) >= --min-sign-px(14) 이고 가로/세로 0.6~1.67 이면 sign_<글자>, 아니면 무시 박스(-2)
  - 추적 구간 안인데 SAM 박스가 없는 프레임(표지판이 화면 밖)은 기존 sign_A/B 박스만 지운다
사용: python merge_sam_fork.py <pseudo dataset> <frames 폴더(frames.csv)> <추적 run 폴더> <A|B> [--min-sign-px 14]
"""
import argparse, bisect, csv, json, os

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("frames"); ap.add_argument("run"); ap.add_argument("letter", choices=("A", "B"))
ap.add_argument("--min-sign-px", type=float, default=14.0)
a = ap.parse_args()
SIGN = {"A": 1, "B": 2}[a.letter]   # train_multitask.CLASSES 의 sign_A=1, sign_B=2 (v4 학습 때 --merge-ab 로 1번 칸에 합침)

T = json.load(open(os.path.join(a.run, "sign_track.json")))
lo, hi = T["window"]
run_rows = list(csv.DictReader(open(os.path.join(a.run, "frames.csv"))))
box_by_name = {r["frame"]: r["sam_box"] for r in T["frames"]}
idx_of = {r["frame"]: r["i"] for r in T["frames"]}
st = [int(r["stamp_ns"]) for r in run_rows]
order = sorted(range(len(st)), key=lambda k: st[k]); st_sorted = [st[k] for k in order]


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


stat = {"matched": 0, "in_window": 0, "sign": 0, "sign_ign": 0, "removed": 0}
for r in csv.DictReader(open(os.path.join(a.frames, "frames.csv"))):
    s = int(r["stamp_ns"])
    k = bisect.bisect_left(st_sorted, s)
    cand = [j for j in (k - 1, k) if 0 <= j < len(st_sorted)]
    if not cand: continue
    j = min(cand, key=lambda j: abs(st_sorted[j] - s))
    if abs(st_sorted[j] - s) > 20e6: continue
    fname = run_rows[order[j]]["name"]
    if fname not in idx_of or not lo <= idx_of[fname] <= hi: continue
    lab = next((os.path.join(a.ds, sp, "labels", r["name"] + ".txt") for sp in os.listdir(a.ds)
                if os.path.exists(os.path.join(a.ds, sp, "labels", r["name"] + ".txt"))), None)
    if lab is None: continue
    stat["matched"] += 1; stat["in_window"] += 1
    sb = box_by_name.get(fname)
    sb640 = [v / 2 for v in sb] if sb else None
    keep = []
    for line in open(lab):
        p = line.split()
        if not p: continue
        c = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:5])
        b = [(cx - w / 2) * 640, (cy - h / 2) * 360, (cx + w / 2) * 640, (cy + h / 2) * 360]
        if c in (1, 2) or (c == -2 and sb640 and iou(b, sb640) >= 0.3):
            stat["removed"] += 1; continue
        keep.append(line.strip())
    if sb640:
        w, h = sb640[2] - sb640[0], sb640[3] - sb640[1]
        good = (w * h) ** 0.5 >= a.min_sign_px and 0.6 <= w / max(h, 1e-6) <= 1.67
        cid = SIGN if good else -2; stat["sign" if good else "sign_ign"] += 1
        keep.append(f"{cid} {(sb640[0] + sb640[2]) / 1280:.6f} {(sb640[1] + sb640[3]) / 720:.6f} {w / 640:.6f} {h / 360:.6f}")
    open(lab, "w").write("".join(x + "\n" for x in keep))
print(f"{os.path.basename(a.ds)} ← {os.path.basename(a.run)} ({a.letter}): {stat}")
