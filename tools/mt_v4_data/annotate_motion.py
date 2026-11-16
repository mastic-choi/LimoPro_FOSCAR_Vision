#!/usr/bin/env python3
"""frames.csv 에 프레임 시각의 주행 속도(/odom)와 차체 기울기(/imu 가속도 방향)를 붙인다 → frames_motion.csv

  speed : 프레임 시각 ±0.5 s 안 /odom twist.linear.x 절댓값의 최댓값 [m/s]  (0 이면 서 있음 또는 손에 들림)
  tilt  : 프레임 시각 ±0.25 s 안 /imu 선가속도 평균이 중력(z)과 이루는 각 [deg]  (손으로 들어 기울이면 커짐)
사용: python annotate_motion.py <frames 폴더> <rosbags 루트>
"""
import csv, glob, os, sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

fr, root = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(fr, "frames.csv"))))
ts = get_typestore(Stores.ROS2_HUMBLE)
bags = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "*", "*")) if os.path.isdir(p)}
out = []
for bag in sorted({r["bag"] for r in rows}):
    od_t, od_v, im_t, im_a = [], [], [], []
    with AnyReader([Path(bags[bag])], default_typestore=ts) as r:
        t0 = r.start_time
        con = [c for c in r.connections if c.topic in ("/odom", "/imu")]
        for c, t, raw in r.messages(connections=con):
            m = r.deserialize(raw, c.msgtype); tt = (t - t0) / 1e9
            if c.topic == "/odom":
                od_t.append(tt); od_v.append(abs(m.twist.twist.linear.x))
            else:
                a = m.linear_acceleration; im_t.append(tt); im_a.append((a.x, a.y, a.z))
    im_a = np.array(im_a, np.float64).reshape(-1, 3)
    for rr in [x for x in rows if x["bag"] == bag]:
        t = float(rr["t"])
        i, j = bisect_left(od_t, t - 0.5), bisect_right(od_t, t + 0.5)
        sp = max(od_v[i:j]) if j > i else -1.0
        i, j = bisect_left(im_t, t - 0.25), bisect_right(im_t, t + 0.25)
        if j > i:
            g = im_a[i:j].mean(0); tilt = float(np.degrees(np.arccos(np.clip(abs(g[2]) / (np.linalg.norm(g) + 1e-9), -1, 1))))
        else:
            tilt = -1.0
        out.append({**rr, "speed": round(sp, 3), "tilt": round(tilt, 1)})
    v = [o for o in out if o["bag"] == bag]
    print(f"{bag}: odom {len(od_t)} imu {len(im_t)}  서 있음(<0.03) {sum(o['speed'] < 0.03 for o in v)}  기울어짐(>10°) {sum(o['tilt'] > 10 for o in v)} / {len(v)}", flush=True)
with open(os.path.join(fr, "frames_motion.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
