#!/usr/bin/env python3
"""rosbag2 → 갈림길 표지판 SAM 2.1 추적용 run 폴더 (구간 안 카메라 프레임 전부, 30 fps 원본 JPEG 그대로).

압축 영상(/camera/color/image_raw/compressed)이 없으면 원본(/camera/color/image_raw, rgb8)을 JPEG 95 로 저장.
bag_frames.py 는 0.5 s 마다 + 멈춘 장면을 빼서 pseudo-label 후보를 고르지만, SAM 2.1 영상 추적은 프레임이 촘촘해야
멀리(14 px 까지) 따라가므로 구간 안 프레임을 하나도 빼지 않는다.

구간: /ar/mission_state 의 mission_id 가 --missions(기본 7 = 3차선, 2 = 갈림길) 인 시간 + 앞뒤 --pad 초.
      표지판은 3차선 구간(M7) 끝에서 처음 보이고 갈림길(M2)에서 지나간다.
      mission_state 가 없는 bag(수동 주행 녹화)은 --range t0:t1 (bag 시작 기준 초)로 직접 주거나 --all.
출력: <out>/<bag>[_<k>]/frames/<bag>_<idx:05d>.jpg  (idx = bag 안 카메라 프레임 번호)
      <out>/<bag>[_<k>]/frames.csv (name, idx, t, stamp_ns, mission_id)
      구간이 여러 번이면(재출발 등) _1, _2 ... 로 나눈다.
사용: python bag_fork_frames.py <out> <bag 폴더>... --msgdir <autorace2026>/src/ar_msgs/msg [--missions 7,2] [--pad 3]
      python bag_fork_frames.py <out> <bag> --msgdir ... --range 120:165
"""
import argparse, csv, glob, os
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

CAM = "/camera/color/image_raw/compressed"
RAW = "/camera/color/image_raw"   # 압축 영상이 없으면 원본(rgb8)을 JPEG(품질 95)로 저장


def raw_to_jpeg(m, q=95):
    """sensor_msgs/Image (rgb8 | bgr8) → JPEG 바이트 (압축 영상이 없는 bag, 예: rawcam_1117_193722)."""
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
    if m.encoding.lower() == "rgb8":
        a = a[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(a), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("bags", nargs="+")
ap.add_argument("--msgdir", required=True, help="팀 레포 autorace2026 의 src/ar_msgs/msg")
ap.add_argument("--missions", default="7,2", help="이 미션 번호 구간의 프레임 (쉼표)")
ap.add_argument("--pad", type=float, default=3.0, help="구간 앞뒤로 더 넣을 초")
ap.add_argument("--range", default="", help="t0:t1 (bag 시작 기준 초). 주면 mission_state 대신 이 구간")
ap.add_argument("--all", action="store_true", help="bag 전체")
a = ap.parse_args()
want = {int(x) for x in a.missions.split(",") if x}

ts = get_typestore(Stores.ROS2_HUMBLE); add = {}
for p in glob.glob(os.path.join(a.msgdir, "*.msg")):
    add.update(get_types_from_msg(Path(p).read_text(), f"ar_msgs/msg/{Path(p).stem}"))
ts.register(add)


def windows(bag):
    """bag 시작 기준 [t0, t1] 목록."""
    if a.all:
        return [(-1e9, 1e9)]
    if a.range:
        t0, t1 = (float(v) for v in a.range.split(":"))
        return [(t0, t1)]
    spans, cur = [], None
    with AnyReader([bag], default_typestore=ts) as r:
        ms = [c for c in r.connections if c.topic == "/ar/mission_state"]
        if not ms:
            return None
        start = r.start_time
        for c, t, raw in r.messages(connections=ms):
            tt = (t - start) / 1e9
            mid = int(r.deserialize(raw, c.msgtype).mission_id)
            if mid in want:
                cur = [tt, tt] if cur is None else [cur[0], tt]
            elif cur is not None and tt - cur[1] > 1.0:   # 1 s 넘게 벗어나면 구간 끝
                spans.append(cur); cur = None
        if cur is not None:
            spans.append(cur)
    merged = []
    for s in spans:   # pad 를 넣고 겹치면 합침
        s = [s[0] - a.pad, s[1] + a.pad]
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    return [tuple(s) for s in merged]


for bag in map(Path, a.bags):
    ws = windows(bag)
    if ws is None:
        print(f"{bag.name}: /ar/mission_state 없음 → --range 또는 --all 로 구간을 주세요"); continue
    if not ws:
        print(f"{bag.name}: 미션 {sorted(want)} 구간 없음"); continue
    runs = []
    for k, (t0, t1) in enumerate(ws):
        rn = bag.name if len(ws) == 1 else f"{bag.name}_{k + 1}"
        d = Path(a.out) / rn
        (d / "frames").mkdir(parents=True, exist_ok=True)
        fh = open(d / "frames.csv", "w", newline=""); w = csv.writer(fh)
        w.writerow(["name", "idx", "t", "stamp_ns", "mission_id"])
        runs.append([t0, t1, d, fh, w, 0])
    with AnyReader([bag], default_typestore=ts) as r:
        cam = [c for c in r.connections if c.topic == CAM] or [c for c in r.connections if c.topic == RAW]
        if not cam:
            print(f"{bag.name}: {CAM}·{RAW} 없음, 건너뜀"); continue
        is_raw = cam[0].topic == RAW
        ms = [c for c in r.connections if c.topic == "/ar/mission_state"]
        start = r.start_time; idx = -1; mission = -1
        for c, t, raw in r.messages(connections=cam + ms):
            tt = (t - start) / 1e9
            if c.topic not in (CAM, RAW):
                mission = int(r.deserialize(raw, c.msgtype).mission_id); continue
            idx += 1
            for run in runs:
                if run[0] <= tt <= run[1]:
                    m = r.deserialize(raw, c.msgtype)
                    fn = f"{bag.name}_{idx:05d}.jpg"
                    open(run[2] / "frames" / fn, "wb").write(raw_to_jpeg(m) if is_raw else bytes(m.data))   # 압축 영상은 JPEG 그대로
                    run[4].writerow([fn, idx, round(tt, 3), m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, mission])
                    run[5] += 1
    for t0, t1, d, fh, w, n in runs:
        fh.close()
        print(f"{d.name}: {max(t0, 0):.1f}~{t1:.1f} s, 프레임 {n}장 -> {d}/frames", flush=True)
