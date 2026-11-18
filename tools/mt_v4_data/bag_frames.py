#!/usr/bin/env python3
"""rosbag2 → 학습 후보 프레임 (원본 1280x720 JPEG 그대로) + 메타.

압축 영상(/camera/color/image_raw/compressed, 없으면 원본 /camera/color/image_raw 를 JPEG 95 로)을 시간순으로 읽어
  - 직전에 남긴 프레임에서 --every 초 이상 지났고
  - 직전에 남긴 프레임과 화면 전체(160x90 흑백) 평균 절대차가 --still 이상인 것만 남긴다 (멈춘 동안 같은 장면 제거,
    dataset/README.md 의 LimoPro_frames_20261027 정지 중복 제거와 같은 값 2.5).
/ar/mission_state 가 있으면 그 시각의 미션 번호를 같이 적는다 (M7 = 갈림길).

사용: python bag_frames.py <출력 폴더> <bag 폴더>... [--every 0.5] [--still 2.5]
출력: <출력>/orig/<bag>/<bag>_<t초>.jpg, <출력>/frames.csv (name,bag,t,stamp_ns,mission_id,diff)
"""
import argparse, csv, glob, os, sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

def raw_to_jpeg(m, q=95):
    """sensor_msgs/Image (rgb8 | bgr8) → JPEG 바이트 (압축 영상이 없는 bag, 예: rawcam_1117_193722)."""
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
    if m.encoding.lower() == "rgb8":
        a = a[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(a), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("bags", nargs="+")
ap.add_argument("--every", type=float, default=0.5); ap.add_argument("--still", type=float, default=2.5)
ap.add_argument("--msgdir", required=True, help="팀 레포 autorace2026 의 src/ar_msgs/msg")
a = ap.parse_args()

ts = get_typestore(Stores.ROS2_HUMBLE); add = {}
for p in glob.glob(os.path.join(a.msgdir, "*.msg")):
    add.update(get_types_from_msg(Path(p).read_text(), f"ar_msgs/msg/{Path(p).stem}"))
ts.register(add)

out = Path(a.out); (out / "orig").mkdir(parents=True, exist_ok=True)
csv_path = out / "frames.csv"
done = set()
if csv_path.exists():
    done = {r["bag"] for r in csv.DictReader(open(csv_path))}
fh = open(csv_path, "a", newline=""); w = csv.writer(fh)
if not done:
    w.writerow(["name", "bag", "t", "stamp_ns", "mission_id", "diff"])

for bag in a.bags:
    bag = Path(bag); name = bag.name
    if name in done:
        print("이미 함:", name); continue
    (out / "orig" / name).mkdir(exist_ok=True)
    kept = seen = 0; last_t, last_g = -1e9, None; mission = -1
    with AnyReader([bag], default_typestore=ts) as r:
        cam = [c for c in r.connections if c.topic == "/camera/color/image_raw/compressed"] or \
              [c for c in r.connections if c.topic == "/camera/color/image_raw"]   # 압축 영상이 없으면 원본
        if not cam:
            print("영상 없음, 건너뜀:", name); continue
        is_raw = cam[0].topic == "/camera/color/image_raw"
        ms = [c for c in r.connections if c.topic == "/ar/mission_state"]
        t0 = r.start_time
        for c, t, raw in r.messages(connections=cam + ms):
            tt = (t - t0) / 1e9
            if c.topic == "/ar/mission_state":
                mission = int(r.deserialize(raw, c.msgtype).mission_id); continue
            seen += 1
            if tt - last_t < a.every:
                continue
            m = r.deserialize(raw, c.msgtype)
            buf = raw_to_jpeg(m) if is_raw else bytes(m.data)
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (720, 1280):
                continue
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
            d = 999.0 if last_g is None else float(np.abs(g - last_g).mean())
            if d < a.still:
                continue
            fn = f"{name}_{tt:07.2f}"
            open(out / "orig" / name / (fn + ".jpg"), "wb").write(buf)
            st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            w.writerow([fn, name, round(tt, 3), st, mission, round(d, 2)])
            kept += 1; last_t, last_g = tt, g
    fh.flush()
    print(f"{name}: 영상 {seen}장 중 {kept}장", flush=True)
fh.close()
