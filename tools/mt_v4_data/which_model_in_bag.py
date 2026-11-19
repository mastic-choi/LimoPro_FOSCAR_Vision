#!/usr/bin/env python3
"""차량에 올라간 모델이 어느 버전인지 bag 으로 알아내기 (11-18, v5 사고 때 씀).

bag 에 기록된 차량 모델 출력 /ar/lane/sl_mask (정지선 마스크)를 후보 모델들의 정지선 출력과 프레임마다 비교해 IoU 를 낸다.
11-18 13시대 주행은 v5 와 IoU 0.99, v4 와는 중앙값 0 이었다.
사용: python -I tools/mt_v4_data/which_model_in_bag.py --bundle <LimoPro_mt_v4> --msgs <팀레포>/src/ar_msgs/msg --weights v4=<pt> v5=<pt> <bag 폴더> <프레임 폴더>
"""
import os, sys, glob, csv, bisect, cv2, numpy as np, torch
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bundle", required=True, help="학습 번들 폴더 (scripts/, models/HybridNets 가 있는 LimoPro_mt_v4)")
ap.add_argument("--msgs", required=True, help="ar_msgs/msg 폴더 (팀 레포 src/ar_msgs/msg)"); ap.add_argument("--weights", nargs="+", required=True, help="이름=best.pt ... (예: v4=... v5=...)")
ap.add_argument("bag", help="rosbag 폴더"); ap.add_argument("frames", help="같은 bag 의 프레임 폴더 (bag_fork_frames.py --all)")
A = ap.parse_args(); bag = Path(os.path.abspath(A.bag)); fr = os.path.abspath(A.frames)
ts = get_typestore(Stores.ROS2_HUMBLE); add = {}
for p in glob.glob(os.path.join(A.msgs, "*.msg")): add.update(get_types_from_msg(Path(p).read_text(), f"ar_msgs/msg/{Path(p).stem}"))
ts.register(add)
masks = []
with AnyReader([bag], default_typestore=ts) as r:
    con = [c for c in r.connections if c.topic == "/ar/lane/sl_mask"]
    for c, t, raw in r.messages(connections=con):
        m = r.deserialize(raw, c.msgtype); a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width]
        masks.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, a > 0))
print("sl_mask", len(masks), masks[0][1].shape, m.encoding)
rows = list(csv.DictReader(open(f"{fr}/frames.csv"))); st = [int(x["stamp_ns"]) for x in rows]
sys.path.insert(0, os.path.join(A.bundle, "scripts"))
import train_multitask as tm, make_model_compare_gif as mm
os.chdir(mm.ROOT)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
M = {w.split("=", 1)[0]: tm.build_model(mm, w.split("=", 1)[1]).cuda().eval() for w in A.weights}
pairs = []
for s, mk in masks[::5]:
    j = bisect.bisect_left(st, s)
    j = min([k for k in (j - 1, j) if 0 <= k < len(st)], key=lambda k: abs(st[k] - s))
    if abs(st[j] - s) < 5e6: pairs.append((j, mk))
agree = {v: [] for v in M}; robot_px = []
with torch.no_grad():
    for j, mk in pairs:
        im = cv2.resize(cv2.imread(f"{fr}/frames/{rows[j]['name']}"), (640, 360), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((cv2.cvtColor(cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114,) * 3), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]).cuda()
        rm = cv2.resize(mk.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST) > 0
        robot_px.append(rm.mean())
        for v in M:
            *_, sl = tm.heads(M[v], x); p = (sl[0, tm.PAD:-tm.PAD] >= 0.5).cpu().numpy()
            u = (p | rm).sum(); agree[v].append(((p & rm).sum() / u) if u > 50 else np.nan)
rp = np.array(robot_px); print("짝 지은 프레임", len(pairs), "| 차량 정지선 마스크가 있는 프레임", int((rp > 0.001).sum()))
for v in agree:
    a = np.array(agree[v]); print(f"{v}: 정지선 마스크 IoU (둘 중 하나라도 50px 넘는 프레임 {np.isfinite(a).sum()}장) 중앙값 {np.nanmedian(a):.3f}, 평균 {np.nanmean(a):.3f}")
