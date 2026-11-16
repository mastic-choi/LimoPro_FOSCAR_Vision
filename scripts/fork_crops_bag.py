#!/usr/bin/env python3
"""실차 bag → 갈림길 분류기 크롭 (A / B / none).

한 번 주행(bag) 안에서는 갈림길 표지판이 바뀌지 않는다. 그래서 라벨은 bag 단위 표지판(A|B)과
"이 트랙이 그 표지판인가"만 정하면 된다. 트랙은 검출 박스를 시간순으로 IoU로 이어 붙인 것.

1) extract: M7 구간(앞뒤 여유)에서 표지판류 검출을 같은 stamp 의 압축 영상(1280x720)에서 자른다.
   크롭 규칙은 fork_crop_rule.crop_rect (로봇과 같음). 트랙별 몽타주 tracks/<id>.png, 한눈에 보기 tracks_overview.png.
     python scripts/fork_crops_bag.py extract <bag> <ar_msgs/msg 폴더> <out>
2) label: 표지판 트랙 번호를 주면 그 트랙은 <sign>/, 나머지는 none/ 으로 복사한다. --drop 트랙은 버린다.
     python scripts/fork_crops_bag.py label <out> --sign B --tracks 3,7 [--drop 5]

bag 은 ar.bash rec 녹화(/camera/color/image_raw/compressed, /ar/perception/detections, /ar/mission_state).
"""
import argparse, csv, glob, os, shutil, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fork_crop_rule as rule

CANDIDATES = {"sign_A", "sign_B", "slow", "parking", "pedestrian", "bump"}   # ar_fork_sign candidate_classes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def read_bag(bag, msgdir, mission, pad_before, pad_after, min_score):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    ts = get_typestore(Stores.ROS2_HUMBLE); add = {}
    for p in glob.glob(os.path.join(msgdir, "*.msg")):
        add.update(get_types_from_msg(Path(p).read_text(), f"ar_msgs/msg/{Path(p).stem}"))
    ts.register(add)
    topics = ("/ar/mission_state", "/ar/perception/detections", "/camera/color/image_raw/compressed")
    win, dets, imgs = [], [], {}
    with AnyReader([Path(bag)], default_typestore=ts) as r:
        t0 = r.start_time
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in topics]):
            m = r.deserialize(raw, c.msgtype); tt = (t - t0) / 1e9
            if c.topic == "/ar/mission_state":
                if m.mission_id == mission: win.append(tt)
                continue
            st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            if c.topic.endswith("compressed"):
                imgs[st] = bytes(m.data)
            else:
                dets.append((tt, st, m.image_width, m.image_height,
                             [(d.class_name, d.score, (d.x1, d.y1, d.x2, d.y2)) for d in m.detections
                              if d.class_name in CANDIDATES and d.score >= min_score]))
    if not win:
        sys.exit(f"미션 {mission} 구간 없음")
    a, b = win[0] - pad_before, win[-1] + pad_after
    return [d for d in dets if a <= d[0] <= b and d[4]], imgs, (win[0], win[-1])


def extract(args):
    dets, imgs, (a, b) = read_bag(args.bag, args.msgdir, args.mission, args.pad_before, args.pad_after, args.min_score)
    out = Path(args.out); (out / "raw").mkdir(parents=True, exist_ok=True); (out / "tracks").mkdir(exist_ok=True)
    tracks, rows, miss = [], [], 0   # track = [마지막 시각, 마지막 박스, id]
    for tt, st, dw, dh, ds in dets:
        if st not in imgs:
            miss += 1; continue
        img = cv2.imdecode(np.frombuffer(imgs[st], np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]; used = set()
        for k, (cls, score, box) in enumerate(sorted(ds, key=lambda d: -d[1])):
            best, bi = 0.0, None
            for i, tr in enumerate(tracks):
                if i in used or tt - tr[0] > args.max_gap: continue
                v = iou(tr[1], box)
                if v > best: best, bi = v, i
            if bi is None or best < args.iou:
                tracks.append([tt, box, len(tracks)]); bi = len(tracks) - 1
            else:
                tracks[bi][0], tracks[bi][1] = tt, box
            used.add(bi); tid = tracks[bi][2]
            x, y, s, _ = rule.crop_rect(box, dw, dh, W, H)
            f = f"t{tid:03d}_{tt:07.2f}_{k}_{cls}_{score:.2f}.png"
            cv2.imwrite(str(out / "raw" / f), img[y:y + s, x:x + s])
            rows.append([f, tid, round(tt, 3), cls, round(score, 3), *[round(v, 1) for v in box], x, y, s])
    with open(out / "crops.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["file", "track", "t", "class", "score", "x1", "y1", "x2", "y2", "cx", "cy", "cs"])
        wr.writerows(rows)
    # 트랙 몽타주
    by = {}
    for r_ in rows: by.setdefault(r_[1], []).append(r_)
    overview = []
    for tid, rs in sorted(by.items()):
        pick = [rs[int(i)] for i in np.linspace(0, len(rs) - 1, min(16, len(rs)))]
        tiles = []
        for r_ in pick:
            t = cv2.resize(cv2.imread(str(out / "raw" / r_[0])), (96, 96), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{r_[3][:6]} {r_[4]:.2f}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
            tiles.append(t)
        row = np.hstack(tiles + [np.zeros((96, 96, 3), np.uint8)] * (16 - len(tiles)))
        cv2.imwrite(str(out / "tracks" / f"{tid:03d}.png"), row)
        head = np.zeros((96, 150, 3), np.uint8)
        cls = {}
        for r_ in rs: cls[r_[3]] = cls.get(r_[3], 0) + 1
        cv2.putText(head, f"track {tid}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(head, f"n={len(rs)} {rs[0][2]:.1f}-{rs[-1][2]:.1f}s", (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for j, (k_, v) in enumerate(sorted(cls.items(), key=lambda kv: -kv[1])[:2]):
            cv2.putText(head, f"{k_} {v}", (4, 66 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        overview.append(np.hstack([head, row[:, :96 * 8]]))
    if overview:
        cv2.imwrite(str(out / "tracks_overview.png"), np.vstack(overview))
    print(f"미션 구간 {a:.1f}-{b:.1f} s, 크롭 {len(rows)}장, 트랙 {len(by)}개, 영상 못 찾음 {miss} 프레임 → {out}")


def label(args):
    out = Path(args.out)
    sign = {int(x) for x in args.tracks.split(",") if x}
    drop = {int(x) for x in args.drop.split(",") if x}
    for lab in (args.sign, "none"):
        d = out / lab
        if d.exists(): shutil.rmtree(d)
        d.mkdir()
    n = {args.sign: 0, "none": 0, "drop": 0}
    with open(out / "crops.csv") as fh:
        for r_ in csv.DictReader(fh):
            tid = int(r_["track"])
            if tid in drop: n["drop"] += 1; continue
            lab = args.sign if tid in sign else "none"
            shutil.copy(out / "raw" / r_["file"], out / lab / r_["file"]); n[lab] += 1
    print(n)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
e = sub.add_parser("extract")
e.add_argument("bag"); e.add_argument("msgdir"); e.add_argument("out")
e.add_argument("--mission", type=int, default=7)
e.add_argument("--pad-before", type=float, default=2.0); e.add_argument("--pad-after", type=float, default=2.0)
e.add_argument("--min-score", type=float, default=0.10)
e.add_argument("--iou", type=float, default=0.3); e.add_argument("--max-gap", type=float, default=0.4)
lb = sub.add_parser("label")
lb.add_argument("out"); lb.add_argument("--sign", choices=["A", "B"], required=True)
lb.add_argument("--tracks", required=True); lb.add_argument("--drop", default="")
args = ap.parse_args()
extract(args) if args.cmd == "extract" else label(args)
