#!/usr/bin/env python3
"""README 대문 그림 만들기: 대회장 rosbag 프레임에 다중 작업 모델(HybridNets) 예측을 그린다 (11-19).

모든 그림은 실제 대회장(11-18) rosbag 프레임으로 만든다. 기본 bag 두 개는 v6 학습에 쓰지 않은 것이다.
  arall_1118_135742 : 출발 ~ 라바콘까지 전 코스 (신호등, 갈림길, 횡단보도, 과속방지턱, 천천히, 3차선, 라바콘)
  ar4_1118_135516   : 주차 미션
만드는 것 (--out 폴더):
  drive.gif         전 코스 주행 (약 2.7배속) + 주차
  missions.jpg      미션마다 한 장씩 3x3
  task_det.jpg      검출만 2x2
  task_da.jpg       주행가능영역(DA)만 2x2
  task_lane.jpg     차선 + 흰 가로선(정지선)만 2x2
  v5_vs_v6.gif      (--cmp-weights 를 주면) 신호등 진입 구간에서 v5 와 v6 나란히: v5 는 DA·차선이 꺼짐
프레임 폴더: bag_fork_frames.py --all 로 뽑은 <frames-root>/<bag>/frames/*.jpg + frames.csv (name, t, mission_id)
사용:
  <번들>/.venv/bin/python -I tools/mt_v4_data/readme_media.py --bundle <LimoPro_mt_v4> --weights <v6 best.pt> \
      --frames-root <label_work/fullbag1118> --out docs/images/readme [--cmp-weights <v5 best.pt>]
"""
import argparse, csv, os, shutil, subprocess, sys, tempfile

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--frames-root", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--cmp-weights", default=""); ap.add_argument("--tag", default="v6")
ap.add_argument("--course-bag", default="arall_1118_135742"); ap.add_argument("--park-bag", default="ar4_1118_135516")
ap.add_argument("--det-thr", type=float, default=0.4)
ap.add_argument("--picks", nargs="*", default=["2=arall_1118_135742_01285", "4=ar4_1118_135516_00090", "7=arall_1118_135742_01025", "8=arall_1118_135742_01709"],
                help="미션 칸 프레임 직접 지정 (미션=프레임 이름). 기본값은 11-19 에 눈으로 고른 것: 표지판이 보이는 갈림길, 사람 다리 없는 주차 칸, 흔들리지 않은 3차선·라바콘")
a = ap.parse_args()
OUT = os.path.abspath(a.out); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(a.bundle, "scripts"))
import train_multitask as tm, make_model_compare_gif as m  # noqa: E402
os.chdir(m.ROOT); sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
NAMES = ["pedestrian", "fork_sign", "-", "bump", "slow", "parking", "traffic_light", "red", "yellow", "green", "limo", "cone"]
BOX = {0: (255, 160, 0), 1: (255, 0, 200), 3: (0, 200, 255), 4: (0, 255, 120), 5: (255, 255, 0), 6: (220, 220, 220),
       7: (60, 60, 255), 8: (0, 220, 255), 9: (60, 255, 60), 10: (255, 120, 120), 11: (0, 140, 255)}
MIS = {"0": "lane", "1": "traffic light", "2": "fork (A/B sign)", "3": "crosswalk", "4": "parking", "5": "speed bump", "6": "slow", "7": "three-lane", "8": "cones"}
DA_C, LL_C, SL_C = np.array([70, 200, 70]), (60, 60, 255), np.array([0, 150, 255])
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()


def load(w):
    return tm.build_model(m, w).cuda().eval()


def frames(bag):
    R = os.path.join(a.frames_root, bag)
    return R, list(csv.DictReader(open(os.path.join(R, "frames.csv"))))


def infer(model, R, rows):
    """rows -> [(img640x360, da, ll, sl, dets)]"""
    res = []
    for k in range(0, len(rows), 32):
        ch = rows[k:k + 32]
        ims = [cv2.resize(cv2.imread(os.path.join(R, "frames", x["name"])), (640, 360), interpolation=cv2.INTER_AREA) for x in ch]
        x = torch.from_numpy(np.stack([((cv2.cvtColor(cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114,) * 3),
                                                      cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for im in ims])).cuda()
        with torch.no_grad():
            reg, cls, anchors, da, ll, sl = tm.heads(model, x)
            outs = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, a.det_thr, 0.5)
        f = lambda p: (p.float()[:, tm.PAD:-tm.PAD] >= 0.5).cpu().numpy()
        D, L, S = f(da), f(ll), f(sl)
        for q in range(len(ch)):
            dets = [((int(b[0]), int(b[1]) - tm.PAD, int(b[2]), int(b[3]) - tm.PAD), int(c), float(s))
                    for b, c, s in zip(outs[q]["rois"], outs[q]["class_ids"], outs[q]["scores"]) if int(c) not in (2, 10, 11)]
            res.append((ims[q], D[q], L[q], S[q], dets))
    return res


def draw(r, da=True, ll=True, sl=True, det=True, head="", foot=True):
    im, D, L, S, dets = r; o = im.copy()
    if da: o[D] = (0.55 * o[D] + 0.45 * DA_C).astype(np.uint8)
    if sl: o[S] = (0.35 * o[S] + 0.65 * SL_C).astype(np.uint8)
    if ll: o[L] = LL_C
    if det:
        for (x1, y1, x2, y2), c, s in dets:
            col = BOX.get(c, (255, 255, 255)); cv2.rectangle(o, (x1, y1), (x2, y2), col, 2)
            t = f"{NAMES[c]} {s:.2f}"; (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            y = max(y1, th + 4); cv2.rectangle(o, (x1, y - th - 4), (x1 + tw + 4, y), col, -1)
            cv2.putText(o, t, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    if head:
        cv2.rectangle(o, (0, 0), (640, 24), (20, 20, 20), -1); cv2.putText(o, head, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if foot:
        x0 = 6; cv2.rectangle(o, (0, 338), (640, 360), (20, 20, 20), -1)
        for name, col, on in (("drivable area", tuple(int(v) for v in DA_C), da), ("lane", LL_C, ll), ("stop line", tuple(int(v) for v in SL_C), sl), ("boxes", (220, 220, 220), det)):
            if not on: continue
            cv2.rectangle(o, (x0, 343), (x0 + 12, 355), col, -1); cv2.putText(o, name, (x0 + 16, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            x0 += 30 + 8 * len(name)
    return o


def grid(tiles, cols, gap=4):
    h, w = tiles[0].shape[:2]; rows = (len(tiles) + cols - 1) // cols
    g = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); g[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    return g


def gif(frames_bgr, path, fps, width):
    tmp = tempfile.mkdtemp(dir=OUT)
    for i, f in enumerate(frames_bgr): cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), f)
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"), "-vf", f"{vf},palettegen=stats_mode=diff:max_colors=160", os.path.join(tmp, "pal.png")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"), "-i", os.path.join(tmp, "pal.png"),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", path], check=True)
    shutil.rmtree(tmp); print(path, f"{os.path.getsize(path) / 1e6:.1f} MB, {len(frames_bgr)}프레임")


def save(img, name):
    p = os.path.join(OUT, name); cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(p)


model = load(a.weights)
Rc, rc = frames(a.course_bag); Rp, rp = frames(a.park_bag)
C = infer(model, Rc, rc); P = infer(model, Rp, rp)
ALL = [(a.course_bag, row, r) for row, r in zip(rc, C)] + [(a.park_bag, row, r) for row, r in zip(rp, P)]
mis = lambda row: MIS.get(row["mission_id"], "start")
t0c = float(rc[0]["t"]); t0p = float(rp[0]["t"])

# 1. 전 코스 주행 GIF (12 프레임마다 = 0.4 s, 8 fps 재생 -> 약 3.2 배속, README 에 맞게 10 MB 아래)
seq = []
for bag, rows, res, t0 in ((a.course_bag, rc, C, t0c), (a.park_bag, rp, P, t0p)):
    for i in range(0, len(rows), 12):
        seq.append(draw(res[i], head=f"venue rosbag {bag} (not in training)  t={float(rows[i]['t']) - t0:4.1f}s  mission: {mis(rows[i])}"))
gif(seq, os.path.join(OUT, "drive.gif"), 8, 480)


# 2. 미션마다 한 장 (검출 미션은 그 클래스가 크고 확실한 프레임, 나머지는 구간 가운데)
def box_score(r, cls):
    return max([s * min(1.0, (b[3] - b[1]) / 70) for b, c, s in r[4] if c in cls] or [0])


WANT = {"1": {7, 8, 9}, "2": {1}, "3": {0}, "4": {5}, "5": {3}, "6": {4}}
picks = {}
for mid in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
    cand = [(bag, row, r) for bag, row, r in ALL if row["mission_id"] == mid]
    if not cand: continue
    if mid in WANT:
        best = max(cand, key=lambda x: box_score(x[2], WANT[mid]) + (0.3 * box_score(x[2], {6}) if mid == "1" else 0)
                   - (0.5 if mid != "2" and box_score(x[2], {1}) > 0 else 0))   # 다른 미션에서 fork_sign 이 뜬 프레임은 피함
    else:
        best = cand[len(cand) // 2]
    picks[mid] = best
BYNAME = {row["name"][:-4]: (bag, row, r) for bag, row, r in ALL}
for sp in a.picks:
    k, n = sp.split("=", 1); picks[k] = BYNAME[n]
picks = dict(sorted(picks.items()))
tiles = [draw(r, head=f"mission {mid}: {MIS[mid]}", foot=False) for mid, (bag, row, r) in picks.items()]
save(grid(tiles, 3), "missions.jpg")


# 3. 작업별 2x2 (YOLOPv2 README 처럼)
def pick(mids):
    return [picks[x][2] for x in mids if x in picks][:4]


save(grid([draw(r, da=False, ll=False, sl=False, foot=False) for r in pick(["1", "2", "3", "6"])], 2), "task_det.jpg")
save(grid([draw(r, ll=False, sl=False, det=False, foot=False) for r in pick(["0", "2", "7", "8"])], 2), "task_da.jpg")
lane_mids = sorted(picks, key=lambda x: -picks[x][2][3].sum())[:2]   # 정지선이 가장 많이 보이는 두 미션
lane_mids += [x for x in ["0", "5", "6", "7"] if x not in lane_mids][:4 - len(lane_mids)]
save(grid([draw(r, da=False, det=False, foot=False) for r in pick(lane_mids)], 2), "task_lane.jpg")
print("미션별 프레임:", {k: (v[0], v[1]["name"]) for k, v in picks.items()}, "| 차선 그림 미션:", lane_mids)

# 4. v5 vs v6 (신호등 진입 구간): v5 는 이 bag 62.3 s 부터 DA·차선이 꺼짐
if a.cmp_weights:
    sel = [i for i, row in enumerate(rc) if 58.0 <= float(row["t"]) - t0c <= 69.0][::4]
    old = load(a.cmp_weights); O = infer(old, Rc, [rc[i] for i in sel]); seq = []
    for j, i in enumerate(sel):
        t = float(rc[i]["t"]) - t0c
        l = draw(O[j], head=f"v5 (withdrawn)  t={t:4.1f}s  DA {O[j][1][180:].mean():.2f}", foot=False)
        r = draw(C[i], head=f"{a.tag}  t={t:4.1f}s  DA {C[i][1][180:].mean():.2f}", foot=False)
        seq.append(np.hstack([l, np.full((360, 6, 3), 255, np.uint8), r]))
    gif(seq, os.path.join(OUT, "v5_vs_v6.gif"), 8, 760)
