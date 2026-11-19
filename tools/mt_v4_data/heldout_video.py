#!/usr/bin/env python3
"""학습에 안 쓴 bag 을 다중 작업 모델로 돌려 확인 영상(mp4)과 16장 몽타주를 만든다 (11-18).

DA 초록, 차선 자홍, 정지선(흰 가로선) 주황, 검출 박스. 위에 bag 시간·미션·화면 아래 절반 DA 비율을 적는다.
사용: python -I tools/mt_v4_data/heldout_video.py --bundle <LimoPro_mt_v4> --weights <best.pt> --frames-root <fullbag1118> --out <폴더> arall_1118_135742 ar4_1118_135516
"""
import os, sys, csv, subprocess, cv2, numpy as np, torch
import argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bundle", required=True, help="학습 번들 폴더 (scripts/, models/HybridNets 가 있는 LimoPro_mt_v4)")
ap.add_argument("--weights", required=True); ap.add_argument("--frames-root", required=True, help="bag_fork_frames.py --all 출력 (<bag>/frames, frames.csv)")
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="v6"); ap.add_argument("bags", nargs="+")
A = ap.parse_args(); A.out = os.path.abspath(A.out)
sys.path.insert(0, os.path.join(A.bundle, "scripts"))
import train_multitask as tm, make_model_compare_gif as m
os.chdir(m.ROOT); sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets")); import utils.utils as util
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
model = tm.build_model(m, A.weights).cuda().eval()
reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
NAMES = ["pedestrian", "fork_sign", "unused", "bump", "slow", "parking", "traffic_light", "red", "yellow", "green", "limo", "cone"]
COL = {0: (255, 160, 0), 1: (255, 0, 200), 3: (0, 200, 255), 4: (0, 255, 120), 5: (255, 255, 0), 6: (200, 200, 200), 7: (0, 0, 255), 8: (0, 220, 255), 9: (0, 255, 0), 10: (255, 120, 120), 11: (0, 140, 255)}
MIS = {"-1": "start", "0": "lane", "1": "traffic light", "2": "fork", "3": "pedestrian", "4": "parking", "5": "bump", "6": "slow", "7": "three-lane", "8": "cones"}
OUT = A.out; os.makedirs(OUT, exist_ok=True)
def draw(im, da, ll, sl, dets, head):
    o = im.copy()
    o[da] = (0.6 * o[da] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
    o[sl] = (0.4 * o[sl] + 0.6 * np.array([0, 140, 255])).astype(np.uint8)
    o[ll] = (255, 0, 255)
    for (x1, y1, x2, y2), c, s in dets:
        col = COL.get(c, (255, 255, 255)); cv2.rectangle(o, (x1, y1), (x2, y2), col, 2)
        txt = f"{NAMES[c]} {s:.2f}"; (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(y1, th + 4); cv2.rectangle(o, (x1, y - th - 4), (x1 + tw + 4, y), col, -1); cv2.putText(o, txt, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(o, (0, 0), (640, 26), (20, 20, 20), -1); cv2.putText(o, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(o, "DA green | lane magenta | stop line orange | " + A.tag, (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return o
for bag in A.bags:
    R = os.path.join(A.frames_root, bag); rows = list(csv.DictReader(open(f"{R}/frames.csv")))[::2]
    tmp = f"{OUT}/{bag}_raw.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 360)); samples = []
    for k in range(0, len(rows), 32):
        ch = rows[k:k + 32]; ims = [cv2.resize(cv2.imread(f"{R}/frames/{x['name']}"), (640, 360), interpolation=cv2.INTER_AREA) for x in ch]
        x = torch.from_numpy(np.stack([((cv2.cvtColor(cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114,) * 3), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for im in ims])).cuda()
        with torch.no_grad():
            reg, cls, anchors, da, ll, sl = tm.heads(model, x)
            outs = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, 0.4, 0.5)
        f = lambda p: (p.float()[:, tm.PAD:-tm.PAD] >= 0.5).cpu().numpy()
        D, L, S = f(da), f(ll), f(sl)
        for q, row in enumerate(ch):
            dets = [((int(b[0]), int(b[1]) - tm.PAD, int(b[2]), int(b[3]) - tm.PAD), int(c), float(s)) for b, c, s in zip(outs[q]["rois"], outs[q]["class_ids"], outs[q]["scores"])]
            fr = draw(ims[q], D[q], L[q], S[q], dets, f"{bag} (not in training)  t={float(row['t']):5.1f}s  mission: {MIS.get(row['mission_id'], row['mission_id'])}  DA {D[q][180:].mean():.2f}")
            vw.write(fr); samples.append((float(row["t"]), row["mission_id"], fr))
    vw.release(); mp4 = f"{OUT}/{bag}_{A.tag}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True); os.remove(tmp)
    # 몽타주: 미션마다 고르게 16장
    idx = np.linspace(0, len(samples) - 1, 16).round().astype(int); tiles = [cv2.resize(samples[i][2], (480, 270)) for i in idx]
    cv2.imwrite(f"{OUT}/{bag}_{A.tag}_montage.jpg", np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 16, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(mp4, f"{len(samples)}프레임 {len(samples) / 15:.0f}초")
