"""기존 학습 사진(10/27) 10장을 양방향 점수로 다시 위치 찾기 -> 원래 라벨 | 도면 겹침 | 도면 라벨."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
import localize as L, track as T, render as R
D = config.TRAIN_DATA
names = ["frame_001413.png", "frame_003133.png", "frame_000984.png", "frame_004073.png", "frame_003547.png",
         "frame_002577.png", "frame_005952.png", "frame_000539.png", "frame_005994.png", "frame_002750.png"]
def ov(im, d, l):
    o = im.copy(); o[d] = (0.55 * o[d] + 0.45 * np.array([0, 200, 0])).astype(np.uint8); o[l] = (0, 0, 255); return o
Tt = lambda x, t: cv2.putText(cv2.resize(x, (320, 180)), t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
rows = []
for k, n in enumerate(names, 1):
    img = cv2.imread(f"{D}/images/{n}")
    da0 = cv2.imread(f"{D}/da_masks/{n}", 0) > 0; ll0 = cv2.imread(f"{D}/ll_masks/{n}", 0) > 0
    mk, pts = L.markings(img); dto = T.obs_dt(pts)
    b, pose = T.global_init(pts, dto, 4.0)
    da, ll = R.image_labels(pose, 4.0)
    iou = (da & da0).sum() / max((da | da0).sum(), 1)
    print(f"{k}. {n}  양방향 점수 {b:.1f}  위치 x {pose[0]:.0f} y {pose[1]:.0f} {pose[2]:.0f}°  원래 라벨과 da IoU {iou:.2f}", flush=True)
    rows.append(np.hstack([Tt(ov(img, da0, ll0), f"{k}. {n} label"), Tt(L.overlay(img, pose), f"map  score {b:.0f}"), Tt(ov(img, da, ll), f"map label  da IoU {iou:.2f}")]))
cv2.imwrite(SP + "/results/train_conv_v2.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
