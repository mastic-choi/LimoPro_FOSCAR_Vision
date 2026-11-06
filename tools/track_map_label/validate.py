"""도면 라벨(추적 위치) vs 2차 HybridNets 예측(화이트밸런스 사진) 일치도 + 몽타주."""
import os, sys, json, glob, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
ROOT = config.LIMO_TRAIN; sys.path.insert(0, ROOT + "/scripts")
import make_model_compare_gif as m, train_seg as ts, render as R
model, *_r, fwd, _s = ts.build_hybridnets(m)
model.load_state_dict(torch.load(config.HYBRIDNETS_CKPT, map_location="cpu")); model = model.to(m.device).eval()
NEW = sorted(glob.glob(os.path.join(config.RAW_IMAGES, '*.jpg')))
# 사용: validate.py [poses 파일] [출력 접미사]. 기본은 results/poses_v2.json (없으면 poses.json)
PFILE = sys.argv[1] if len(sys.argv) > 1 else (SP + "/results/poses_v2.json" if os.path.exists(SP + "/results/poses_v2.json") else SP + "/results/poses.json")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
poses = json.load(open(PFILE)); print("poses:", PFILE)
def wb(img): g = img.reshape(-1, 3).astype(np.float32).mean(0); return np.clip(img * (g.mean() / (g + 1e-6)), 0, 255).astype(np.uint8)
def pred(img):
    x = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - m.IMAGENET_MEAN) / m.IMAGENET_STD
    with torch.no_grad(): da, ll = fwd(model, torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(m.device))
    return da[0, 12:-12].cpu().numpy() > .5, ll[0, 12:-12].cpu().numpy() > .5
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
rows, tiles = [], []
pick = set(int(k * (len(NEW) - 1) / 23) for k in range(24)) | {127, 159, 199, 254, 350, 446, 478, 509, 637}
for p in poses:
    i = p["i"]
    if p.get("x") is None: rows.append([i, np.nan, np.nan, np.nan, np.nan, np.nan]); continue
    img = cv2.resize(cv2.imread(NEW[i]), (640, 360), interpolation=cv2.INTER_AREA)
    da, ll = R.image_labels((p["x"], p["y"], p["yaw"]), p["pitch"])
    pda, pll = pred(img); wda, wll = pred(wb(img))
    rows.append([i, p["score"], iou(da, wda), iou(ll, wll), iou(da, pda), iou(ll, pll)])
    if i in pick:
        def ov(im, d, l):
            o = im.copy(); o[d] = (0.55 * o[d] + 0.45 * np.array([0, 200, 0])).astype(np.uint8); o[l] = (0, 0, 255); return o
        T = lambda x, t: cv2.putText(cv2.resize(x, (320, 180)), t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        tiles.append((i, np.hstack([T(img, f"{i:06d} score {p['score'] if p['score'] is None else round(p['score'])} {p.get('source', '')}"), T(ov(img, da, ll), "map label"), T(ov(img, pda, pll), f"round2 pred da {iou(da, pda):.2f}")])))
A = np.array(rows, float); np.save(SP + f"/results/agree{TAG}.npy", A)
ok = np.isfinite(A[:, 1])
for lo, hi in ((0, 25), (25, 40), (40, 999)):
    s = ok & (A[:, 1] > lo) & (A[:, 1] <= hi)
    if s.any(): print(f"점수 {lo}~{hi}: {s.sum()}장  도면라벨 vs 2차예측(WB) da IoU 중앙값 {np.nanmedian(A[s, 2]):.3f} ll {np.nanmedian(A[s, 3]):.3f} | 원본사진 예측 da {np.nanmedian(A[s, 4]):.3f} ll {np.nanmedian(A[s, 5]):.3f}")
tiles.sort(key=lambda t: t[0])
cv2.imwrite(SP + f"/results/validate_montage{TAG}.jpg", np.vstack([t[1] for t in tiles]), [cv2.IMWRITE_JPEG_QUALITY, 78])
print("몽타주", len(tiles))
