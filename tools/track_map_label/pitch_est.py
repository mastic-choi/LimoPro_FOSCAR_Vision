import sys, os, glob, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
ROOT = config.LIMO_TRAIN; sys.path.insert(0, ROOT + "/scripts")
import make_model_compare_gif as m, train_seg as ts, cam
model, *_r, fwd, _s = ts.build_hybridnets(m)
model.load_state_dict(torch.load(config.HYBRIDNETS_CKPT, map_location="cpu")); model = model.to(m.device).eval()
NEW = sorted(glob.glob(os.path.join(config.RAW_IMAGES, '*.jpg')))
def load(i): return cv2.resize(cv2.imread(NEW[i]), (640, 360), interpolation=cv2.INTER_AREA)
def wb(img): g = img.reshape(-1, 3).astype(np.float32).mean(0); return np.clip(img * (g.mean() / (g + 1e-6)), 0, 255).astype(np.uint8)
def pred(img):
    x = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - m.IMAGENET_MEAN) / m.IMAGENET_STD
    with torch.no_grad(): da, ll = fwd(model, torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(m.device))
    return da[0, 12:-12].cpu().numpy() > .5, ll[0, 12:-12].cpu().numpy() > .5
def lines_of(ll):
    n, lab, st, _ = cv2.connectedComponentsWithStats(ll.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 150: continue
        ys, xs = np.nonzero(lab == k); keep = ys > 165  # 너무 먼 곳 제외
        if keep.sum() < 80: continue
        out.append(np.c_[xs[keep], ys[keep]].astype(np.float64))
    return out
res = {}
for i in [int(x) for x in sys.argv[2].split(",")]:
    img = load(i); _, ll = pred(wb(img)); comps = lines_of(ll)
    best = None
    for p in np.arange(-4, 16.01, 0.25):
        dirs, offs = [], []
        for c in comps:
            g = cam.ipm(c, p); g = g[np.isfinite(g).all(1) & (g[:, 0] < 2.0)]
            if len(g) < 40: continue
            mu = g.mean(0); u, s, vt = np.linalg.svd(g - mu, full_matrices=False)
            dvec = vt[0] * np.sign(vt[0][0]); dirs.append(dvec); offs.append((mu, s[1] / max(s[0], 1e-6)))
        if len(dirs) < 2: continue
        ang = np.degrees(np.arctan2([d[1] for d in dirs], [d[0] for d in dirs]))
        score = np.std(ang) + 20 * np.mean([o[1] for o in offs])  # 평행 + 각 선이 곧은 정도
        if best is None or score < best[0]:
            mean_dir = np.array([np.cos(np.radians(ang.mean())), np.sin(np.radians(ang.mean()))]); nrm = np.array([-mean_dir[1], mean_dir[0]])
            lat = sorted(float(o[0] @ nrm) for o in offs)
            best = (score, p, ang.mean(), np.diff(lat) * 1000)
    res[i] = best
    print(f"{i:06d} 선 {len(comps)}개  최적 pitch {best[1]:.2f}°  차 방향 대비 차선 각도 {best[2]:.1f}°  선 간격 mm {np.round(best[3]).astype(int).tolist()}  점수 {best[0]:.2f}")
