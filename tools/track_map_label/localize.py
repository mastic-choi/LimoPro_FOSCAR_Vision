"""대회장 사진 1장 -> 트랙 도면 위 차 위치(x, y, 방향). 바닥의 밝은 표시를 BEV로 펴서 도면 표시와 거리변환으로 맞춤."""
import os, sys, glob, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK; import cam
PITCH = 4.0; RES = 5.0
dev = torch.device("cuda")
DT = torch.tensor(np.load(SP + "/map/dt_mm.npy"), device=dev)[None, None]  # 1,1,H,W (mm)
PAINT = np.load(SP + "/map/paint.npy")
Hm, Wm = PAINT.shape
NEW = sorted(glob.glob(os.path.join(config.RAW_IMAGES, '*.jpg')))
def load(i): return cv2.resize(cv2.imread(NEW[i]), (640, 360), interpolation=cv2.INTER_AREA)
def markings(img):
    L = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0])
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    floor = np.zeros_like(th, bool); floor[175:] = True
    t = max(25, np.percentile(th[floor], 94))
    mk = (th > t) & floor
    mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    ys, xs = np.nonzero(mk)
    g = cam.ipm(np.c_[xs, ys], PITCH)
    ok = np.isfinite(g).all(1) & (g[:, 0] < 2.2) & (np.abs(g[:, 1]) < 1.2)
    return mk, g[ok]
def score(pts_m, X, Y, P):
    """pts_m (N,2) 차체 m; X,Y (mm), P(rad) 텐서 (M,) -> 평균 잘린 거리(mm) (M,)"""
    p = torch.tensor(pts_m * 1000, device=dev, dtype=torch.float32)
    c, s = torch.cos(P)[:, None], torch.sin(P)[:, None]
    mx = X[:, None] + c * p[None, :, 0] - s * p[None, :, 1]
    my = Y[:, None] + s * p[None, :, 0] + c * p[None, :, 1]
    gx = mx / RES / (Wm - 1) * 2 - 1; gy = (6000 - my) / RES / (Hm - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]  # 1,M,N,2
    d = torch.nn.functional.grid_sample(DT, grid, align_corners=True, padding_mode="border")[0, 0]
    out = (gx.abs() > 1) | (gy.abs() > 1)
    return torch.where(out, torch.full_like(d, 80.0), d.clamp(max=80.0)).mean(1)
def search(pts, center=None, span=None):
    sub = pts[np.random.default_rng(0).permutation(len(pts))[:350]]
    if center is None:
        xs = torch.arange(100, 9900, 40.0, device=dev); ys = torch.arange(100, 5900, 40.0, device=dev); ps = torch.deg2rad(torch.arange(0, 360, 3.0, device=dev))
    else:
        (cx, cy, cp), (sx, sp_) = center, span
        xs = torch.arange(cx - sx, cx + sx + 1, sx / 6, device=dev); ys = torch.arange(cy - sx, cy + sx + 1, sx / 6, device=dev)
        ps = torch.deg2rad(torch.arange(cp - sp_, cp + sp_ + 1e-6, sp_ / 6, device=dev))
    G = torch.cartesian_prod(xs, ys, ps); best, bi = None, None
    for k in range(0, len(G), 40000):
        sc = score(sub, G[k:k + 40000, 0], G[k:k + 40000, 1], G[k:k + 40000, 2])
        v, j = sc.min(0)
        if best is None or v < best: best, bi = float(v), G[k + int(j)].tolist()
    return best, (bi[0], bi[1], float(np.degrees(bi[2])))
def localize(pts):
    b, pose = search(pts)
    for sx, sp_ in ((80, 6), (30, 2), (10, 0.75)):
        b, pose = search(pts, pose, (sx, sp_))
    return b, pose
def overlay(img, pose, pitch=PITCH):
    X, Y, P = pose; P = np.radians(P)
    ys, xs = np.nonzero(PAINT); mx, my = xs * RES, 6000 - ys * RES
    dx, dy = mx - X, my - Y
    car = np.c_[np.cos(P) * dx + np.sin(P) * dy, -np.sin(P) * dx + np.cos(P) * dy] / 1000
    keep = (car[:, 0] > 0.25) & (car[:, 0] < 3.0) & (np.abs(car[:, 1]) < 1.5)
    uv = cam.project(car[keep], pitch); uv = uv[np.isfinite(uv).all(1)]
    o = img.copy(); u, v = np.round(uv).astype(int).T; ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 360)
    o[v[ok], u[ok]] = (255, 0, 255)
    return o
if __name__ == "__main__":
    tiles = []
    for i in [int(x) for x in sys.argv[2].split(",")]:
        img = load(i); mk, pts = markings(img); b, pose = localize(pts)
        print(f"{i:06d}  점 {len(pts)}  위치 x {pose[0]:.0f} y {pose[1]:.0f} mm  방향 {pose[2]:.1f}°  평균 거리오차 {b:.1f} mm", flush=True)
        T = lambda x, t: cv2.putText(cv2.resize(x, (320, 180)), t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        mkv = cv2.cvtColor((mk * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tiles.append(np.hstack([T(img, f"{i:06d}"), T(mkv, f"markings {len(pts)}"), T(overlay(img, pose), f"map err {b:.0f}mm x{pose[0]:.0f} y{pose[1]:.0f} {pose[2]:.0f}deg")]))
    cv2.imwrite(SP + "/results/loc_probe.jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
