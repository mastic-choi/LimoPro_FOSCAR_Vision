"""세션 전체 위치 추적. 점수 = 사진표시->도면 거리 + 도면표시(카메라에 보여야 할 것)->사진표시 거리 (양방향, mm, 80에서 자름)."""
import os, sys, glob, cv2, numpy as np, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
import cam, localize as L
dev = L.dev; RES = L.RES
PAINT = L.PAINT; Hm, Wm = PAINT.shape
ys, xs = np.nonzero(PAINT[::4, ::4]); MAPPTS = torch.tensor(np.c_[xs * 4 * RES, 6000 - ys * 4 * RES], device=dev, dtype=torch.float32)  # 20mm 간격 도면 표시 점
# 사진 바닥 BEV 격자 (차체 좌표): x 0.2~2.4m, y -1.3~1.3m, 10mm
BX0, BX1, BY0, BY1, BR = 0.2, 2.4, -1.3, 1.3, 0.01
gx, gy = np.meshgrid(np.arange(BX0, BX1, BR), np.arange(BY0, BY1, BR), indexing="ij")
def visible_mask(pitch):
    uv = cam.project(np.c_[gx.ravel(), gy.ravel()], pitch).reshape(gx.shape + (2,))
    ok = np.isfinite(uv).all(-1); u, v = np.nan_to_num(uv[..., 0]), np.nan_to_num(uv[..., 1])
    return ok & (u >= 2) & (u < 638) & (v >= 178) & (v < 358)
VIS = {}
def obs_dt(pts):
    img = np.ones(gx.shape, np.uint8)
    i = np.round((pts[:, 0] - BX0) / BR).astype(int); j = np.round((pts[:, 1] - BY0) / BR).astype(int)
    ok = (i >= 0) & (i < gx.shape[0]) & (j >= 0) & (j < gx.shape[1]); img[i[ok], j[ok]] = 0
    return cv2.distanceTransform(img, cv2.DIST_L2, 5) * BR * 1000  # mm
def score2(pts, dtobs, poses, pitch):
    """poses (M,3) [X mm, Y mm, P rad]"""
    a = L.score(pts[np.random.default_rng(0).permutation(len(pts))[:350]], poses[:, 0], poses[:, 1], poses[:, 2])
    vis = VIS.setdefault(pitch, torch.tensor(visible_mask(pitch), device=dev))
    D = torch.tensor(dtobs, device=dev, dtype=torch.float32)
    out = []
    for k in range(0, len(poses), 2000):
        P = poses[k:k + 2000]; c, s = torch.cos(P[:, 2])[:, None], torch.sin(P[:, 2])[:, None]
        dx = MAPPTS[None, :, 0] - P[:, 0:1]; dy = MAPPTS[None, :, 1] - P[:, 1:2]
        cx = (c * dx + s * dy) / 1000; cy = (-s * dx + c * dy) / 1000
        i = torch.round((cx - BX0) / BR).long(); j = torch.round((cy - BY0) / BR).long()
        inb = (i >= 0) & (i < gx.shape[0]) & (j >= 0) & (j < gx.shape[1])
        i, j = i.clamp(0, gx.shape[0] - 1), j.clamp(0, gx.shape[1] - 1)
        v = inb & vis[i, j]
        d = D[i, j].clamp(max=80.0)
        n = v.sum(1).clamp(min=1)
        b = torch.where(v, d, torch.zeros_like(d)).sum(1) / n
        b = torch.where(v.sum(1) < 30, torch.full_like(b, 80.0), b)
        out.append(b)
    return a + torch.cat(out)
def grid(center, sx, sp, pitch_list):
    cx, cy, cp = center
    xs_ = torch.arange(cx - sx, cx + sx + 1e-3, sx / 5, device=dev); ys_ = torch.arange(cy - sx, cy + sx + 1e-3, sx / 5, device=dev)
    ps_ = torch.deg2rad(torch.arange(cp - sp, cp + sp + 1e-6, sp / 5, device=dev))
    return torch.cartesian_prod(xs_, ys_, ps_)
def refine(pts, dtobs, pose, pitch, steps=((120, 10), (40, 3), (12, 1))):
    best = None
    for sx, sp in steps:
        G = grid(pose, sx, sp, None); sc = score2(pts, dtobs, G, pitch); v, j = sc.min(0)
        pose = (float(G[j, 0]), float(G[j, 1]), float(np.degrees(float(G[j, 2])))); best = float(v)
    return best, pose
def global_init(pts, dtobs, pitch, topk=60):
    sub = pts[np.random.default_rng(0).permutation(len(pts))[:350]]
    xs_ = torch.arange(100, 9900, 50.0, device=dev); ys_ = torch.arange(100, 5900, 50.0, device=dev); ps_ = torch.deg2rad(torch.arange(0, 360, 4.0, device=dev))
    G = torch.cartesian_prod(xs_, ys_, ps_); sc = torch.cat([L.score(sub, G[k:k + 50000, 0], G[k:k + 50000, 1], G[k:k + 50000, 2]) for k in range(0, len(G), 50000)])
    order = torch.argsort(sc)
    cands = []
    for j in order[:5000].tolist():  # 서로 떨어진 후보만
        p = G[j]
        if all(float(torch.hypot(p[0] - q[0], p[1] - q[1])) > 250 or abs(float(p[2] - q[2])) > 0.35 for q in cands): cands.append(p)
        if len(cands) >= topk: break
    res = [refine(pts, dtobs, (float(p[0]), float(p[1]), float(np.degrees(float(p[2])))), pitch) for p in cands]
    return min(res)
if __name__ == "__main__":
    pitch = 4.0
    frames = [int(x) for x in sys.argv[1].split(",")]
    tiles = []
    for i in frames:
        img = L.load(i); mk, pts = L.markings(img); dto = obs_dt(pts)
        b, pose = global_init(pts, dto, pitch)
        print(f"{i:06d} 양방향 점수 {b:.1f}  x {pose[0]:.0f} y {pose[1]:.0f} {pose[2]:.1f}°", flush=True)
        T = lambda x, t: cv2.putText(cv2.resize(x, (320, 180)), t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        tiles.append(np.hstack([T(img, f"{i:06d}"), T(L.overlay(img, pose), f"score {b:.0f} x{pose[0]:.0f} y{pose[1]:.0f} {pose[2]:.0f}")]))
    cv2.imwrite(SP + "/results/loc_probe2.jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
