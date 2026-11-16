"""세션 전체 위치 추적: 후보를 여러 개 모으고 Viterbi로 전체가 물리적으로 이어지는 경로 하나를 고른다.

  python track_session.py          # 후보 추적(약 20분) -> results/candidates.json -> 경로 고르기 -> results/poses_v2.json
  python track_session.py --reuse  # candidates.json을 다시 써서 경로 고르기만 (몇 분)

1) 10장마다 전역 탐색 -> 서로 400mm 이상 떨어진 상위 후보 2개 (직선 구간처럼 비슷한 곳이 여러 군데라서)
2) 후보마다 앞뒤 25장씩 이어 추적. 모든 결과를 프레임별 후보로 저장
3) Viterbi. 비용 = 위치 점수/5 + 이동 벌점 + 건너뛴 프레임 x SKIP
   이동 벌점: 프레임당 MOVE mm, YAW°까지는 0, 넘으면 각각 /8, /2.
   SKIP(12)은 점수 60인 후보를 쓰는 비용보다 커서, 후보가 있는 프레임은 거의 다 경로에 들어간다.
4) 경로에서 빠진 프레임: 앞뒤 위치를 보간한 뒤 그 근처(±60mm, ±6°)에서 다시 맞춤
결과 source: tracked(경로) | interp(보간 후 다시 맞춤) | interp_raw(보간값 그대로, 단서 부족)
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config; SP = config.WORK
import cam, localize as L, track as T
dev = L.dev
PITCHES = [3.5, 4.0, 4.5]
MOVE, YAW, SKIP, MAXSKIP = 75.0, 12.0, 12.0, 90
REUSE = "--reuse" in sys.argv
CAND_FILE = SP + "/results/candidates.json"
N = len(L.NEW); t0 = time.time()
MK = [None] * N
def marks(i):
    if MK[i] is None:
        mk, _ = L.markings(L.load(i)); ys, xs = np.nonzero(mk); MK[i] = np.c_[xs, ys]
    return MK[i]
cache = {}
def obs(i, pitch):
    k = (i, pitch)
    if k not in cache:
        g = cam.ipm(marks(i), pitch); ok = np.isfinite(g).all(1) & (g[:, 0] < 2.2) & (np.abs(g[:, 1]) < 1.2); p = g[ok]
        cache[k] = (p, T.obs_dt(p))
    return cache[k]
def wrap(a): return (a + 180) % 360 - 180
def track_step(i, pose, steps=((90, 9), (30, 3), (10, 1))):
    best = None
    for pitch in PITCHES:
        p, d = obs(i, pitch)
        if len(p) < 150: continue
        b, ps = T.refine(p, d, pose, pitch, steps)
        if best is None or b < best[0]: best = (b, ps, pitch)
    return best
def global_cands(i, k=2):
    p, d = obs(i, 4.0)
    if len(p) < 300: return []
    sub = p[np.random.default_rng(0).permutation(len(p))[:350]]
    xs_ = torch.arange(100, 9900, 50.0, device=dev); ys_ = torch.arange(100, 5900, 50.0, device=dev); ps_ = torch.deg2rad(torch.arange(0, 360, 4.0, device=dev))
    G = torch.cartesian_prod(xs_, ys_, ps_)
    sc = torch.cat([L.score(sub, G[j:j + 50000, 0], G[j:j + 50000, 1], G[j:j + 50000, 2]) for j in range(0, len(G), 50000)])
    cands = []
    for j in torch.argsort(sc)[:6000].tolist():
        q = G[j]
        if all(float(torch.hypot(q[0] - c[0], q[1] - c[1])) > 250 or abs(float(q[2] - c[2])) > 0.35 for c in cands): cands.append(q)
        if len(cands) >= 60: break
    res = sorted(T.refine(p, d, (float(q[0]), float(q[1]), float(np.degrees(float(q[2])))), 4.0) for q in cands)
    out = []
    for b, ps in res:
        if all(np.hypot(ps[0] - o[1][0], ps[1] - o[1][1]) > 400 for o in out): out.append((b, ps))
        if len(out) >= k: break
    return out

CAND = [[] for _ in range(N)]  # (score, (x, y, yaw), pitch)
def add(i, r):
    if r is None: return
    r = (float(r[0]), tuple(float(v) for v in r[1]), float(r[2]))
    for c in CAND[i]:
        if np.hypot(c[1][0] - r[1][0], c[1][1] - r[1][1]) < 30 and abs(wrap(c[1][2] - r[1][2])) < 4:
            if r[0] < c[0]: CAND[i].remove(c); CAND[i].append(r)
            return
    CAND[i].append(r)

if REUSE and os.path.exists(CAND_FILE):
    CAND = [[(c[0], tuple(c[1]), c[2]) for c in cs] for cs in json.load(open(CAND_FILE))]
    print("후보 불러옴", CAND_FILE, flush=True)
else:
    for a in range(0, N, 10):
        for b, pose in global_cands(a):
            if b > 36: continue
            r = track_step(a, pose, ((30, 3), (10, 1))); add(a, r)
            if r is None or r[0] > 32: continue
            for direction in (1, -1):
                cur, bad = r[1], 0
                for i in range(a + direction, a + direction * 26, direction):
                    if i < 0 or i >= N: break
                    s = track_step(i, cur)
                    if s is None: break
                    add(i, s); cur = s[1]; bad = bad + 1 if s[0] > 40 else 0
                    if bad >= 3: break
        if a % 50 == 0: print(f"앵커 {a:06d}까지 {round(time.time() - t0)} s, 후보 평균 {np.mean([len(c) for c in CAND[:a + 1]]):.1f}", flush=True)
    json.dump(CAND, open(CAND_FILE, "w"))
# 1차 추적(poses.json) 결과도 후보로 넣음
if os.path.exists(SP + "/results/poses.json"):
    for p in json.load(open(SP + "/results/poses.json")):
        if p.get("score") is not None and p["score"] <= 40: add(p["i"], (p["score"], (p["x"], p["y"], p["yaw"]), p["pitch"]))

# Viterbi (프레임 건너뛰기 허용)
frames = [i for i in range(N) if CAND[i]]
def trans(c1, c2, g):
    d = np.hypot(c2[1][0] - c1[1][0], c2[1][1] - c1[1][1]); dy = abs(wrap(c2[1][2] - c1[1][2]))
    return max(0.0, d - MOVE * g) / 8 + max(0.0, dy - YAW * g) / 2 + SKIP * (g - 1)
cost, back = {}, {}
for n, i in enumerate(frames):
    prev = [j for j in frames[max(0, n - MAXSKIP):n] if i - j <= MAXSKIP]
    for k, c in enumerate(CAND[i]):
        best, arg = c[0] / 5 + SKIP * i, None
        for j in prev:
            for m, c0 in enumerate(CAND[j]):
                v = cost[(j, m)] + trans(c0, c, i - j) + c[0] / 5
                if v < best: best, arg = v, (j, m)
        cost[(i, k)] = best; back[(i, k)] = arg
end = min((v + SKIP * (N - 1 - key[0]), key) for key, v in cost.items())[1]
path = []
while end is not None: path.append(end); end = back[end]
chosen = {i: CAND[i][k] for i, k in path[::-1]}
jumps = [(a, b) for a, b in zip(sorted(chosen), sorted(chosen)[1:]) if np.hypot(chosen[b][1][0] - chosen[a][1][0], chosen[b][1][1] - chosen[a][1][1]) > MOVE * (b - a) + 150]
print(f"Viterbi 경로 {len(chosen)}장 / 후보 있는 프레임 {len(frames)}장, 큰 점프 {len(jumps)}번 {jumps[:10]}  {round(time.time() - t0)} s", flush=True)

out, keys = [], sorted(chosen)
for i in range(N):
    if i in chosen:
        s, ps, pitch = chosen[i]
        out.append({"i": i, "x": round(ps[0], 1), "y": round(ps[1], 1), "yaw": round(ps[2] % 360, 2), "pitch": pitch, "score": round(s, 2), "source": "tracked"}); continue
    prev = max([k for k in keys if k < i], default=None); nxt = min([k for k in keys if k > i], default=None)
    if prev is None or nxt is None:
        pred = np.array(chosen[prev if prev is not None else nxt][1], float)
    else:
        w = (i - prev) / (nxt - prev); a, b = np.array(chosen[prev][1], float), np.array(chosen[nxt][1], float)
        pred = np.r_[(1 - w) * a[:2] + w * b[:2], (a[2] + w * wrap(b[2] - a[2])) % 360]
    r = track_step(i, tuple(pred), ((60, 6), (20, 2), (8, 0.8)))
    if r is not None and r[0] <= 40 and np.hypot(r[1][0] - pred[0], r[1][1] - pred[1]) < 120:
        out.append({"i": i, "x": round(r[1][0], 1), "y": round(r[1][1], 1), "yaw": round(r[1][2] % 360, 2), "pitch": r[2], "score": round(r[0], 2), "source": "interp"})
    else:
        out.append({"i": i, "x": round(float(pred[0]), 1), "y": round(float(pred[1]), 1), "yaw": round(float(pred[2]), 2), "pitch": 4.0,
                    "score": None if r is None else round(r[0], 2), "source": "interp_raw"})
json.dump(out, open(SP + "/results/poses_v2.json", "w"))
src = [o["source"] for o in out]; sc = np.array([o["score"] if o["score"] is not None else np.nan for o in out])
print("tracked", src.count("tracked"), "interp", src.count("interp"), "interp_raw", src.count("interp_raw"),
      "| 점수<=25", int((sc <= 25).sum()), "25~40", int(((sc > 25) & (sc <= 40)).sum()), ">40", int((sc > 40).sum()), "| 총", round(time.time() - t0), "s")
