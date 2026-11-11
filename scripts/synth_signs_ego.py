#!/usr/bin/env python3
"""ego_v4(차선·주행영역 마스크가 있는 프레임)에 표지판·신호등을 합성해 분할+검출 라벨을 한 장에 같이 갖는
HybridNets 학습 데이터(synth_signs_v2)를 만든다.

synth_signs.py(v1)와 같은 합성기(원근·조명색·흐림·가림)를 쓰고, 달라진 점은
  - 배경 = ego_v4의 각 split(train/val/test/test_venue/test_bag) 프레임. 마스크(da/ll)를 같이 복사
  - 기둥 밑동: 트랙 끝이 보이면 트랙 바깥, 아니면 ego 차선 선 바로 옆, 12%는 화면 끝 잘림 (place 참고)
  - 물체가 덮은 픽셀은 da/ll 마스크에서 지운다
  - 대회장/일반 비율은 맞추지 않는다(ego_v4 자연 비율). 클래스 균형만 맞춘다(가장 적게 나온 클래스부터)
  - train은 --train-copies번 다른 물체로 합성(파일명 _s0, _s1 ...), 나머지 split은 1번
  - --empty 비율만큼은 물체 없이 그대로 둔다(음성 샘플)
출력 <out>/<split>/{images,da_masks,ll_masks,labels}/<name>_s<k>.png|txt, classes.txt, meta.csv, counts.json
사용 예:
  .venv/Scripts/python scripts/synth_signs_ego.py --ego dataset/ego_v4 --out dataset/synth_signs_v2
"""
import argparse, csv, json, os, random, sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_signs as ss  # noqa: E402

ROOT = ss.ROOT


def pool_of(name):
    return "venue" if name[:1] in ("b", "v") else "general"


def track_edges(hsv, da, r):
    """r행에서 ego 차선(da)부터 바깥으로 걸어가며 트랙 매트(어두운 매트 또는 노란 선)가 끝나는 x를 찾는다.
    (왼쪽 끝, 오른쪽 끝) — 화면 끝까지 매트면 None. ego_v4의 da는 지금 달리는 차선만 칠해져 있어서
    da 밖이라도 옆 차선(매트 위)일 수 있기 때문에 따로 찾는다."""
    H, W = da.shape
    on = np.where(da[r] > 0)[0]
    if len(on) == 0:
        return None, None
    band = slice(max(0, r - 2), min(H, r + 3))
    v, s, h = (hsv[band, :, i].mean(0) for i in (2, 1, 0))
    vmat = np.median(hsv[r, on, 2])
    mat = (v < vmat * 1.5 + 20) | ((h > 10) & (h < 40) & (s > 70))

    def walk(x, step):
        run, last = 0, x
        while 0 <= x < W:
            if mat[x]:
                run, last = 0, x
            else:
                run += 1
                if run >= max(6, W // 80):
                    return last
            x += step
        return None
    return walk(on.min(), -1), walk(on.max(), 1)


def place(o, img_hsv, da, ll, rng):
    """o의 d/vh는 그대로 두고 기둥 밑동 위치 u(1280 기준)를 정한다.
    1) 트랙 끝이 보이면 트랙 바깥 3~35cm  2) 안 보이면 ego 차선 선 바로 옆 3~12cm  3) 12%는 화면 끝 잘림"""
    H, W = da.shape
    f = W / 1280.0
    fx = ss.FX1280 * f
    if rng.random() < 0.12:
        o["u"] = float(rng.choice([rng.uniform(-0.1, 0.03), rng.uniform(0.97, 1.1)]) * 1280)
        return "edge"
    vg = o["vh"] * f + fx * ss.CAM_H / o["d"]
    r = int(np.clip(vg, 0, H - 1))
    ppm = fx / o["d"]
    xs = np.arange(W)
    road = (da[r] > 0) | (ll[r] > 0)
    if not road.any():
        o["u"] = float(rng.uniform(0, 1) * 1280)
        return "free"
    le, re_ = track_edges(img_hsv, da, r)
    cand = []
    if le is not None:
        cand += list(xs[(xs <= le - 0.03 * ppm) & (xs >= le - 0.35 * ppm)])
    if re_ is not None:
        cand += list(xs[(xs >= re_ + 0.03 * ppm) & (xs <= re_ + 0.35 * ppm)])
    if cand:
        o["u"] = float(rng.choice(cand) / f)
        return "trackside"
    on = np.where(road)[0]
    side = rng.random() < 0.5
    edge = on.min() if side else on.max()
    off = rng.uniform(0.03, 0.12) * ppm
    o["u"] = float((edge - off if side else edge + off) / f)
    return "laneside"


MONITOR_COLORS = [(40, 40, 230), (30, 200, 240), (60, 210, 60), (220, 120, 30), (235, 235, 235),
                  (200, 60, 200), (40, 140, 240), (120, 230, 200), (20, 20, 20)]   # BGR: 빨강 노랑 초록 파랑 흰 보라 주황 연두 검정


def draw_monitor(frame, u1280, d, vh1280, color, rng, omask):
    """신호등 뒤 방해 모니터(규정집 미션 1): 단색 화면 + 검은 테두리 + 받침대. 화면은 자체 발광, 테두리는 조명색 반사.
    폭 0.35~0.6m, 16:9, 바닥에서 0.05~0.15m 위. 라벨 없음(음성 샘플). 그린 픽셀은 omask에 표시."""
    H, W = frame.shape[:2]
    f = W / 1280; fx = ss.FX1280 * f; ppm = fx / d
    wm = rng.uniform(0.35, 0.6); hm = wm * 9 / 16; h0 = rng.uniform(0.05, 0.15); bz = 0.02
    u, vh = u1280 * f, vh1280 * f
    top = vh + fx * (ss.CAM_H - (h0 + hm + bz)) / d
    bot = vh + fx * (ss.CAM_H - h0) / d
    x0, x1 = u - (wm / 2 + bz) * ppm, u + (wm / 2 + bz) * ppm
    vg = vh + fx * ss.CAM_H / d
    X0, Y0, X1, Y1 = int(max(0, x0)), int(max(0, top)), int(min(W, x1)), int(min(H, vg + 2))
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return False
    roi = frame[Y0:Y1, X0:X1].astype(np.float32)
    rh, rw = roi.shape[:2]
    layer = np.zeros((rh, rw, 3), np.float32); al = np.zeros((rh, rw), np.float32)
    light = ss.scene_light(frame, (x0, top, x1, bot), rng)
    sw, cx = max(2, int(0.03 * ppm)), int(u - X0)
    cv2.rectangle(layer, (cx - sw // 2, int(bot - Y0)), (cx + sw // 2, int(vg - Y0)), (35, 35, 35), -1)
    cv2.rectangle(al, (cx - sw // 2, int(bot - Y0)), (cx + sw // 2, int(vg - Y0)), 1, -1)
    bx0, by0, bx1, by1 = int(x0 - X0), int(top - Y0), int(x1 - X0), int(bot - Y0)
    cv2.rectangle(layer, (bx0, by0), (bx1, by1), tuple(float(v) for v in np.array([22, 22, 24]) * light * 2), -1)
    cv2.rectangle(al, (bx0, by0), (bx1, by1), 1, -1)
    b = int(bz * ppm) + 1
    col = np.array(color, np.float32) * rng.uniform(0.7, 1.05)
    yy = np.linspace(0, 1, max(1, by1 - by0 - 2 * b))[:, None, None]
    scr = col * (1 - 0.15 * rng.random() * yy)          # 화면 위아래로 약간의 밝기 차
    sy0, sy1, sx0, sx1 = max(0, by0 + b), min(rh, by1 - b), max(0, bx0 + b), min(rw, bx1 - b)
    if sy1 > sy0 and sx1 > sx0:
        layer[sy0:sy1, sx0:sx1] = scr[sy0 - (by0 + b):sy1 - (by0 + b)] + light * 8
    sig = rng.uniform(0.6, 1.3)
    pre = cv2.GaussianBlur(layer * al[..., None], (0, 0), sig)
    a2 = cv2.GaussianBlur(al, (0, 0), sig)[..., None]
    c2 = np.where(a2 > 1e-3, pre / np.maximum(a2, 1e-3), 0)
    out = roi * (1 - a2) + c2 * a2 + rng.normal(0, 2.5, roi.shape) * (a2 > 0.01)
    if a2.max() > 0:   # 화면 빛 번짐
        glow = cv2.GaussianBlur((a2[..., 0] > 0.5).astype(np.float32), (0, 0), max(2.0, 0.03 * ppm))[..., None]
        out += glow * col * rng.uniform(0.05, 0.2)
    frame[Y0:Y1, X0:X1] = np.clip(out, 0, 255).astype(np.uint8)
    omask[Y0:Y1, X0:X1] |= a2[..., 0] > 0.5
    return True


def draw_cable(frame, vh1280, rng):
    """트랙 위에 놓인 흰 전원 케이블(실차 bag에서 흰 가로선으로 오탐). 부드러운 곡선, 가까울수록 굵게.
    반환: 케이블 주변 마스크(이 픽셀은 '흰 가로선 아님'으로 학습)."""
    H, W = frame.shape[:2]
    vh = vh1280 * W / 1280
    y_top = int(np.clip(vh + 0.12 * (H - vh), 0, H - 1))
    p = np.array([[rng.uniform(-0.2, 0.3) * W, rng.uniform(y_top, H)], [rng.uniform(0, 1) * W, rng.uniform(y_top, H)],
                  [rng.uniform(0, 1) * W, rng.uniform(y_top, H)], [rng.uniform(0.7, 1.2) * W, rng.uniform(y_top, H)]])
    if rng.random() < 0.5:
        p = p[::-1]
    t = np.linspace(0, 1, 200)[:, None]
    pts = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])
    m = np.zeros((H, W), np.float32)
    base = rng.uniform(0.006, 0.012)   # 굵기(화면 높이 비율, 화면 아래 기준)
    for a_, b_ in zip(pts[:-1], pts[1:]):
        th = max(1, int(base * H * (b_[1] - vh) / max(H - vh, 1) * 2))
        cv2.line(m, tuple(int(v) for v in a_), tuple(int(v) for v in b_), 1.0, th, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    col = np.array(random.choice([(235, 235, 235), (215, 220, 225), (200, 200, 200), (230, 230, 210)]), np.float32)
    light = ss.scene_light(frame, (0, y_top, W, H), rng)
    shadow = cv2.GaussianBlur(np.roll(m, 2, 0), (0, 0), 2) * 0.35
    f = frame.astype(np.float32) * (1 - shadow[..., None])
    f = f * (1 - m[..., None]) + (col * light)[None, None] * m[..., None]
    frame[:] = np.clip(f, 0, 255).astype(np.uint8)
    return cv2.dilate((m > 0.2).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def make_one(job):
    split, name, k, targets, seed, ego, out = job[:7]
    opt = job[7] if len(job) > 7 else {}
    rng = np.random.default_rng(seed)
    random.seed(seed)
    A = globals()["A"]
    img = cv2.imread(os.path.join(ego, split, "images", name))
    da = cv2.imread(os.path.join(ego, split, "da_masks", name), 0)
    ll = cv2.imread(os.path.join(ego, split, "ll_masks", name), 0)
    omask = np.zeros(img.shape[:2], bool)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    labs, metas, pending = [], [], []
    for t in targets:
        for _ in range(10):
            o = ss.sample_object(t, rng, A)
            if rng.random() < opt.get("near_bias", 0.0):   # 차에서 판단에 쓰는 거리(약 2m 이내) 비중을 높임
                o["d"] = float(rng.uniform(0.4, 2.0))
            how = place(o, hsv, da, ll, rng)
            trial, tm = img.copy(), np.zeros_like(omask)
            mon = None
            if t in ss.LIGHTS and rng.random() < opt.get("monitor_p", 0.0):
                # 신호등 바로 뒤 단색 모니터. 모니터는 트랙 위에 두지 않으므로 차선 옆 배치면 바깥으로 더 밀어냄
                f = da.shape[1] / 1280
                d_m = o["d"] + rng.uniform(0.1, 0.35)
                side = 1 if o["u"] > 640 else -1
                if how in ("laneside", "free"):
                    o["u"] = float(o["u"] + side * 0.35 * ss.FX1280 / o["d"])
                u_m = o["u"] + side * rng.uniform(0.0, 0.12) * ss.FX1280 / d_m
                mm = np.zeros_like(omask)
                if not draw_monitor(trial, u_m, d_m, o["vh"], MONITOR_COLORS[int(rng.integers(len(MONITOR_COLORS)))], rng, mm):
                    continue
                if (mm & (da > 0)).sum() > 0.05 * max(mm.sum(), 1):   # 지금 차선 위로 걸치면 다시 뽑기
                    continue
                tm |= mm
                mon = "monitor"
            got = ss.draw_object(trial, o, rng, tm)
            key = [g for g in got if g[0] == t]
            main = [g for g in got if g[0] == t or g[0] == "traffic_light"]
            if not key or key[0][5] < 0.35:
                continue
            if any(ss.overlap(g[1:5], l[1:5]) > 0.1 for g in main for l in labs):
                continue
            img, omask = trial, omask | tm
            labs += got
            if o["occlude"] > 0:
                box = [g for g in got if g[0] == "traffic_light" or g[0] in ss.SIGNS][0][1:5]
                pending.append((len(metas), o, box))
            metas.append(dict(cls=t, d=round(o["d"], 2), u=round(o["u"]), place=how,
                              occ=round(o["occlude"], 2), occ_type="", vis=round(key[0][5], 2), monitor=mon))
            break
    for mi, o, box in pending:
        extra, metas[mi]["occ_type"] = ss.add_occluder(img, o, box, rng, A, omask)
        labs += extra
    neg = None
    if rng.random() < opt.get("cable_p", 0.0):   # 흰 케이블 1~2개: 그 자리는 '흰 가로선 아님'
        neg = np.zeros(img.shape[:2], bool)
        for _ in range(int(rng.integers(1, 3))):
            neg |= draw_cable(img, float(rng.uniform(262, 335)), rng)
        metas.append(dict(cls="cable"))
    # 압축 손실 흉내(배경은 원래 PNG라 JPEG 한 번 거쳐 물체와 질감을 맞춘다)
    q = int(rng.integers(75, 96))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(enc, 1)
    da = np.where(omask, 0, da).astype(np.uint8)
    ll = np.where(omask, 0, ll).astype(np.uint8)
    stem = os.path.splitext(name)[0] + f"_s{k}"
    d = os.path.join(out, split)
    cv2.imwrite(os.path.join(d, "images", stem + ".png"), img)
    cv2.imwrite(os.path.join(d, "da_masks", stem + ".png"), da)
    cv2.imwrite(os.path.join(d, "ll_masks", stem + ".png"), ll)
    if neg is not None:
        os.makedirs(os.path.join(d, "neg_sl"), exist_ok=True)
        cv2.imwrite(os.path.join(d, "neg_sl", stem + ".png"), neg.astype(np.uint8) * 255)
    sld = opt.get("sl_labels")   # 프레임별 깨끗한 흰 가로선 라벨(make_stopline_labels.py --out) -> 합성 물체가 덮은 곳만 지움
    if sld and os.path.exists(os.path.join(sld, split, "sl_masks", name)):
        slm = cv2.imread(os.path.join(sld, split, "sl_masks", name), 0) > 0
        slv = cv2.imread(os.path.join(sld, split, "sl_valid", name), 0) > 0
        slm &= ~(cv2.dilate(omask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        for sub, mk in (("sl_masks", slm), ("sl_valid", slv)):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
            cv2.imwrite(os.path.join(d, sub, stem + ".png"), mk.astype(np.uint8) * 255)
    H, W = img.shape[:2]
    with open(os.path.join(d, "labels", stem + ".txt"), "w") as fh:
        for c, x0, y0, x1, y1, vis in labs:
            fh.write(f"{ss.CLASSES.index(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    return split, stem, pool_of(name), name, int(omask.sum()), metas, [c for c, *_ in labs]


def init_worker(adir):
    globals()["A"] = ss.load_assets(adir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ego", default="dataset/ego_v4")
    ap.add_argument("--out", default="dataset/synth_signs_v2")
    ap.add_argument("--assets", default="dataset/synth_assets_v1")
    ap.add_argument("--train-copies", type=int, default=2)
    ap.add_argument("--empty", type=float, default=0.1, help="물체 없이 두는 비율")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near-bias", type=float, default=0.0, help="물체 거리를 0.4~2m에서 뽑는 비율 (나머지는 0.4~3m)")
    ap.add_argument("--class-weight", default="", help="클래스 비중, 예: sign_A=2,sign_B=2,parking=2")
    ap.add_argument("--monitor-p", type=float, default=0.0, help="신호등(불빛) 물체 중 뒤에 단색 방해 모니터를 세우는 비율")
    ap.add_argument("--cable-p", type=float, default=0.0, help="흰 케이블을 까는 이미지 비율 (neg_sl 마스크 생성)")
    ap.add_argument("--sl-labels", default="", help="프레임별 흰 가로선 라벨 폴더 (make_stopline_labels.py --out 결과)")
    a = ap.parse_args()
    os.chdir(ROOT)
    rng = np.random.default_rng(a.seed)
    cw = {t: 1.0 for t in ss.TARGETS}
    for kv in filter(None, a.class_weight.split(",")):
        k_, v_ = kv.split("="); cw[k_.strip()] = float(v_)
    opt = dict(near_bias=a.near_bias, monitor_p=a.monitor_p, cable_p=a.cable_p, sl_labels=a.sl_labels)
    splits = [s for s in sorted(os.listdir(a.ego)) if os.path.isdir(os.path.join(a.ego, s, "images"))]
    jobs = []
    for split in splits:
        for sub in ["images", "da_masks", "ll_masks", "labels"]:
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        names = sorted(os.listdir(os.path.join(a.ego, split, "images")))
        copies = a.train_copies if split == "train" else 1
        cnt = {t: 0 for t in ss.TARGETS}
        for k in range(copies):
            for n in names:
                if rng.random() < a.empty:
                    ts = []
                else:
                    order = sorted(ss.TARGETS, key=lambda t: cnt[t] / cw[t] + rng.random())
                    ts = order[:2 if rng.random() < 0.5 else 1]
                    if len(ts) == 2 and all(t in ss.LIGHTS for t in ts):
                        ts = [ts[0], [t for t in order if t in ss.SIGNS][0]]
                    for t in ts:
                        cnt[t] += 1
                jobs.append((split, n, k, ts, int(rng.integers(1 << 31)), a.ego, a.out, opt))
        print(f"{split}: {len(names)} frames x {copies}")
    print(f"planned {len(jobs)} images")
    rows = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.assets,)) as pool:
        for i, r in enumerate(pool.imap_unordered(make_one, jobs, chunksize=16)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    rows.sort()
    with open(os.path.join(a.out, "classes.txt"), "w") as fh:
        fh.write("\n".join(ss.CLASSES) + "\n")
    with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "pool", "ego_frame", "masked_px", "objects"])
        for split, stem, p, n, mpx, metas, _ in rows:
            w.writerow([split, stem, p, n, mpx, json.dumps(metas, ensure_ascii=False)])
    counts = {}
    for split, stem, p, n, mpx, metas, cl in rows:
        c = counts.setdefault(split, {"images": 0, "empty": 0, **{x: 0 for x in ss.CLASSES}})
        c["images"] += 1
        c["empty"] += not cl
        for x in cl:
            c[x] += 1
    json.dump(counts, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
    show = ["images", "empty"] + ss.CLASSES[:10]
    print("split        " + " ".join(f"{x[:7]:>7s}" for x in show))
    for s, c in counts.items():
        print(f"{s:12s} " + " ".join(f"{c[x]:7d}" for x in show))
    pc = {}
    for split, stem, p, *_ , cl in rows:
        if split == "train":
            pc[p] = pc.get(p, 0) + len([x for x in cl if x in ss.TARGETS])
    print("train boxes by pool:", pc)


if __name__ == "__main__":
    main()
