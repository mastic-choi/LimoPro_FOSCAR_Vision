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


def make_one(job):
    split, name, k, targets, seed, ego, out = job
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
            how = place(o, hsv, da, ll, rng)
            trial, tm = img.copy(), np.zeros_like(omask)
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
                              occ=round(o["occlude"], 2), occ_type="", vis=round(key[0][5], 2)))
            break
    for mi, o, box in pending:
        extra, metas[mi]["occ_type"] = ss.add_occluder(img, o, box, rng, A, omask)
        labs += extra
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
    a = ap.parse_args()
    os.chdir(ROOT)
    rng = np.random.default_rng(a.seed)
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
                    order = sorted(ss.TARGETS, key=lambda t: cnt[t] + rng.random())
                    ts = order[:2 if rng.random() < 0.5 else 1]
                    if len(ts) == 2 and all(t in ss.LIGHTS for t in ts):
                        ts = [ts[0], [t for t in order if t in ss.SIGNS][0]]
                    for t in ts:
                        cnt[t] += 1
                jobs.append((split, n, k, ts, int(rng.integers(1 << 31)), a.ego, a.out))
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
