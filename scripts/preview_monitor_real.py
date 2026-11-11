#!/usr/bin/env python3
"""모니터 증강의 '실제처럼 보이게' 효과를 하나씩 켜서 비교하는 그림을 만든다 (mt_online_aug.add_monitor real=...).

같은 사진·같은 seed라 모니터 위치·크기·색은 칸마다 같고 효과만 다르다.
칸: 기존 | 1 match(선명도·노이즈) | 2 persp(원근) | 3 screen(비네팅·반사·날아감) | 4 jitter(땡땡이 흔들기) | 전부
위 절반은 단색, 아래 절반은 땡땡이. 각 칸은 모니터 주변을 2배(선형 보간)로 확대.
출력: <out>/real_crops.jpg (확대 비교), <out>/real_full.jpg (기존 vs 전부, 원본 크기)

사용 예:
  .venv/Scripts/python scripts/preview_monitor_real.py %USERPROFILE%/Downloads/LimoPro_dataset_mt_v1.zip
"""
import argparse, csv, io, os, sys, zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_online_aug as oa

VARIANTS = [("기존", ()), ("1 match", ("match",)), ("2 persp", ("persp",)), ("3 screen", ("screen",)),
            ("4 jitter", ("jitter",)), ("전부", oa.REAL_ALL)]
LABELS = ["base", "1 match", "2 persp", "3 screen", "4 jitter", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out", default="outputs/synth_preview/online_aug")
    ap.add_argument("--n", type=int, default=5, help="단색·땡땡이 각각 몇 장")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    P = next(n for n in z.namelist() if n.endswith("synth_signs_v2/train/labels/") or "/synth_signs_v2/train/labels/" in n)
    P = P[:P.index("synth_signs_v2/")] + "synth_signs_v2/train/"
    kept = []
    for n in [n for n in z.namelist() if n.startswith(P + "klein_refine") and n.endswith(".csv")]:
        kept += [r["image"] for r in csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")) if r["status"] == "kept"]
    rng0 = np.random.default_rng(a.seed); rng0.shuffle(kept)
    rd = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
    rows, full = [], []
    want = [False] * a.n + [True] * a.n
    ki = 0
    for dots in want:
        while ki < len(kept):
            k = kept[ki]; ki += 1
            bx = []
            for l in z.read(P + "labels/" + k + ".txt").decode().split("\n"):
                v = l.split()
                if len(v) == 5:
                    c, x, y, w, h = int(v[0]), *map(float, v[1:])
                    bx.append([(x - w / 2) * 640, (y - h / 2) * 360, (x + w / 2) * 640, (y + h / 2) * 360, c])
            b = np.array(bx, np.float32).reshape(-1, 5)
            if not ((b[:, 4] == oa.TL_CLS) & (b[:, 2] - b[:, 0] >= 10)).any():   # 확대해서 볼 만한 신호등
                continue
            base = rd(P + "images/" + k + ".png")[..., :3]
            m = lambda s: (rd(P + s + "/" + k + ".png") > 0).astype(np.uint8)
            da0, ll0 = m("da_masks"), m("ll_masks")
            seed = int(rng0.integers(1 << 30))
            outs = []
            for _, real in VARIANTS:
                img = base.copy(); z0 = np.zeros_like(da0)
                ok = oa.add_monitor(img, da0.copy(), ll0.copy(), z0.copy(), z0.copy(), b, np.random.default_rng(seed),
                                    real=real, dots=dots)
                if not ok:
                    break
                outs.append(img)
            if len(outs) != len(VARIANTS):
                continue
            diff = np.zeros(base.shape[:2], bool)
            for o in outs:
                diff |= np.abs(o.astype(np.int16) - base).max(2) > 12
            ys, xs = np.nonzero(diff)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
            hw = max(40, (x1 - x0) * 0.75); hh = hw * 0.75
            X0, X1 = int(max(0, cxm - hw)), int(min(640, cxm + hw)); Y0, Y1 = int(max(0, cym - hh)), int(min(360, cym + hh))
            tiles = []
            for (name, _), o in zip(VARIANTS, outs):
                t = cv2.resize(o[Y0:Y1, X0:X1], (240, 180), interpolation=cv2.INTER_LINEAR)
                tiles.append(t)
            rows.append(np.hstack(tiles))
            full.append(np.hstack([outs[0], outs[-1]]))
            break
    head = np.full((28, 240 * len(VARIANTS), 3), 255, np.uint8)
    for i, t in enumerate(LABELS):
        cv2.putText(head, t, (i * 240 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(os.path.join(a.out, "real_crops.jpg"), np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(a.out, "real_full.jpg"), np.vstack(full[::2]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("rows", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
