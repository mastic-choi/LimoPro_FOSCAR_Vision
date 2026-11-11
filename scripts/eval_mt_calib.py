#!/usr/bin/env python3
"""다중 작업 HybridNets의 점수 보정(calibration) 측정: ECE, MCE, NLL, Brier + temperature scaling.

검출 (D-ECE 방식, Küppers et al. 2020)
  NMS 뒤 예측 박스(점수 >= --min-score)마다 '같은 클래스 정답과 IoU >= 0.5로 맞았나'(1/0)를 정한다.
  한 정답에는 점수 높은 예측 하나만 맞음, 나머지는 틀림(중복).
  ECE = Σ_b (n_b/N)|정답률_b - 평균점수_b|, MCE = max_b |정답률_b - 평균점수_b| (예측 --mce-min개 이상인 구간만),
  NLL = 맞음/틀림에 대한 이진 교차 엔트로피, Brier = (점수 - 맞음)^2 평균. 구간은 --bins개 등간격.
  주의: 놓친 정답(예측이 없는 것)은 점수가 없어 calibration에 들어가지 않는다(재현율은 AP로 본다).
분할: da / ll / 흰 가로선 픽셀 확률 vs 정답 마스크의 픽셀 ECE·NLL (가로선은 라벨을 믿을 픽셀만)
보정: val 예측으로 temperature T를 맞춘다(전체 1개, 클래스별). p' = sigmoid(logit(p) / T). NLL 최소.
      T > 1 = 점수가 과신(너무 높음), T < 1 = 과소(너무 낮음).
출력 <out>/calib_report.md, calib.json(T 값 포함), reliability_<split>.png
사용: python scripts/eval_mt_calib.py <best.pt> [--out outputs/train_mt_v2/calib]
"""
import argparse, json, os, sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_multitask as tm
import make_model_compare_gif as m

C = tm.CLASSES


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def metrics(p, y, bins=15, mce_min=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return dict(n=0)
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    ece, mce, rel = 0.0, 0.0, []
    for b in range(bins):
        s = idx == b
        if not s.any():
            continue
        gap = abs(y[s].mean() - p[s].mean())
        ece += s.mean() * gap
        if s.sum() >= mce_min:
            mce = max(mce, gap)
        rel.append((float(p[s].mean()), float(y[s].mean()), int(s.sum())))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    nll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return dict(n=int(len(p)), acc=round(float(y.mean()), 4), conf=round(float(p.mean()), 4), ece=round(float(ece), 4),
                mce=round(float(mce), 4), nll=round(nll, 4), brier=round(float(((p - y) ** 2).mean()), 4), rel=rel)


def fit_T(p, y):
    """NLL 최소 temperature (격자 + 국소 탐색)."""
    z, y = logit(np.asarray(p)), np.asarray(y, float)
    if len(z) < 20:
        return 1.0
    def nll(T):
        q = np.clip(sig(z / T), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    grid = np.exp(np.linspace(np.log(0.2), np.log(5), 200))
    T = grid[np.argmin([nll(t) for t in grid])]
    for _ in range(30):   # 황금분할 비슷한 국소 탐색
        lo, hi = T / 1.05, T * 1.05
        T = min((lo, T, hi), key=nll)
    return float(T)


def match(dets, gts):
    """dets: [(score, cls, box)], gts: (N,5) -> 예측마다 1/0"""
    out = np.zeros(len(dets))
    used = np.zeros(len(gts), bool)
    order = np.argsort([-d[0] for d in dets])
    for j in order:
        s, c, b = dets[j]
        g = np.where((gts[:, 4] == c) & ~used)[0]
        if not len(g):
            continue
        gg = gts[g, :4]
        x1, y1 = np.maximum(gg[:, 0], b[0]), np.maximum(gg[:, 1], b[1])
        x2, y2 = np.minimum(gg[:, 2], b[2]), np.minimum(gg[:, 3], b[3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / ((gg[:, 2] - gg[:, 0]) * (gg[:, 3] - gg[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        k = int(np.argmax(iou))
        if iou[k] >= 0.5:
            used[g[k]] = True
            out[j] = 1
    return out


def collect(model, util, split, a, amp):
    items = tm.seg_items(a.data, split)
    if split == "val" and a.with_cones:
        items += tm.det_items(a.cones, "val", 300, 0)
    dl = torch.utils.data.DataLoader(tm.Items(items, False), batch_size=a.batch, num_workers=0, collate_fn=tm.collate)
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()
    D = dict(score=[], cls=[], size=[], y=[])
    E = 20
    hist = {k: np.zeros((E, 3)) for k in ("da", "ll", "sl")}   # 구간별 [개수, 확률합, 정답합]
    nll_px = {k: [0.0, 0] for k in hist}
    with torch.no_grad():
        for x, da, ll, sl, slv, segv, slf, ann, names in dl:
            x = x.to(m.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                reg, cls, anchors, pda, pll, psl = tm.heads(model, x)
            out = util.postprocess(x, anchors.float(), reg.float(), cls.float(), reg_t, clip_t, a.min_score, 0.5)
            pda, pll, psl = (tm.ts.crop(p).float().cpu().numpy() for p in (pda, pll, psl))
            for k in range(len(names)):
                g = ann[k].numpy(); g = g[g[:, 4] >= 0]
                o = out[k]
                dets = [(float(s), int(c), np.asarray(b, np.float32)) for b, c, s in zip(o["rois"], o["class_ids"], o["scores"])]
                dets = sorted(dets, key=lambda d: -d[0])[:100]
                ys = match(dets, g)
                for (s, c, b), yv in zip(dets, ys):
                    D["score"].append(s); D["cls"].append(c); D["y"].append(yv)
                    D["size"].append(float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))))
                if segv[k] > 0:
                    for key, pm, tg, vm in (("da", pda[k], da[k].numpy(), None), ("ll", pll[k], ll[k].numpy(), None),
                                            ("sl", psl[k], sl[k].numpy(), slv[k].numpy() if slf[k] > 0 else None)):
                        if key == "sl" and vm is None:
                            continue
                        msk = np.ones_like(tg, bool) if vm is None else vm > 0
                        pp, tt = pm[msk].ravel(), tg[msk].ravel()
                        bi = np.clip((pp * E).astype(int), 0, E - 1)
                        hist[key][:, 0] += np.bincount(bi, minlength=E)
                        hist[key][:, 1] += np.bincount(bi, weights=pp, minlength=E)
                        hist[key][:, 2] += np.bincount(bi, weights=tt, minlength=E)
                        pc = np.clip(pp, 1e-6, 1 - 1e-6)
                        nll_px[key][0] += float(-(tt * np.log(pc) + (1 - tt) * np.log(1 - pc)).sum()); nll_px[key][1] += len(pp)
    D = {k: np.array(v) for k, v in D.items()}
    seg = {}
    for key, h in hist.items():
        n = h[:, 0].sum()
        if n == 0:
            continue
        nz = h[:, 0] > 0
        gap = np.abs(h[nz, 2] / h[nz, 0] - h[nz, 1] / h[nz, 0])
        seg[key] = dict(pixels=int(n), ece=round(float((h[nz, 0] / n * gap).sum()), 4),
                        mce=round(float(gap[h[nz, 0] >= 1000].max()) if (h[nz, 0] >= 1000).any() else 0.0, 4),
                        nll=round(nll_px[key][0] / max(nll_px[key][1], 1), 4))
    return D, seg


def reliability_png(rows, path, title):
    """rows: [(이름, rel 목록, 색)] -> 신뢰도 그래프"""
    W, H, P = 420, 420, 40
    img = np.full((H + 30, W + 10, 3), 255, np.uint8)
    cv2.rectangle(img, (P, 10), (W - 10, H - P), (0, 0, 0), 1)
    to = lambda x, y: (int(P + x * (W - 10 - P)), int(H - P - y * (H - P - 10)))
    cv2.line(img, to(0, 0), to(1, 1), (180, 180, 180), 1)
    for t in (0.25, 0.5, 0.75):
        cv2.putText(img, f"{t}", (to(t, 0)[0] - 10, H - P + 15), 0, 0.35, (0, 0, 0), 1)
        cv2.putText(img, f"{t}", (5, to(0, t)[1] + 4), 0, 0.35, (0, 0, 0), 1)
    for li, (name, rel, col) in enumerate(rows):
        pts = [to(c, acc) for c, acc, n in rel if n >= 5]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(img, p1, p2, col, 2)
        for p in pts:
            cv2.circle(img, p, 3, col, -1)
        cv2.putText(img, name, (P + 8, 28 + 16 * li), 0, 0.42, col, 1)
    cv2.putText(img, title, (P, H - 8), 0, 0.42, (0, 0, 0), 1)
    cv2.putText(img, "x: score   y: actual precision", (P, H + 18), 0, 0.4, (80, 80, 80), 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="dataset/synth_signs_v2")
    ap.add_argument("--cones", default="dataset/cones_v1")
    ap.add_argument("--with-cones", action="store_true", help="val에 콘 사진도 넣기 (cones_v1이 있을 때)")
    ap.add_argument("--splits", nargs="*", default=["test", "test_venue", "test_bag"])
    ap.add_argument("--out", default="outputs/train_mt_v2/calib")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--min-size", type=float, default=20, help="'판단에 쓰는 크기' 부분 집계 기준(px)")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    os.chdir(m.ROOT)
    sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
    import utils.utils as util
    os.makedirs(a.out, exist_ok=True)
    model = tm.build_model(m, a.ckpt).to(m.device).eval()
    amp = m.device.type == "cuda" and "ZLUDA" not in torch.cuda.get_device_name(0)

    # 1) val에서 temperature 맞추기
    V, vseg = collect(model, util, "val", a, amp)
    T_all = fit_T(V["score"], V["y"])
    T_cls = {}
    for c in range(len(C)):
        s = V["cls"] == c
        if s.sum() >= 30:
            T_cls[C[c]] = round(fit_T(V["score"][s], V["y"][s]), 3)
    print(f"val: {len(V['score'])} dets, T_all {T_all:.3f}, T_cls {T_cls}", flush=True)

    def apply(D, mode):
        if mode == "raw":
            return D["score"]
        if mode == "T":
            return sig(logit(D["score"]) / T_all)
        return np.array([sig(logit(s) / T_cls.get(C[c], T_all)) for s, c in zip(D["score"], D["cls"])])

    res = {"T_all": round(T_all, 3), "T_cls": T_cls, "val": {"det_raw": {k: v for k, v in metrics(V["score"], V["y"], a.bins).items() if k != "rel"}, "seg": vseg}}
    md = ["# 점수 보정(calibration) 측정", "",
          f"가중치 `{a.ckpt}`. 검출: NMS 뒤 점수 >= {a.min_score} 예측마다 같은 클래스 정답과 IoU >= 0.5로 맞았는지(1/0)와 점수를 비교 "
          f"({a.bins}개 구간). 놓친 정답은 점수가 없어 여기 들어가지 않습니다.", "",
          f"temperature(val에서 맞춤): 전체 T = **{T_all:.3f}**, 클래스별 {json.dumps(T_cls, ensure_ascii=False)}  ",
          "(T > 1이면 점수가 실제보다 높게 나오는 과신, T < 1이면 실제보다 낮게 나오는 과소)", ""]
    for split in a.splits:
        D, seg = collect(model, util, split, a, amp)
        big = D["size"] >= a.min_size
        r = {}
        for mode in ("raw", "T", "Tcls"):
            p = apply(D, mode)
            r[mode] = metrics(p, D["y"], a.bins)
            r[mode + f"_ge{int(a.min_size)}px"] = metrics(p[big], D["y"][big], a.bins)
        per = {}
        for c in range(len(C)):
            s = D["cls"] == c
            if s.sum() >= 10:
                pr, pt = metrics(D["score"][s], D["y"][s], a.bins), metrics(apply(D, "Tcls")[s], D["y"][s], a.bins)
                per[C[c]] = {"raw": {k: v for k, v in pr.items() if k != "rel"}, "Tcls": {k: v for k, v in pt.items() if k != "rel"}}
        res[split] = {"det": {k: {kk: vv for kk, vv in v.items() if kk != "rel"} for k, v in r.items()}, "per_class": per, "seg": seg}
        reliability_png([("raw", r["raw"]["rel"], (0, 0, 220)), ("T (global)", r["T"]["rel"], (0, 150, 0)),
                         ("T (per class)", r["Tcls"]["rel"], (200, 100, 0))],
                        os.path.join(a.out, f"reliability_{split}.png"), f"{split}: detection reliability")
        md += [f"## {split}", "", "| 검출 점수 | 예측 수 | 실제 정답률 | 평균 점수 | ECE | MCE | NLL | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, lab in (("raw", "원래 점수"), ("T", "보정(전체 T)"), ("Tcls", "보정(클래스별 T)"),
                       (f"raw_ge{int(a.min_size)}px", f"원래, {int(a.min_size)}px 이상"), (f"Tcls_ge{int(a.min_size)}px", f"클래스별 T, {int(a.min_size)}px 이상")):
            v = r[k]
            if v.get("n"):
                md.append(f"| {lab} | {v['n']} | {v['acc']:.3f} | {v['conf']:.3f} | {v['ece']:.3f} | {v['mce']:.3f} | {v['nll']:.3f} | {v['brier']:.3f} |")
        md += ["", "| 클래스 | 예측 수 | 정답률 | 평균 점수 | ECE 원래 → 보정 | NLL 원래 → 보정 |", "|---|---:|---:|---:|---|---|"]
        for c, v in per.items():
            md.append(f"| {c} | {v['raw']['n']} | {v['raw']['acc']:.2f} | {v['raw']['conf']:.2f} | {v['raw']['ece']:.3f} → {v['Tcls']['ece']:.3f} | {v['raw']['nll']:.3f} → {v['Tcls']['nll']:.3f} |")
        if seg:
            md += ["", "| 분할 (픽셀) | 픽셀 수 | ECE | MCE | NLL |", "|---|---:|---:|---:|---:|"]
            for k, v in seg.items():
                md.append(f"| {k} | {v['pixels']:,} | {v['ece']:.4f} | {v['mce']:.4f} | {v['nll']:.4f} |")
        md += ["", f"![reliability {split}](reliability_{split}.png)", ""]
        print(split, json.dumps(res[split]["det"]["raw"]), flush=True)
    json.dump(res, open(os.path.join(a.out, "calib.json"), "w"), indent=1)
    open(os.path.join(a.out, "calib_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
