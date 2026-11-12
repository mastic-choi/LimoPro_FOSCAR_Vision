#!/usr/bin/env python3
"""여러 다중 작업 가중치를 같은 test 사진에 나란히 그려 비교 몽타주를 만든다 (릴리즈 후보 고르기용).

열 = 모델, 행 = 같은 사진. split마다 모델끼리 결과가 가장 많이 다른 사진을 골라 그린다.
  det_<split>.jpg   검출 (점수 >= --thr). 맞은 예측 = 클래스 색, 틀린 예측(다른 클래스·오탐) = 빨강,
                    놓친 정답 = 주황 굵은 선, 정답 = 흰 얇은 선,
                    √넓이 14px 미만 정답(v3가 학습에서 무시한 크기, 불빛 제외) 중 놓친 것 = 회색
  lane_<split>.jpg  분할. 주행영역 예측 = 초록, 차선: 맞음 = 노랑 / 틀린 예측 = 빨강 / 놓침 = 파랑,
                    흰 가로선 예측 = 자홍
  summary.jpg, summary.json  split별 집계 (점수 >= thr의 맞음·틀림·놓침·오탐, da/ll/흰 가로선 IoU)
                    --monitor로 eval_mt_monitor.py 결과 json을 주면 모니터 앞 가짜 불빛도 같이 표에 넣음

사용 예:
  python scripts/eval_mt_compare_montage.py \
      --model "v2=outputs/train/hybridnets_mt_v3_seed0/best.pt" \
      --model "v3 ep11=outputs/train/hybridnets_mt_v3_seed0/best_score.pt" \
      --out outputs/train_mt_v3/compare
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import train_multitask as tm
import make_model_compare_gif as m

os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

TW, TH, BAR = 640, 360, 30
SMALL_PX = 14.0
PAL = [(60, 60, 255), (255, 160, 0), (255, 0, 200), (0, 200, 255), (0, 255, 120), (255, 255, 0),
       (200, 200, 200), (0, 0, 255), (0, 220, 255), (0, 255, 0), (255, 120, 120), (0, 140, 255)]
RED, ORANGE, GRAY, WHITE = (0, 0, 255), (0, 150, 255), (150, 150, 150), (255, 255, 255)
FONT_PATHS = ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
              "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def put_texts(img, items):
    """items: [(x, y, text, size, rgb)] 한글 가능. img는 BGR, 제자리 수정."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, t, s, c in items:
        d.text((x, y), t, font=font(s), fill=c)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def box_iou(a, b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    ix0, iy0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix1, iy1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def is_small(gt):
    s = np.sqrt(np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)) < SMALL_PX
    return s & ~np.isin(gt[:, 4], tm.LAMP_CLS)


def det_outcome(dets, gt):
    """dets: [(box, cls, score)] (점수 >= thr), gt (N,5).
    -> 예측 상태 ['tp'|'wrong'|'fa'], 정답 상태 ['ok'|'wrong'|'miss']. 점수 높은 순, 정답 하나에 예측 하나(IoU>=0.5)."""
    order = np.argsort([-d[2] for d in dets])
    used = np.zeros(len(gt), bool)
    pst = [None] * len(dets)
    for j in order:
        b, c, _ = dets[j]
        same = np.where((gt[:, 4] == c) & ~used)[0]
        if len(same):
            io = box_iou(b, gt[same, :4])
            k = int(np.argmax(io))
            if io[k] >= 0.5:
                used[same[k]] = True
                pst[j] = "tp"
                continue
        pst[j] = "wrong" if len(gt) and box_iou(b, gt[:, :4]).max() >= 0.5 else "fa"
    gst = []
    for i, g in enumerate(gt):
        if used[i]:
            gst.append("ok")
        elif any(s == "wrong" and box_iou(d[0], g[None, :4])[0] >= 0.5 for d, s in zip(dets, pst)):
            gst.append("wrong")
        else:
            gst.append("miss")
    return pst, gst


class Seg:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, p, t):
        self.tp += int((p & t).sum()); self.fp += int((p & ~t).sum()); self.fn += int((~p & t).sum())

    def iou(self):
        return self.tp / max(self.tp + self.fp + self.fn, 1) if (self.tp + self.fn) else None


@torch.no_grad()
def infer(model, ds, dev, thr, reg_t, clip_t):
    """사진마다 검출(점수>=thr)·분할 마스크(packbits)·지표."""
    res = []
    for i in range(len(ds)):
        x, da, ll, sl, slv, segv, slf, ann, nm = ds[i]
        reg, cls, anchors, pda, pll, psl = tm.heads(model, x[None].to(dev))
        pda, pll, psl = [(tm.ts.crop(p).float().cpu()[0] >= 0.5).numpy() for p in (pda, pll, psl)]
        # AP용은 학습 evaluate와 같게 점수 0.05 이상·이미지당 상위 100개, 그림·집계용은 thr 이상
        o = util.postprocess(x[None], anchors.float().cpu(), reg.float().cpu(), cls.float().cpu(), reg_t, clip_t, 0.05, 0.5)[0]
        all_dets = []
        for j in (np.argsort(-np.asarray(o["scores"]))[:100] if len(o["scores"]) else []):
            b = np.array(o["rois"][j], np.float32); b[[1, 3]] -= tm.PAD
            all_dets.append((b, int(o["class_ids"][j]), float(o["scores"][j])))
        dets = [d for d in all_dets if d[2] >= thr]
        gt = ann.numpy(); gt = gt[gt[:, 4] >= 0].copy(); gt[:, [1, 3]] -= tm.PAD
        pst, gst = det_outcome(dets, gt)
        small = is_small(gt) if len(gt) else np.zeros(0, bool)
        r = dict(name=nm, dets=dets, all_dets=all_dets, gt=gt, pst=pst, gst=gst, small=small,
                 n_gt=len(gt), ok=gst.count("ok"), wrong=gst.count("wrong"), miss=gst.count("miss"),
                 fa=sum(s != "tp" and s != "wrong" for s in pst),
                 miss_big=int(sum(g == "miss" and not s for g, s in zip(gst, small))),
                 n_big=int((~small).sum()), seg=bool(segv > 0))
        if segv > 0:
            tda, tll = da.numpy() >= 0.5, ll.numpy() >= 0.5
            r["da"], r["ll"] = Seg(), Seg()
            r["da"].add(pda, tda); r["ll"].add(pll, tll)
            r["m"] = {k: np.packbits(v) for k, v in (("pda", pda), ("pll", pll), ("psl", psl), ("tll", tll), ("tda", tda))}
            if slf > 0:
                v = slv.numpy() > 0
                r["sl"] = Seg(); r["sl"].add(psl[v], sl.numpy()[v] >= 0.5)
        res.append(r)
    return res


def unpack(r, k):
    return np.unpackbits(r["m"][k])[:TH * TW].reshape(TH, TW).astype(bool)


def label(img, x1, y1, text, bg, fg=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y1, th + 4)
    cv2.rectangle(img, (x1, y - th - 4), (x1 + tw + 4, y), bg, -1)
    cv2.putText(img, text, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def det_tile(img, r, name):
    img = img.copy()
    for g, st, sm in zip(r["gt"], r["gst"], r["small"]):   # 정답 먼저 (예측이 위에 그려지도록)
        x1, y1, x2, y2 = g[:4].round().astype(int)
        if st == "miss":
            cv2.rectangle(img, (x1, y1), (x2, y2), GRAY if sm else ORANGE, 1 if sm else 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 1)
    for (b, c, s), st in zip(r["dets"], r["pst"]):
        x1, y1, x2, y2 = b.round().astype(int)
        col = PAL[c] if st == "tp" else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label(img, x1, y1, f"{tm.CLASSES[c]} {s:.2f}", col, (0, 0, 0) if st == "tp" else WHITE)
    bar = np.full((BAR, TW, 3), 35, np.uint8)
    txt = f"{name}   맞음 {r['ok']}  틀림 {r['wrong']}  놓침 {r['miss']}  오탐 {r['fa']}"
    put_texts(bar, [(8, 4, txt, 18, (255, 255, 255))])
    return np.vstack([bar, img])


def lane_tile(img, r, name):
    img = img.copy().astype(np.float32)
    pda, pll, psl, tll = (unpack(r, k) for k in ("pda", "pll", "psl", "tll"))
    def tint(mask, bgr, a):
        img[mask] = img[mask] * (1 - a) + np.array(bgr, np.float32) * a
    tint(pda & ~pll, (0, 200, 0), 0.30)
    tint(psl, (255, 0, 255), 0.55)
    tint(pll & tll, (0, 230, 255), 0.85)    # 맞음: 노랑
    tint(pll & ~tll, (0, 0, 255), 0.85)     # 틀린 예측: 빨강
    tint(~pll & tll, (255, 120, 0), 0.85)   # 놓침: 파랑
    img = img.clip(0, 255).astype(np.uint8)
    cs, _ = cv2.findContours(unpack(r, "tda").astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, WHITE, 2)   # 정답 주행영역(내 차로) 테두리
    bar = np.full((BAR, TW, 3), 35, np.uint8)
    txt = f"{name}   da {r['da'].iou():.3f}   ll {r['ll'].iou() or 0:.3f}"
    if "sl" in r and r["sl"].iou() is not None:
        txt += f"   흰 가로선 {r['sl'].iou():.3f}"
    put_texts(bar, [(8, 4, txt, 18, (255, 255, 255))])
    return np.vstack([bar, img])


def header(width, title, lines):
    h = 46 + 26 * len(lines)
    bar = np.full((h, width, 3), 25, np.uint8)
    put_texts(bar, [(12, 8, title, 26, (255, 255, 255))] +
              [(12, 46 + 26 * i, t, 18, (210, 210, 210)) for i, t in enumerate(lines)])
    return bar


def col_titles(names):
    bar = np.full((40, TW * len(names), 3), 60, np.uint8)
    put_texts(bar, [(TW * i + 12, 7, n, 24, (255, 230, 120)) for i, n in enumerate(names)])
    return bar


def pick_rows(per_model, key, n):
    """모델끼리 가장 다른 사진 index. key='det' | 'lane'."""
    k = len(per_model[0])
    score = np.zeros(k)
    for i in range(k):
        rs = [pm[i] for pm in per_model]
        if key == "det":
            g = np.array([r["gst"] for r in rs]) if rs[0]["n_gt"] else np.zeros((len(rs), 0))
            score[i] = (g != g[:1]).any(0).sum() + 0.5 * (max(r["fa"] for r in rs) - min(r["fa"] for r in rs))
        else:
            if not rs[0]["seg"] or rs[0]["ll"].tp + rs[0]["ll"].fn < 800:
                continue
            lls = [r["ll"].iou() for r in rs]; das = [r["da"].iou() for r in rs]
            score[i] = (max(lls) - min(lls)) + 0.5 * (max(das) - min(das))
    idx = [i for i in np.argsort(-score, kind="stable") if score[i] > 0][:n]
    return idx, score


def summarize(res):
    s = dict(n_img=len(res), n_gt=0, ok=0, wrong=0, miss=0, fa=0, n_big=0, miss_big=0)
    da, ll, sl = Seg(), Seg(), Seg()
    for r in res:
        for k in ("n_gt", "ok", "wrong", "miss", "fa", "n_big", "miss_big"):
            s[k] += r[k]
        if r["seg"]:
            for a, b in ((da, r["da"]), (ll, r["ll"])):
                a.tp += b.tp; a.fp += b.fp; a.fn += b.fn
            if "sl" in r:
                sl.tp += r["sl"].tp; sl.fp += r["sl"].fp; sl.fn += r["sl"].fn
    s["da_iou"], s["ll_iou"], s["sl_iou"] = da.iou(), ll.iou(), sl.iou()
    # AP50: 학습 evaluate와 같은 계산(tm.ap50). big = √넓이 14px 미만 정답(불빛 제외)을 '무시'로 뺀 AP:
    # 그 정답과 같은 클래스로 IoU>=0.5 겹치고 큰 정답과는 안 겹치는 예측은 맞음도 오탐도 아님
    dets, gts, dets_b, gts_b = [], {}, [], {}
    for r in res:
        gt, sm = r["gt"], r["small"]
        gts[r["name"]], gts_b[r["name"]] = gt, gt[~sm]
        for b, c, sc in r["all_dets"]:
            dets.append((r["name"], c, sc, b))
            g_s, g_b = gt[sm & (gt[:, 4] == c)], gt[~sm & (gt[:, 4] == c)]
            hit_s = len(g_s) and box_iou(b, g_s[:, :4]).max() >= 0.5
            hit_b = len(g_b) and box_iou(b, g_b[:, :4]).max() >= 0.5
            if not (hit_s and not hit_b):
                dets_b.append((r["name"], c, sc, b))
    ap, ap_b = tm.ap50(dets, gts, tm.NC), tm.ap50(dets_b, gts_b, tm.NC)
    s["ap50"] = {tm.CLASSES[c]: v for c, v in ap.items()}
    s["ap50_14px"] = {tm.CLASSES[c]: v for c, v in ap_b.items()}
    no_cone = [v for k, v in s["ap50"].items() if v is not None and k != "cone"]
    s["map50_no_cone"] = float(np.mean(no_cone)) if no_cone else None
    return s


def table_img(rows, cols, title, note=""):
    cw = [max(150, 12 * max(len(str(c)) for c in [h] + [r[j] for r in rows]) + 20) for j, h in enumerate(cols)]
    cw[0] = max(cw[0], 260)
    W, H = sum(cw) + 24, 60 + 34 * (len(rows) + 1) + (40 if note else 10)
    img = np.full((H, W, 3), 255, np.uint8)
    items, y = [(12, 10, title, 26, (20, 20, 20))], 56
    x = 12
    for j, h in enumerate(cols):
        items.append((x, y, h, 18, (90, 90, 90))); x += cw[j]
    for i, r in enumerate(rows):
        y = 56 + 34 * (i + 1)
        if r[0] == "":
            continue
        if i % 2 == 0:
            cv2.rectangle(img, (6, y - 6), (W - 6, y + 26), (245, 245, 245), -1)
        x = 12
        for j, v in enumerate(r):
            items.append((x, y, str(v), 18, (20, 20, 20))); x += cw[j]
    if note:
        items.append((12, H - 34, note, 16, (110, 110, 110)))
    put_texts(img, items)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help='"이름=가중치.pt" (여러 번)')
    ap.add_argument("--summary-only", action="store_true", help="추론 없이 <out>/summary.json으로 요약 표만 다시 그림")
    ap.add_argument("--monitor", action="append", default=[], help='"이름=eval_mt_monitor json" (여러 번, 선택)')
    ap.add_argument("--out", default="outputs/train_mt_v3/compare")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else str(m.device))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summary_only:
        d = json.load(open(os.path.join(a.out, "summary.json"), encoding="utf-8"))
        write_summary(a, list(d["models"].items()), d["summary"])
        return
    models = [s.split("=", 1) for s in a.model]
    names = [n for n, _ in models]
    dev = torch.device(a.device)
    reg_t, clip_t = util.BBoxTransform(), util.ClipBoxes()

    seg, cone = os.path.join(ROOT, "dataset/synth_signs_v2"), os.path.join(ROOT, "dataset/cones_v1")
    # 콘: 학습 스크립트 test 채점과 같은 묶음 (콘 test + 콘 val 중 학습 검증에 안 쓴 나머지, seed 0, cone_val_n 300)
    splits = [("test", tm.seg_items(seg, "test")), ("test_venue", tm.seg_items(seg, "test_venue")),
              ("test_bag", tm.seg_items(seg, "test_bag")),
              ("cone", tm.det_items(cone, "test", 0, 0) + tm.det_items(cone, "val", 0, 0)[300:])]
    dsets = {s: tm.Items(items, train=False) for s, items in splits}
    results = {s: [] for s in dsets}
    for n, w in models:
        model = tm.build_model(m, w).to(dev).eval()
        for s, ds in dsets.items():
            results[s].append(infer(model, ds, dev, a.thr, reg_t, clip_t))
            print(f"[{n}] {s}: {len(ds)} imgs", flush=True)
        del model

    summary = {s: {n: summarize(results[s][j]) for j, n in enumerate(names)} for s in dsets}
    legend_det = ["색 박스 = 맞은 예측(클래스 색) · 빨강 = 틀린 예측(다른 클래스·오탐) · 흰 얇은 선 = 정답",
                  f"주황 굵은 선 = 놓친 정답 · 회색 = 놓친 정답 중 √넓이 {SMALL_PX:.0f}px 미만(v3가 학습에서 무시한 크기) · 점수 >= {a.thr}"]
    legend_lane = ["초록 = 주행영역(내 차로) 예측 · 흰 테두리 = 정답 주행영역 · 차선: 노랑 = 맞음, 빨강 = 틀린 예측, 파랑 = 놓침 · 자홍 = 흰 가로선 예측"]
    for s, ds in dsets.items():
        per_model = results[s]
        kinds = ["det"] + (["lane"] if per_model[0] and per_model[0][0]["seg"] else [])
        for kind in kinds:
            idx, _ = pick_rows(per_model, kind, a.rows)
            if not idx:
                continue
            rows = []
            for i in idx:
                img = ds.load(ds.items[i])[0]
                tiles = [(det_tile if kind == "det" else lane_tile)(img, pm[i], n) for pm, n in zip(per_model, names)]
                row = np.hstack(tiles)
                cv2.putText(row, per_model[0][i]["name"].split("/")[-1][:40], (6, BAR + TH - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                rows.append(row)
                rows.append(np.full((6, row.shape[1], 3), 0, np.uint8))
            sm = summary[s]
            if kind == "det":
                lines = [f"{n}: 정답 {v['n_gt']}개 중 맞음 {v['ok']} · 틀림 {v['wrong']} · 놓침 {v['miss']}"
                         f" (14px 이상만 {v['miss_big']}/{v['n_big']}) · 오탐 {v['fa']}" for n, v in sm.items()] + legend_det
                title = f"{s} 검출 비교 — 모델끼리 결과가 다른 사진 {len(idx)}장 ({len(ds)}장 중)"
            else:
                lines = [f"{n}: 전체 da {v['da_iou']:.4f} · ll {v['ll_iou']:.4f}"
                         + (f" · 흰 가로선 {v['sl_iou']:.4f}" if v["sl_iou"] is not None else "") for n, v in sm.items()] + legend_lane
                title = f"{s} 차선 비교 — 차선 IoU가 모델끼리 가장 다른 사진 {len(idx)}장 ({len(ds)}장 중)"
            W = TW * len(names)
            out = np.vstack([header(W, title, lines), col_titles(names)] + rows)
            p = os.path.join(a.out, f"{kind}_{s}.jpg")
            cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"-> {p} ({len(idx)} rows)", flush=True)

    write_summary(a, models, summary)


def write_summary(a, models, summary):
    names = [n for n, _ in models]
    # 요약 표
    cols = ["split / 모델", "정답", "맞음", "틀림", "놓침", "놓침(14px+)", "오탐", "mAP50(콘 제외)", "콘 AP50", "콘 AP50(14px+)",
            "da IoU", "ll IoU", "흰 가로선"]
    rows = []
    f = lambda x, d=4: "-" if x is None else f"{x:.{d}f}"
    for s in summary:
        for n in names:
            v = summary[s][n]
            rows.append([f"{s} · {n}", v["n_gt"], v["ok"], v["wrong"], v["miss"], f"{v['miss_big']}/{v['n_big']}", v["fa"],
                         f(v["map50_no_cone"], 3), f(v["ap50"].get("cone"), 3), f(v["ap50_14px"].get("cone"), 3),
                         f(v["da_iou"]), f(v["ll_iou"]), f(v["sl_iou"])])
        rows.append([""] * len(cols))
    imgs = [table_img(rows[:-1], cols, f"검출 (점수 >= {a.thr}) · 분할 요약",
                      "맞음/틀림/놓침은 정답 박스 기준, 오탐은 어떤 정답과도 IoU 0.5 미만인 예측. "
                      "14px+ = √넓이 14px 미만 정답(불빛 제외)을 빼고 셈. cone = 콘 test + 콘 val 나머지 (학습 test 채점과 같은 묶음)")]
    mon = {}
    for spec in a.monitor:
        n, p = spec.split("=", 1)
        if os.path.exists(p):
            mon[n] = json.load(open(p))
    if mon:
        mcols = ["모니터 / 모델", "불빛 색 맞음", "색 틀림", "놓침", "가짜 불빛(모니터)", "가짜 불빛 사진", "가짜 신호등(모니터)"]
        mrows = []
        for var in ("dots", "solid"):
            for thr in ("thr0.3", "thr0.5"):
                for n, d in mon.items():
                    v = d[var][thr]
                    mrows.append([f"{var} {thr[3:]} · {n}", f"{v['lamp_correct'] * 100:.1f}%", f"{v['lamp_wrong_color'] * 100:.1f}%",
                                  f"{v['lamp_missed'] * 100:.1f}%", v["fake_lamps_on_monitor"], v["images_with_fake_lamp_on_monitor"],
                                  v["fake_tl_on_monitor"]])
                mrows.append([""] * len(mcols))
        imgs.append(table_img(mrows[:-1], mcols, "신호등 뒤 방해 모니터 (193장, 정답 불빛 203개)",
                              "solid = 단색 화면, dots = 땡땡이 화면. 0.3/0.5 = 점수 기준"))
    W = max(i.shape[1] for i in imgs)
    out = np.vstack([np.pad(i, ((0, 10), (0, W - i.shape[1]), (0, 0)), constant_values=255) for i in imgs])
    cv2.imwrite(os.path.join(a.out, "summary.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(dict(thr=a.thr, models=dict(models), summary=summary,
                   monitor={n: {var: d[var] for var in ("clean", "solid", "dots")} for n, d in mon.items()}),
              open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(f"-> {os.path.join(a.out, 'summary.jpg')}", flush=True)



if __name__ == "__main__":
    main()
