#!/usr/bin/env python3
"""모니터 시험 패턴(png)을 신호등 뒤 모니터에 띄운 사진으로 여러 모델을 나란히 비교하는 몽타주.

켜진 불 색(빨강·노랑·초록)마다 대회장 test 사진 하나를 고르고, 패턴 이름의 for-<색>/same-<색>에 맞는 사진에 띄운다
(단색 패턴은 세 사진에 돌아가며). 모니터 위치는 사진마다 고정이라 패턴만 바뀐다. 그림은 모니터 주변을 확대한 것.
  monitor_patterns_<색>.jpg, monitor_patterns_solid.jpg, monitor_patterns.json
표시: 자홍 굵은 선 = 모니터 위 가짜 불빛·가짜 신호등, 색 박스 = 그 밖의 예측, 흰 얇은 선 = 정답

사용 예:
  python scripts/eval_mt_monitor_patterns.py --patterns <패턴 폴더> \
      --model "v2=outputs/train/hybridnets_mt_v3_seed0/best.pt" --model "v3 ep11=..." --out outputs/train_mt_v3/monitor_patterns
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2, numpy as np, torch
import train_multitask as tm
import mt_online_aug as oa
import eval_mt_compare_montage as cm   # 글꼴·박스 그리기 재사용 (HybridNets utils도 여기서 불러옴)

TL = tm.CLASSES.index("traffic_light")
LAMPS = {"red": tm.CLASSES.index("light_red"), "yellow": tm.CLASSES.index("light_yellow"),
         "green": tm.CLASSES.index("light_green")}
TW, TH, BAR, LW = 640, 360, 30, 240
FAKE = (255, 0, 255)   # 자홍: 모니터 위 가짜 (빨간불 클래스 색과 안 겹치게)


def photo_candidates(data, splits, seed, near=False):
    """켜진 불 색마다 모니터가 그려지는 사진 후보. 앞 split(대회장) 먼저, 그 안에서 신호등 폭이 중간에 가까운 순."""
    ds = tm.Items([], False)
    cand = {c: [] for c in LAMPS}
    for split in splits:
        for it in tm.seg_items(data, split):
            img, da, ll, sl, slv, b = ds.load(it)
            t = b[b[:, 4] == TL]
            if len(t) != 1 or t[0, 2] - t[0, 0] < 12:
                continue
            lit = [c for c, k in LAMPS.items() if (b[:, 4] == k).any()]
            if len(lit) != 1:
                continue
            im = img.copy(); z = np.zeros(img.shape[:2], np.uint8)
            if oa.add_monitor(im, da.copy(), ll.copy(), z.copy(), z.copy(), b, np.random.default_rng(seed),
                              real=("match", "persp", "screen"), dots=False):
                cand[lit[0]].append((splits.index(split), float(t[0, 2] - t[0, 0]), it, img, da, ll, b))
    for c, v in cand.items():
        med = float(np.median([x[1] for x in v]))
        v.sort(key=lambda x: (x[0], -x[1] if near else abs(x[1] - med)))   # near: 신호등이 가장 큰(가까운) 사진부터
    return cand


def iou(a, b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return cm.box_iou(a, b) if len(b) else np.zeros(0)


def judge(dets, b, mon, thr):
    """eval_mt_monitor.py와 같은 기준. dets [(box, cls, score)] -> 예측별 표시, 집계."""
    d = [x for x in dets if x[2] >= thr]
    g_l, g_t = b[np.isin(b[:, 4], list(LAMPS.values()))], b[b[:, 4] == TL]
    on_mon = lambda x: mon[int(np.clip((x[0][1] + x[0][3]) / 2, 0, 359)), int(np.clip((x[0][0] + x[0][2]) / 2, 0, 639))]
    marks, r = [], dict(fake_lamp=0, fake_tl=0, lamp="miss")
    for x in d:
        fake = False
        if x[1] in LAMPS.values():
            fake = not (len(g_l) and iou(x[0], g_l[:, :4]).max() >= 0.3) and on_mon(x)
            r["fake_lamp"] += fake
        elif x[1] == TL:
            fake = not (len(g_t) and iou(x[0], g_t[:, :4]).max() >= 0.3) and on_mon(x)
            r["fake_tl"] += fake
        marks.append((x, fake))
    if len(g_l):
        cand = [x for x in d if x[1] in LAMPS.values() and iou(g_l[0, :4], np.array([x[0]]))[0] >= 0.5]
        if cand:
            r["lamp"] = "ok" if max(cand, key=lambda x: x[2])[1] == int(g_l[0, 4]) else "wrong"
    r["fake_lamp"], r["fake_tl"] = int(r["fake_lamp"]), int(r["fake_tl"])
    return marks, r


@torch.no_grad()
def detect(model, im, dev, reg_t, clip_t):
    x = cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = ((cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1)
    x = torch.from_numpy(x)[None].to(dev)
    reg, cls, anchors, *_ = tm.heads(model, x)
    o = cm.util.postprocess(x.cpu(), anchors.float().cpu(), reg.float().cpu(), cls.float().cpu(), reg_t, clip_t, 0.05, 0.5)[0]
    dets = []
    for bb, cc, sc in zip(o["rois"], o["class_ids"], o["scores"]):
        bb = np.array(bb, np.float32); bb[[1, 3]] -= tm.PAD
        dets.append((bb, int(cc), float(sc)))
    return dets


def crop_box(mon, b):
    ys, xs = np.where(mon)
    t = b[b[:, 4] == TL][0]
    x0, y0 = min(xs.min(), t[0]), min(ys.min(), t[1])
    x1, y1 = max(xs.max(), t[2]), max(ys.max(), t[3])
    w, h = (x1 - x0) * 1.5, (y1 - y0) * 1.5
    w = min(max(w, h * 16 / 9, 160), 640); h = w * 9 / 16   # 16:9, 사진 안으로
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = np.clip(cx - w / 2, 0, 640 - w), np.clip(cy - h / 2, 0, 360 - h)
    return float(x0), float(y0), float(w), float(h)


def tile(img, marks, b, cb, head):
    x0, y0, w, h = cb
    s = TW / w
    im = cv2.resize(img[int(y0):int(y0 + h), int(x0):int(x0 + w)], (TW, TH), interpolation=cv2.INTER_CUBIC)
    T = lambda bb: (int((bb[0] - x0) * s), int((bb[1] - y0) * s), int((bb[2] - x0) * s), int((bb[3] - y0) * s))
    for g in b:
        if int(g[4]) == TL or int(g[4]) in LAMPS.values():
            p = T(g); cv2.rectangle(im, p[:2], p[2:], cm.WHITE, 1)
    for (bb, c, sc), fake in sorted(marks, key=lambda m: m[1]):
        p = T(bb)
        col = FAKE if fake else cm.PAL[c]
        cv2.rectangle(im, p[:2], p[2:], col, 3 if fake else 2)
        cm.label(im, p[0], p[1], f"{tm.CLASSES[c]} {sc:.2f}", col, cm.WHITE if fake else (0, 0, 0))
    bar = np.full((BAR, TW, 3), 35, np.uint8)
    cm.put_texts(bar, [(8, 4, head, 18, (255, 255, 255))])
    return np.vstack([bar, im])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--model", action="append", required=True, help='"이름=가중치.pt" (여러 번)')
    ap.add_argument("--out", default="outputs/train_mt_v3/monitor_patterns")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--real", default="match,persp,screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near", action="store_true", help="신호등이 가장 큰(가까운) 사진을 고름. 출력 이름에 _near")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    real = tuple(x for x in a.real.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    reg_t, clip_t = cm.util.BBoxTransform(), cm.util.ClipBoxes()
    names = [s.split("=", 1)[0] for s in a.model]
    models = {n: tm.build_model(cm.m, s.split("=", 1)[1]).to(dev).eval() for n, s in zip(names, a.model)}
    # 사진: 방해 화면 없는 원본에서 모든 모델이 켜진 불 색을 맞히는 것 (대회장 먼저, 신호등 크기 중간쯤)
    cands = photo_candidates(os.path.join(ROOT, "dataset/synth_signs_v2"), ["test_venue", "test"], a.seed, a.near)
    photos = {}
    for c, v in cands.items():
        for sp, w, it, img, da, ll, b in v:
            if all(judge(detect(md, img, dev, reg_t, clip_t), b, np.zeros(img.shape[:2], bool), a.thr)[1]["lamp"] == "ok"
                   for md in models.values()):
                photos[c] = (it, img, da, ll, b)
                print(f"{c}: 후보 {len(v)}장 중 {it['name']} (신호등 폭 {w:.0f}px)", flush=True)
                break

    pats = sorted(glob.glob(os.path.join(a.patterns, "*.png")))
    groups = {c: [] for c in LAMPS}; groups["solid"] = []
    for i, p in enumerate(p for p in pats if os.path.basename(p).startswith("solid")):
        groups["solid"].append((p, list(LAMPS)[i % 3]))
    for p in pats:
        n = os.path.basename(p)
        for c in LAMPS:
            if f"for-{c}_" in n or f"same-{c}_" in n:
                groups[c].append((p, c))

    # 사진 만들기 (모든 모델이 같은 사진)
    samples = {}
    for g, lst in groups.items():
        for p, c in lst:
            it, img, da, ll, b = photos[c]
            im = img.copy(); z = np.zeros(img.shape[:2], np.uint8)
            ok = oa.add_monitor(im, da.copy(), ll.copy(), z.copy(), z.copy(), b, np.random.default_rng(a.seed),
                                real=real, dots=False, pattern=cv2.imread(p))
            assert ok, p
            samples[(g, p)] = (im, np.abs(im.astype(np.int16) - img).max(2) > 12, b, c)

    res = {(n, k): judge(detect(md, im, dev, reg_t, clip_t), b, mon, a.thr)
           for n, md in models.items() for k, (im, mon, b, c) in samples.items()}

    LAMP_KO = {"ok": "맞음", "wrong": "색 틀림", "miss": "놓침"}
    COL_KO = {"red": "빨강", "yellow": "노랑", "green": "초록"}
    summary = {}
    for g, lst in groups.items():
        rows, tot = [], {n: dict(fake_lamp=0, fake_tl=0, img_fake=0, ok=0) for n in names}
        for p, c in lst:
            im, mon, b, _ = samples[(g, p)]
            cb = crop_box(mon, b)
            thumb = cv2.resize(cv2.imread(p), (LW - 16, int((LW - 16) * 9 / 16)), interpolation=cv2.INTER_AREA)
            left = np.full((BAR + TH, LW, 3), 45, np.uint8)
            left[BAR + 8:BAR + 8 + thumb.shape[0], 8:LW - 8] = thumb
            pn = os.path.splitext(os.path.basename(p))[0]
            cm.put_texts(left, [(8, 4, f"켜진 불: {COL_KO[c]}", 18, (255, 230, 120))] +
                         [(8, BAR + 20 + thumb.shape[0] + 22 * i, t, 15, (230, 230, 230))
                          for i, t in enumerate(pn.replace("dots_", "").replace("_on-", "  on-").split("  "))])
            tiles = [left]
            for n in names:
                marks, r = res[(n, (g, p))]
                t = tot[n]; t["fake_lamp"] += r["fake_lamp"]; t["fake_tl"] += r["fake_tl"]
                t["img_fake"] += int(r["fake_lamp"] > 0); t["ok"] += int(r["lamp"] == "ok")
                head = f"{n}   가짜 불빛 {r['fake_lamp']} · 가짜 신호등 {r['fake_tl']} · 진짜 불빛 {LAMP_KO[r['lamp']]}"
                tiles.append(tile(im, marks, b, cb, head))
            rows += [np.hstack(tiles), np.zeros((6, LW + TW * len(names), 3), np.uint8)]
        summary[g] = tot
        W = LW + TW * len(names)
        lines = [f"{n}: 가짜 불빛 {v['fake_lamp']}개 ({v['img_fake']}장) · 가짜 신호등 {v['fake_tl']}개 · 진짜 불빛 맞음 {v['ok']}/{len(lst)}"
                 for n, v in tot.items()]
        lines.append(f"자홍 굵은 선 = 모니터 위 가짜 불빛·가짜 신호등 · 색 박스 = 그 밖의 예측 · 흰 얇은 선 = 정답 · 점수 >= {a.thr} · "
                     f"모니터 효과 {','.join(real) or '없음'} · 모니터 주변 확대")
        title = ("단색 화면" if g == "solid" else f"켜진 불 {COL_KO[g]} — 땡땡이 화면") + f" {len(lst)}종 · {'가까운' if a.near else '중간 거리'} 신호등 (사진: {photos[list(LAMPS)[0] if g == 'solid' else g][0]['name']}"
        title += " 외)" if g == "solid" else ")"
        bar = np.full((40, W, 3), 60, np.uint8)
        cm.put_texts(bar, [(12, 7, "패턴", 24, (255, 230, 120))] + [(LW + TW * i + 12, 7, n, 24, (255, 230, 120)) for i, n in enumerate(names)])
        out = np.vstack([cm.header(W, title, lines), bar] + rows)
        path = os.path.join(a.out, f"monitor_patterns_{g}{'_near' if a.near else ''}.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"-> {path}", flush=True)
    json.dump(dict(thr=a.thr, real=list(real), photos={c: v[0]["name"] for c, v in photos.items()}, summary=summary),
              open(os.path.join(a.out, f"monitor_patterns{'_near' if a.near else ''}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
