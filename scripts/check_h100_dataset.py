#!/usr/bin/env python3
"""H100에서 v2 학습에 쓴 데이터셋 zip(LimoPro_dataset_mt_v1.zip)을 풀지 않고 검사한다.

1) zip SHA256
2) synth_signs_v2 각 split의 장수, 라벨·마스크(labels, da/ll_masks, sl_masks, sl_valid)가 이 PC와 같은지 (내용 해시 비교)
3) Klein으로 다듬은 train 사진에 학습 중 모니터·케이블 증강(mt_online_aug)을 그려 성공률과 미리보기
4) cones_v1 박스 중 √넓이 < 14px(--ignore-small 14) 비율
결과: <out>/check.json, <out>/klein_monitor.jpg

사용 예:
  .venv/Scripts/python scripts/check_h100_dataset.py %USERPROFILE%/Downloads/LimoPro_dataset_mt_v1.zip
"""
import argparse, csv, hashlib, io, json, os, sys, zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SPLITS = ("train", "val", "test", "test_venue", "test_bag")
SUBS = ("images", "labels", "da_masks", "ll_masks", "sl_masks", "sl_valid")
LOCAL = os.path.join(ROOT, "dataset", "synth_signs_v2")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def content_key(name, b):
    """라벨은 글자 그대로, 마스크는 PNG를 풀어 픽셀(0/1)로 비교 (PNG 압축 방식이 달라도 같으면 같음)."""
    if name.endswith(".txt"):
        return hashlib.sha1(b.replace(b"\r\n", b"\n").strip()).hexdigest()
    m = cv2.imdecode(np.frombuffer(b, np.uint8), 0)
    return hashlib.sha1(np.packbits(m > 0).tobytes()).hexdigest() if m is not None else "bad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "h100_dataset_check"))
    ap.add_argument("--n-aug", type=int, default=400, help="모니터 증강을 시험할 Klein 사진 수")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"zip": a.zip, "bytes": os.path.getsize(a.zip)}
    print("sha256 ...", flush=True)
    res["sha256"] = sha256_file(a.zip)
    print(res["sha256"], flush=True)

    z = zipfile.ZipFile(a.zip)
    names = [n for n in z.namelist() if not n.endswith("/")]
    pre = next(n[:n.index("synth_signs_v2/")] for n in names if "synth_signs_v2/" in n)
    res["prefix"] = pre
    res["top"] = sorted({n[len(pre):].split("/")[0] for n in names if n.startswith(pre)})
    idx = defaultdict(dict)   # (split, sub) -> {파일 이름: zip 경로}
    for n in names:
        if n.startswith(pre + "synth_signs_v2/"):
            p = n[len(pre + "synth_signs_v2/"):].split("/")
            if len(p) == 3 and p[0] in SPLITS and p[1] in SUBS:
                idx[(p[0], p[1])][p[2]] = n

    # 2) 장수, 라벨·마스크 비교
    cmp = {}
    for s in SPLITS:
        row = {"images": len(idx[(s, "images")])}
        for sub in SUBS[1:]:
            remote = idx[(s, sub)]
            ld = os.path.join(LOCAL, s, sub)
            local = set(os.listdir(ld)) if os.path.isdir(ld) else set()
            same = diff = 0; diff_ex = []
            for f in sorted(set(remote) & local):
                if content_key(f, z.read(remote[f])) == content_key(f, open(os.path.join(ld, f), "rb").read()):
                    same += 1
                else:
                    diff += 1; diff_ex.append(f) if len(diff_ex) < 5 else None
            row[sub] = dict(server=len(remote), local=len(local), same=same, diff=diff, diff_ex=diff_ex,
                            only_server=len(set(remote) - local), only_local=len(local - set(remote)))
        cmp[s] = row
        print(s, json.dumps(row, ensure_ascii=False), flush=True)
    res["compare"] = cmp

    # Klein 기록
    kcsv = [n for n in names if n.startswith(pre + "synth_signs_v2/train/klein_refine") and n.endswith(".csv")]
    kept = set(); st = Counter()
    for n in kcsv:
        for r in csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")):
            st[r.get("status")] += 1
            if r.get("status") == "kept":
                kept.add(r["image"])
    res["klein"] = {"csv": kcsv, "status": dict(st), "kept": len(kept)}
    print("klein", res["klein"], flush=True)

    # 3) Klein 사진에 모니터·케이블 증강
    import train_multitask as tm
    import mt_online_aug as oa
    rd = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
    rng = np.random.default_rng(0)
    imgs = idx[("train", "images")]; labs = idx[("train", "labels")]
    cand = [k for k in sorted(kept) if k + ".png" in imgs and k + ".txt" in labs]
    rng.shuffle(cand)
    tried = ok = 0; tiles = []; tl_keep = []
    for k in cand:
        if tried >= a.n_aug:
            break
        txt = z.read(labs[k + ".txt"]).decode()
        bx = []
        for l in txt.split("\n"):
            v = l.split()
            if len(v) == 5:
                c, x, y, w, h = int(v[0]), *map(float, v[1:])
                bx.append([(x - w / 2) * 640, (y - h / 2) * 360, (x + w / 2) * 640, (y + h / 2) * 360, c])
        b = np.array(bx, np.float32).reshape(-1, 5)
        if not (b[:, 4] == oa.TL_CLS).any():
            continue
        tried += 1
        img = rd(imgs[k + ".png"])[..., :3].copy()
        m = lambda sub: (rd(idx[("train", sub)][k + ".png"]) > 0).astype(np.uint8) if k + ".png" in idx[("train", sub)] \
            else np.zeros((360, 640), np.uint8)
        da, ll, sl, slv = m("da_masks"), m("ll_masks"), m("sl_masks"), m("sl_valid")
        for t in b[b[:, 4] == oa.TL_CLS]:   # 신호등 몸체를 얼마나 골라내는지 (박스 대비 몸체 비율)
            fg = oa.tl_foreground(img, t[:4])
            tl_keep.append(0.0 if fg is None else float(fg.mean()))
        o = img.copy()
        if oa.add_monitor(img, da, ll, sl, slv, b, rng):
            ok += 1
            if rng.random() < 0.3 and len(tiles) < 10:
                tiles.append(np.hstack([o, img]))
    for k in cand[:4]:
        img = rd(imgs[k + ".png"])[..., :3].copy(); o = img.copy()
        z0 = np.zeros((360, 640), np.uint8)
        oa.add_cable(img, z0.copy(), z0.copy(), rng)
        tiles.append(np.hstack([o, img]))
    if tiles:
        cv2.imwrite(os.path.join(a.out, "klein_monitor.jpg"), cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    tk = np.array(tl_keep)
    res["monitor_on_klein"] = dict(tl_images=tried, drawn=ok, rate=round(ok / max(tried, 1), 3),
                                   tl_fg_ratio_median=round(float(np.median(tk)), 3) if len(tk) else None,
                                   tl_fg_none_or_lt20pct=int((tk < 0.2).sum()), tl_boxes=len(tk))
    print("monitor", res["monitor_on_klein"], flush=True)

    # 4) 작은 박스 비율 (--ignore-small 14). 콘 사진은 학습 때 비율 유지로 640x360 안에 넣으므로 원본 크기로 축척 계산
    from PIL import Image
    cl = [n for n in names if "/cones_v1/labels/" in n and n.endswith(".txt")]
    ci = {n.split("/cones_v1/images/")[1].rsplit(".", 1)[0]: n for n in names if "/cones_v1/images/" in n}
    allb = small = 0; by_src = Counter(); small_src = Counter(); noimg = 0
    for n in cl:
        key = n.split("/cones_v1/labels/")[1][:-4]
        if key not in ci:
            noimg += 1; continue
        w0, h0 = Image.open(z.open(ci[key])).size
        s = min(640 / w0, 360 / h0)
        src = "fsoco" if "fsoco" in key.lower() else "tracon"
        for l in z.read(n).decode().split("\n"):
            v = l.split()
            if len(v) == 5:
                w, h = float(v[3]) * w0 * s, float(v[4]) * h0 * s
                allb += 1; by_src[src] += 1
                if (w * h) ** 0.5 < 14:
                    small += 1; small_src[src] += 1
    res["cones"] = dict(label_files=len(cl), image_files=len(ci), no_image=noimg, boxes=allb, small_lt14=small,
                        small_rate=round(small / max(allb, 1), 3), by_src=dict(by_src), small_by_src=dict(small_src))
    print("cones", res["cones"], flush=True)
    sb = Counter(); ss = Counter()   # 합성 train 박스(640x360 그대로)
    for f, n in labs.items():
        for l in z.read(n).decode().split("\n"):
            v = l.split()
            if len(v) == 5:
                c = tm.CLASSES[int(v[0])]; sb[c] += 1
                if (float(v[3]) * 640 * float(v[4]) * 360) ** 0.5 < 14:
                    ss[c] += 1
    res["synth_small_lt14"] = {c: f"{ss[c]}/{sb[c]}" for c in tm.CLASSES if sb[c]}
    print("synth small", res["synth_small_lt14"], flush=True)
    json.dump(res, open(os.path.join(a.out, "check.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
