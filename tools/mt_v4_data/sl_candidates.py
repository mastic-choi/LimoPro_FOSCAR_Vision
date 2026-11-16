#!/usr/bin/env python3
"""정지선(흰 가로선, sl) 검수 후보 만들기. 여러 데이터셋의 모든 프레임에 multitask-v3real 의 sl 확률을 구해
  <out>/<ds>/<split>/sl_prob/<stem>.png  (확률 x255, 학습 코드는 읽지 않음)
에 저장하고, 정지선이 잡혔거나(확률 > 0.5 픽셀 >= --min-px) 기존 sl 라벨이 있는 프레임을 후보로 모은다 → <out>/candidates.json

synth_signs_v2 는 같은 배경(meta.csv 의 ego_frame)에 표지판만 바꿔 붙인 변형이 여러 장이라, (split, ego_frame) 으로 묶어
대표 한 장만 검수하고 판정을 묶음 전체에 쓴다. bag 데이터는 한 장씩.

사용 (v3real 묶음 venv):
  python sl_candidates.py <bundle> <ckpt> <out> synth=<.../synth_signs_v2> bag=<.../bag_mt_v1>
"""
import csv, json, os, sys
from collections import defaultdict

import cv2
import numpy as np
import torch

bundle, ckpt, out = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
dsets = [(x.split("=", 1)[0], os.path.abspath(x.split("=", 1)[1])) for x in sys.argv[4:]]
MIN_PX = 80
sys.path.insert(0, os.path.join(bundle, "scripts"))
import make_model_compare_gif as m   # noqa: E402
import train_multitask as tm   # noqa: E402
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import train_seg as ts   # noqa: E402
model = tm.build_model(m, ckpt).to(m.device).eval()

groups = defaultdict(list)   # key -> [member dict]
stats = defaultdict(lambda: defaultdict(int))
for ds, root in dsets:
    ego = {}
    mp = os.path.join(root, "meta.csv")
    if os.path.exists(mp):
        ego = {r["image"]: r["ego_frame"] for r in csv.DictReader(open(mp))}
    for sp in sorted(os.listdir(root)):
        idir = os.path.join(root, sp, "images")
        if sp == "excluded" or not os.path.isdir(idir):
            continue
        od = os.path.join(out, ds, sp, "sl_prob"); os.makedirs(od, exist_ok=True)
        names = sorted(os.listdir(idir))
        for i in range(0, len(names), 16):
            ch = names[i:i + 16]
            imgs = [cv2.imread(os.path.join(idir, n)) for n in ch]
            xs = []
            for im in imgs:
                p = cv2.copyMakeBorder(im, tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                xs.append(((cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - tm.MEAN) / tm.STD).transpose(2, 0, 1))
            with torch.no_grad():
                *_, psl = tm.heads(model, torch.from_numpy(np.stack(xs)).to(m.device))
            psl = ts.crop(psl).float().cpu().numpy()
            for n, p in zip(ch, psl):
                stem = n[:-4]
                cv2.imwrite(os.path.join(od, stem + ".png"), (p * 255).round().astype(np.uint8))
                n_pos = int((p > 0.5).sum())
                exm = os.path.join(root, sp, "sl_masks", n)
                ex_pos = int((cv2.imread(exm, 0) > 0).sum()) if os.path.exists(exm) else -1   # -1 = 기존 라벨 없음
                stats[(ds, sp)]["frames"] += 1
                if n_pos >= MIN_PX or ex_pos > 0:
                    stats[(ds, sp)]["cand"] += 1
                    key = f"{ds}/{sp}/{ego.get(stem, stem)}"
                    groups[key].append({"ds": ds, "root": root, "split": sp, "stem": stem, "n_pos": n_pos, "ex_pos": ex_pos,
                                        "has_ex": os.path.exists(exm)})
            if (i // 16) % 50 == 0:
                print(f"  {ds}/{sp} {min(i + 16, len(names))}/{len(names)}", flush=True)

items = []
for key, mem in groups.items():
    mem.sort(key=lambda x: (not x["stem"].endswith("_s0"), x["stem"]))
    rep = mem[0]
    items.append({"id": key.replace("/", "__").replace(".png", ""), "key": key, "ds": rep["ds"], "split": rep["split"],
                  "rep": rep["stem"], "root": rep["root"], "n_pos": rep["n_pos"], "ex_pos": rep["ex_pos"], "has_ex": rep["has_ex"],
                  "members": [x["stem"] for x in mem]})
items.sort(key=lambda it: ({"bag": 0}.get(it["ds"], 1), {"test_bag1116": 0, "test_venue": 1, "test": 2, "test_bag": 3, "val": 4}.get(it["split"], 5), it["rep"]))
json.dump(items, open(os.path.join(out, "candidates.json"), "w"), ensure_ascii=False)
for (ds, sp), s in sorted(stats.items()):
    print(f"{ds}/{sp}: 프레임 {s['frames']}, 후보 {s['cand']}")
print(f"검수 묶음 {len(items)}개 (프레임 {sum(len(it['members']) for it in items)}장)")
