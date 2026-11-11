#!/usr/bin/env python3
"""검출 점수(confidence)가 언제 낮게 나오는지: 정답 박스마다 '그 자리에서 정답 클래스 점수'와
'다른 클래스 중 가장 높은 점수(경쟁 점수)'를 재서 클래스·크기·조명별로 묶는다.

정답 박스와 IoU 0.5 이상으로 겹치는 예측(앵커+회귀) 중
  true  = 정답 클래스 점수의 최댓값
  rival = 같은 앵커들에서 다른 클래스 점수의 최댓값 (신호등 몸체-불빛 쌍은 서로 경쟁으로 치지 않음)
결과: <out>/conf_stats.json, conf_report.md, 표준 출력 요약
사용: python scripts/eval_mt_conf.py <best.pt> [--splits test test_venue test_bag] [--out outputs/train_mt_v2/conf]
"""
import argparse, collections, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_multitask as tm
import make_model_compare_gif as m

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--splits", nargs="*", default=["test", "test_venue", "test_bag"])
ap.add_argument("--data", default="dataset/synth_signs_v2")
ap.add_argument("--out", default="outputs/train_mt_v2/conf")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
os.chdir(m.ROOT)
sys.path.insert(0, os.path.join(m.MODELS_DIR, "HybridNets"))
import utils.utils as util

C = tm.CLASSES
LAMP = {C.index(c) for c in ("light_red", "light_yellow", "light_green")}
TL = C.index("traffic_light")
model = tm.build_model(m, a.ckpt).to(m.device).eval()
reg_t = util.BBoxTransform()
amp = m.device.type == "cuda" and "ZLUDA" not in torch.cuda.get_device_name(0)


def size_bin(s):
    return "<14px" if s < 14 else "14-20px" if s < 20 else "20-30px" if s < 30 else ">=30px"


def iou(box, b):  # box (4,), b (N,4)
    x1 = np.maximum(box[0], b[:, 0]); y1 = np.maximum(box[1], b[:, 1])
    x2 = np.minimum(box[2], b[:, 2]); y2 = np.minimum(box[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


recs = []
with torch.no_grad():
    for split in a.splits:
        items = tm.seg_items(a.data, split)
        dl = torch.utils.data.DataLoader(tm.Items(items, False), batch_size=a.batch, num_workers=0, collate_fn=tm.collate)  # Windows: 스크립트 최상위 코드라 워커를 쓰면 멈춤
        for x, *_, ann, names in dl:
            x = x.to(m.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                reg, cls, anchors, *_ = tm.heads(model, x)
            boxes = reg_t(anchors.float(), reg.float()).cpu().numpy()      # (B,A,4) x1y1x2y2 (패딩 포함 좌표)
            cls = cls.float().cpu().numpy()                                  # (B,A,C)
            for k, name in enumerate(names):
                g = ann[k].numpy(); g = g[g[:, 4] >= 0]
                stem = name.split("/")[-1]
                light = "venue" if stem[:1] in ("v", "b") else "general"
                for gb in g:
                    c = int(gb[4])
                    if c >= 10:
                        continue
                    ov = iou(gb[:4], boxes[k])
                    sel = ov >= 0.5
                    if not sel.any():
                        sel = ov >= ov.max() - 1e-6
                    sc = cls[k][sel]
                    true = float(sc[:, c].max())
                    others = [j for j in range(len(C)) if j != c and not (c == TL and j in LAMP) and not (c in LAMP and j == TL)]
                    rv = sc[:, others]
                    rival_j = others[int(np.unravel_index(rv.argmax(), rv.shape)[1])]
                    s = float(np.sqrt((gb[2] - gb[0]) * (gb[3] - gb[1])))
                    recs.append(dict(split=split, img=stem, cls=C[c], size=round(s, 1), bin=size_bin(s), light=light,
                                     true=round(true, 3), rival=round(float(rv.max()), 3), rival_cls=C[rival_j]))

os.makedirs(a.out, exist_ok=True)
json.dump(recs, open(os.path.join(a.out, "conf_stats.json"), "w"), indent=0)


def table(key_fn, keys, title):
    lines = [f"\n### {title}\n", "| 구분 | 개수 | 정답 점수 중앙값 | 정답 점수 <0.5 비율 | 경쟁 점수 중앙값 | 정답<경쟁 비율 |", "|---|---:|---:|---:|---:|---:|"]
    for kk in keys:
        r = [x for x in recs if key_fn(x) == kk]
        if not r:
            continue
        t = np.array([x["true"] for x in r]); v = np.array([x["rival"] for x in r])
        lines.append(f"| {kk} | {len(r)} | {np.median(t):.2f} | {np.mean(t < 0.5):.0%} | {np.median(v):.2f} | {np.mean(t < v):.0%} |")
    return "\n".join(lines)


md = ["# 검출 점수(confidence) 분석", f"가중치 `{a.ckpt}`, split {', '.join(a.splits)}, 정답 박스 {len(recs)}개 (limo/cone 제외)"]
md.append(table(lambda x: x["cls"], C[:10], "클래스별"))
md.append(table(lambda x: x["bin"], ["<14px", "14-20px", "20-30px", ">=30px"], "크기별 (박스 √면적)"))
md.append(table(lambda x: x["light"], ["general", "venue"], "조명별"))
md.append(table(lambda x: x["split"], a.splits, "split별"))
low = [x for x in recs if x["true"] < 0.5]
cnt = collections.Counter((x["cls"], x["rival_cls"]) for x in low if x["rival"] > x["true"])
md.append("\n### 정답 점수 <0.5인데 다른 클래스가 더 높은 경우 (정답 → 경쟁)\n")
md += [f"- {a_} → {b_}: {n}" for (a_, b_), n in cnt.most_common(12)]
open(os.path.join(a.out, "conf_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
