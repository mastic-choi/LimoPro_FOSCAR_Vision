#!/usr/bin/env python3
"""다중 작업 실험 비교표 (1차 / 실험 A / 실험 B) -> markdown.
사용: python -I scripts/summarize_mt_exp.py <출력.md> 이름=결과폴더 [이름=결과폴더 ...]
각 결과 폴더: test.json (train_multitask.py), eval/ab_stats.json (eval_mt_ab.py), probe.txt (선택)
"""
import json, os, sys

out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
L = ["# 다중 작업 실험 비교", ""]


def load(d, f):
    p = os.path.join(d, f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


T = {n: load(d, "test.json") for n, d in runs}
AB = {n: load(d, "eval/ab_stats.json") for n, d in runs}
names = [n for n, _ in runs]
L += ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for n in names:
    if T[n] is None:
        print(f"경고: {n} test.json 없음")
L.append("| best epoch | " + " | ".join(str(T[n]["best"]["epoch"]) if T[n] else "-" for n in names) + " |")
for sp, lab in (("test", "기존 test"), ("test_venue", "대회장 test"), ("test_bag", "test_bag")):
    L.append(f"| {lab} da / ll | " + " | ".join(
        f"{T[n][sp]['all']['da_iou']:.3f} / {T[n][sp]['all']['ll_iou']:.3f}" if T[n] else "-" for n in names) + " |")
    L.append(f"| {lab} mAP50 | " + " | ".join(f"{T[n][sp]['all']['map50']:.3f}" if T[n] else "-" for n in names) + " |")
    for c in ("sign_A", "sign_B", "parking", "pedestrian", "bump"):
        L.append(f"| {lab} AP50 {c} | " + " | ".join(
            f"{T[n][sp]['all']['ap50'][c]:.2f}" if T[n] and T[n][sp]['all']['ap50'].get(c) is not None else "-" for n in names) + " |")
L.append("| 대회장 test 흰 가로선 IoU | " + " | ".join(
    f"{T[n]['test_venue']['all']['sl_iou']:.3f}" if T[n] and T[n]['test_venue']['all'].get('sl_iou') is not None else "-" for n in names) + " |")
for sp in ("train", "test", "test_venue", "test_bag"):
    L.append(f"| A/B 2택 정답률 ({sp}) | " + " | ".join(
        f"{AB[n]['stats'][sp]['two_way_acc']:.0%}" if AB[n] and sp in AB[n]["stats"] else "-" for n in names) + " |")
    L.append(f"| A/B 최종 예측 정답률 ({sp}) | " + " | ".join(
        f"{AB[n]['stats'][sp]['final_acc']:.0%}" if AB[n] and sp in AB[n]["stats"] else "-" for n in names) + " |")
L.append("")
for n, d in runs:
    p = os.path.join(d, "probe.txt")
    if os.path.exists(p):
        L += [f"## linear probe: {n}", "", "```", open(p, encoding="utf-8").read().strip(), "```", ""]
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
