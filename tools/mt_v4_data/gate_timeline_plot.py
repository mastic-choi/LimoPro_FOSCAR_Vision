#!/usr/bin/env python3
"""DA 주행 검사 타임라인 그림 (11-18): 대회장 bag 마다 v4·v5·v6 의 화면 아래 절반 DA 비율을 시간축으로 그린다.

DA·차선이 통째로 꺼진 프레임(DA < 3%, LL < 0.3%, 다른 모델은 DA > 30%)을 점으로 찍고 개수를 제목에 적는다. 신호등 미션 구간은 회색 띠.
숫자 판정은 bag_da_gate.py, 이 그림은 눈으로 보는 용도. 한글 글꼴: Noto Sans CJK.
사용: python -I tools/mt_v4_data/gate_timeline_plot.py --bundle <LimoPro_mt_v4> --frames-root <fullbag1118> --weights <v4.pt> <v5.pt> <v6.pt> --out gate_timeline.png [--cache cache.json]
"""
import os, sys, csv, glob, json, cv2, numpy as np, torch
import argparse
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bundle", required=True, help="학습 번들 폴더 (scripts/, models/HybridNets 가 있는 LimoPro_mt_v4)")
ap.add_argument("--frames-root", required=True); ap.add_argument("--out", required=True, help="그림 파일 (.png)")
ap.add_argument("--weights", nargs=3, required=True, metavar=("V4", "V5", "V6"), help="v4 v5 v6 best.pt 순서")
ap.add_argument("--cache", default="", help="모델 출력 캐시 json (다시 그릴 때 추론 생략)")
A = ap.parse_args(); A.out = os.path.abspath(A.out); A.cache = os.path.abspath(A.cache) if A.cache else ""
sys.path.insert(0, os.path.join(A.bundle, "scripts"))
import train_multitask as tm, make_model_compare_gif as m
os.chdir(m.ROOT)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
V = ("v4", "v5", "v6")
M = {v: tm.build_model(m, w).cuda().eval() for v, w in zip(V, A.weights)}
runs = sorted(glob.glob(os.path.join(A.frames_root, "arall*"))) + sorted(glob.glob(os.path.join(A.frames_root, "ar4*")))
CACHE = A.cache
res = json.load(open(CACHE)) if CACHE and os.path.exists(CACHE) else {}
for r in runs:
    b = os.path.basename(r)
    if b in res: continue
    rows = list(csv.DictReader(open(f"{r}/frames.csv")))[::2]; o = {"t": [float(x["t"]) for x in rows], "mission": [x["mission_id"] for x in rows]}
    for v in V: o[v + "_da"], o[v + "_ll"] = [], []
    with torch.no_grad():
        for k in range(0, len(rows), 64):
            x = torch.from_numpy(np.stack([((cv2.cvtColor(cv2.copyMakeBorder(cv2.resize(cv2.imread(f"{r}/frames/{q['name']}"), (640, 360), interpolation=cv2.INTER_AREA), tm.PAD, tm.PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114,) * 3), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1) for q in rows[k:k + 64]])).cuda()
            for v in V:
                *_, da, ll, _ = tm.heads(M[v], x)
                o[v + "_da"] += (da.float()[:, tm.PAD:-tm.PAD][:, 180:] >= 0.5).float().mean((1, 2)).tolist(); o[v + "_ll"] += (ll.float()[:, tm.PAD:-tm.PAD][:, 180:] >= 0.5).float().mean((1, 2)).tolist()
    res[b] = o
    if CACHE: json.dump(res, open(CACHE, "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(fp)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
COL = {"v4": "#2a78d6", "v5": "#eb6834", "v6": "#1baf7a"}; STY = {"v4": "-", "v5": (0, (4, 2)), "v6": "-"}
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
order = ["arall_1118_135742", "ar4_1118_135516", "arall_1118_134536", "arall_1118_135201", "arall_1118_135555", "arall_1118_134729", "arall_1118_134843", "arall_1118_135010", "arall_1118_135105", "ar4_1118_135356", "ar4_1118_135437"]
fig, axes = plt.subplots(len(order), 1, figsize=(14, 2.0 * len(order) + 1.6), facecolor=SURF)
for ax, b in zip(axes, order):
    o = res[b]; t = np.array(o["t"]); ax.set_facecolor(SURF)
    mis = np.array(o["mission"]); tl = t[mis == "1"]
    if len(tl): ax.axvspan(tl.min(), tl.max(), color="#f1efe8", zorder=0); ax.text(tl.min() + 0.5, 0.93, "신호등 미션", fontsize=9, color=MUTED, va="top")
    cnt = {}
    for v in V:
        da, ll = np.array(o[v + "_da"]), np.array(o[v + "_ll"]); others = [np.array(o[w + "_da"]) for w in V if w != v]
        off = (da < 0.03) & (ll < 0.003) & (np.max(others, axis=0) > 0.3); cnt[v] = int(off.sum())
        ax.plot(t, da, linestyle=STY[v], color=COL[v], lw=2 if v != "v5" else 1.6, label=v, zorder=3 if v == "v6" else 2)
        if off.any(): ax.scatter(t[off], np.zeros(off.sum()) + 0.015, s=10, color=COL[v], zorder=4, linewidths=0)
    held = "  (학습 안 씀)" if b in ("arall_1118_135742", "ar4_1118_135516") else ""
    ax.set_title(f"{b}{held}     DA 꺼진 프레임  v4 {cnt['v4']} · v5 {cnt['v5']} · v6 {cnt['v6']}", loc="left", fontsize=11, color=INK, fontweight="bold" if held else "normal")
    ax.set_ylim(0, 1); ax.set_xlim(t.min(), t.max()); ax.set_yticks([0, 0.5, 1]); ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8); [s.set_visible(False) for s in ax.spines.values()]; ax.set_ylabel("DA", color=MUTED, fontsize=9)
    ax.text(t.max(), np.array(o["v6_da"])[-30:].mean(), " v6", color=INK, fontsize=8, va="center", clip_on=False)
axes[-1].set_xlabel("bag 시작부터 시간 (초)", color=MUTED)
fig.suptitle("11-18 대회장 bag DA 주행 검사: 화면 아래 절반 중 DA 비율 (0 = DA 통째로 꺼짐)", x=0.01, ha="left", fontsize=14, color=INK, y=0.995)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, ["v4 (지금 차량 기준)", "v5 (사용 금지)", "v6 (새 릴리즈)"], loc="upper right", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 0.995))
fig.tight_layout(rect=(0, 0, 1, 0.975)); out = A.out; fig.savefig(out, dpi=110); print(out)
