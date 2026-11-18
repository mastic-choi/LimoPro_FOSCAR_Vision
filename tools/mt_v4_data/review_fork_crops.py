#!/usr/bin/env python3
"""갈림길 분류기 크롭 사람 검수: 크롭마다 글자(A/B)를 직접 고르거나 버린다 (사용자 11-18).

대회장 갈림길 표지판은 양면(앞 A, 뒤 B)이라, 한 통과 안에서도 뒷면 글자가 잡힌 크롭이 섞인다.
통과 단위로 붙인 글자(폴더)는 '추정'으로만 보여 주고, 사람이 크롭마다 정한다.
모양이 안 보이는 크롭(흔들림·가림·현수막 등)은 버린다: 안 보이는 것에 글자를 붙이면 배경으로 배움.

  serve <crops 루트> [--port 7865] : 웹 앱. <루트>/<run>/{A,B}/*.png 를 run·크기 순으로 보여 줌.
                                      판정은 즉시 <루트>/crop_letter.csv (file, guess, choice), 다시 켜면 이어서.
  apply <crops 루트>               : choice 가 A/B 이면 그 폴더로, drop 이면 <run>/dropped/ 로 옮김 (지우지 않음)
키: 1 A · 2 B · 3 버림 · Backspace 이전
"""
import csv, os, shutil, sys

import cv2
import numpy as np

cmd, root = sys.argv[1], os.path.abspath(sys.argv[2])
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 7865
OUT = os.path.join(root, "crop_letter.csv")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import fork_crop_rule as rule  # noqa: E402


def items():
    out = []
    for run in sorted(os.listdir(root)):
        p = os.path.join(root, run, "crops.csv")
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            if r["label"] not in ("A", "B") or not r["file"]: continue
            f = next((os.path.join(root, run, L, r["file"]) for L in ("A", "B") if os.path.exists(os.path.join(root, run, L, r["file"]))), None)
            if f: out.append(dict(file=f, guess=os.path.basename(os.path.dirname(f)), size=int(r["size_px_1280"]), run=run))
    return out


def done_map():
    return {r["file"]: r["choice"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else {}


if cmd == "apply":
    n = {"A": 0, "B": 0, "drop": 0, "same": 0}
    for f, c in done_map().items():
        if not os.path.exists(f): continue
        cur = os.path.basename(os.path.dirname(f)); run_dir = os.path.dirname(os.path.dirname(f))
        if c == cur: n["same"] += 1; continue
        d = os.path.join(run_dir, "dropped", cur) if c == "drop" else os.path.join(run_dir, c)
        os.makedirs(d, exist_ok=True); shutil.move(f, os.path.join(d, os.path.basename(f))); n[c] += 1
    print("옮김 (A로, B로, 버림) / 그대로:", n); sys.exit(0)

import gradio as gr  # noqa: E402
IT = items(); done = done_map()


def save():
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "guess", "choice"])
        for it in IT:
            if it["file"] in done: w.writerow([it["file"], it["guess"], done[it["file"]]])


def view(k):
    if k >= len(IT):
        return None, f"**끝** ({len(done)}/{len(IT)} 판정). 창을 닫으셔도 됩니다.", k
    it = IT[k]; im = cv2.imread(it["file"])
    col = cv2.resize(im, (300, 300), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(cv2.resize(rule.preprocess(im, rgb=False, size=96), (300, 300), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    pic = cv2.cvtColor(np.hstack([col, np.full((300, 10, 3), 30, np.uint8), g]), cv2.COLOR_BGR2RGB)
    prev = done.get(it["file"], "-")
    return pic, (f"**{k + 1} / {len(IT)}** · 통과 추정 **{it['guess']}** · 지름 {it['size']}px · {it['run']} · 이전 판정: {prev}"
                 "  (왼쪽 원본, 오른쪽 분류기가 보는 흑백)"), k


def decide(k, c):
    if k < len(IT): done[IT[k]["file"]] = c; save()
    return view(k + 1)


def first():
    k = next((i for i, it in enumerate(IT) if it["file"] not in done), len(IT)); return view(k)


KEY_JS = """
() => {
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const m = {'1': 'btn-a', '2': 'btn-b', '3': 'btn-drop', 'Backspace': 'btn-back'};
    const id = m[e.key];
    if (id) { e.preventDefault(); document.getElementById(id)?.click(); }
  });
}
"""
with gr.Blocks(title="갈림길 크롭 글자 검수", js=KEY_JS) as demo:
    gr.Markdown("**1** A · **2** B · **3** 버림 (흔들려 뭉개짐·가림·현수막·글자 안 보임) · **Backspace** 이전  —  "
                "양면 표지판이라 '통과 추정'과 다를 수 있음: 보이는 면의 글자를 고르세요")
    info = gr.Markdown(); img = gr.Image(show_label=False, height=330); idx = gr.State(0)
    with gr.Row():
        ba = gr.Button("1 A", elem_id="btn-a", variant="primary"); bb_ = gr.Button("2 B", elem_id="btn-b", variant="primary")
        bd = gr.Button("3 버림", elem_id="btn-drop"); bk = gr.Button("이전", elem_id="btn-back")
    outs = [img, info, idx]
    ba.click(lambda k: decide(k, "A"), idx, outs); bb_.click(lambda k: decide(k, "B"), idx, outs)
    bd.click(lambda k: decide(k, "drop"), idx, outs); bk.click(lambda k: view(max(0, k - 1)), idx, outs)
    demo.load(first, None, outs)
demo.launch(server_name="127.0.0.1", server_port=port)
