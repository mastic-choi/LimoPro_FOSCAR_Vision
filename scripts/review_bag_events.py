#!/usr/bin/env python3
"""주행 rosbag의 문제 순간(lane_follower/status 기준)을 사람이 분류하는 웹 앱.

키: 1 = 트랙 안 코너(차선 안에서 돌고 있음), 2 = 트랙 이탈, 3 = 그 밖, Backspace = 이전
화면: 그 순간 1초 전 | 그 순간 | 1초 후 (카메라 원본). 판정은 즉시 <분석 폴더>/event_labels.csv에 저장되고, 다시 켜면 이어서 한다.

사용 예:
  .venv/Scripts/python scripts/review_bag_events.py dataset/bags_20261107_analysis
입력: <분석 폴더>/bag_ensemble_check.json (문제 순간 목록), <주행>/cam_t.npy, <주행>/frames/*.jpg
"""
import csv, glob, json, os, sys, time
import cv2, numpy as np
import gradio as gr

A = sys.argv[1] if len(sys.argv) > 1 else "dataset/bags_20261107_analysis"
EV = json.load(open(os.path.join(A, "bag_ensemble_check.json"), encoding="utf-8"))
OUT = os.path.join(A, "event_labels.csv")
CHOICE = {"corner": "트랙 안 코너", "off": "트랙 이탈", "other": "그 밖"}
WHY = {"none": "차선 못 찾음", "width": "차로 폭 과대", "offset": "중심 오차 큼", "conf": "신뢰도 하락"}
BAGS = {os.path.basename(d).split("_")[0]: d for d in glob.glob(os.path.join(A, "*_1107_*"))}
CACHE = {}


def bag_frames(name):
    if name not in CACHE:
        d = BAGS[name]
        fr = sorted(glob.glob(os.path.join(d, "frames", "*.jpg")))
        CACHE[name] = (np.load(os.path.join(d, "cam_t.npy")), fr, np.array([int(os.path.basename(f)[:-4]) for f in fr]))
    return CACHE[name]


def frame_at(name, t):
    cam_t, fr, idx = bag_frames(name)
    return cv2.cvtColor(cv2.imread(fr[int(np.argmin(np.abs(cam_t[idx] - t)))]), cv2.COLOR_BGR2RGB)


def key(r):
    return f"{r['bag']}_{r['t']}"


def load_done():
    if not os.path.exists(OUT):
        return {}
    return {r["key"]: r["choice"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}


def save(done):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "bag", "t", "why", "choice", "time"])
        for r in EV:
            k = key(r)
            if k in done:
                w.writerow([k, r["bag"], r["t"], r["why"], done[k], time.strftime("%Y-%m-%d %H:%M:%S")])
    os.replace(tmp, OUT)


def view(k):
    done = load_done()
    if k >= len(EV):
        return None, None, None, f"### 끝났습니다. {len(done)}/{len(EV)}개 판정. 결과: {OUT}", k
    r = EV[k]
    prev = done.get(key(r))
    info = (f"### {k + 1} / {len(EV)}  ·  {r['bag']} {r['t']}초  ·  {WHY[r['why']]}  ·  4개 모델 일치도 {r['agree']:.2f}"
            f"{'  ·  이전 판정: ' + CHOICE[prev] if prev else ''}  ·  완료 {len(done)}개")
    return frame_at(r["bag"], r["t"] - 1), frame_at(r["bag"], r["t"]), frame_at(r["bag"], r["t"] + 1), info, k


def decide(k, c):
    if k < len(EV):
        done = load_done(); done[key(EV[k])] = c; save(done)
    return view(min(k + 1, len(EV)))


def first():
    done = load_done()
    return view(next((k for k, r in enumerate(EV) if key(r) not in done), len(EV)))


KEY_JS = """
() => {
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const m = {'1': 'btn-corner', '2': 'btn-off', '3': 'btn-other', 'Backspace': 'btn-back'};
    const id = m[e.key];
    if (id) { e.preventDefault(); document.getElementById(id)?.click(); }
  });
}
"""

with gr.Blocks(title="주행 문제 순간 분류", js=KEY_JS) as demo:
    gr.Markdown("**1** 트랙 안 코너(차선 안에서 돌고 있음) · **2** 트랙 이탈 · **3** 그 밖 · **Backspace** 이전")
    info = gr.Markdown()
    with gr.Row():
        a = gr.Image(label="1초 전", height=320)
        b = gr.Image(label="그 순간", height=320)
        c = gr.Image(label="1초 후", height=320)
    idx = gr.State(0)
    with gr.Row():
        b1 = gr.Button("1 트랙 안 코너", elem_id="btn-corner", variant="primary")
        b2 = gr.Button("2 트랙 이탈", elem_id="btn-off", variant="primary")
        b3 = gr.Button("3 그 밖", elem_id="btn-other")
        bb = gr.Button("이전", elem_id="btn-back")
    outs = [a, b, c, info, idx]
    b1.click(lambda k: decide(k, "corner"), idx, outs)
    b2.click(lambda k: decide(k, "off"), idx, outs)
    b3.click(lambda k: decide(k, "other"), idx, outs)
    bb.click(lambda k: view(max(0, k - 1)), idx, outs)
    demo.load(first, None, outs)

if __name__ == "__main__":
    demo.queue().launch(server_port=7862, inbrowser=True)
