#!/usr/bin/env python3
"""4차 데이터셋 후보(round4_candidates.py 결과)를 사람이 분류하는 웹 앱.

키: 1 = 사용, 2 = 버림, 3 = da 비움(트랙 밖, 내 차로 없음 / edge·off 후보), Backspace = 이전
  edge : 1이면 3개 모델 합의 라벨을 씀 (HybridNets가 코너 안쪽을 깎은 프레임)
  off  : 1이면 3개 모델 합의 라벨, 3이면 da·ll 모두 비운 라벨 (edge도 3 가능)
  wide_ego : wide에서 버린 3차선 라벨을 내 차로만 남기게 바꾼 것 (wide_to_ego.py). 1이면 사용
  wide : 1이면 원래 사람 라벨을 다시 넣음 (차로가 화면 아래를 꽉 채운 정상 코너. 3차선 전체를 칠한 것은 2)
판정은 즉시 <후보 폴더>/decisions.csv에 저장되고, 다시 켜면 이어서 한다.

사용 예:
  .venv/Scripts/python scripts/review_round4.py dataset/round4_candidates
"""
import csv, json, os, sys, time
import cv2
import gradio as gr

D = sys.argv[1] if len(sys.argv) > 1 else "dataset/round4_candidates"
ITEMS = json.load(open(os.path.join(D, "items.json"), encoding="utf-8"))
OUT = os.path.join(D, "decisions.csv")
CHOICE = {"use": "사용", "drop": "버림", "empty": "da 비움"}
HELP = {
    "edge": "코너 안쪽 깎임 — 오른쪽 '3개 모델 합의'가 내 차로를 맞게 칠했으면 **1 사용**, 틀렸으면 **2 버림**, 트랙 밖이라 내 차로가 없으면 **3 da 비움**",
    "off": "트랙 이탈 — 합의 라벨이 맞으면 **1 사용**, 내 차로가 안 보이면 **3 da 비움**, 애매하면 **2 버림**",
    "wide_ego": "3차선 라벨을 내 차로만 남기게 바꾼 것 — 오른쪽이 내 차로만 맞게 칠했으면 **1 사용**, 다른 차로가 섞이거나 잘렸으면 **2 버림**",
    "wide": "넓은 da 라벨 — 내 차로만 칠한 정상 라벨이면 **1 사용**, 3차선 전체 등 기준과 다르면 **2 버림**",
}


def load_done():
    if not os.path.exists(OUT):
        return {}
    return {r["id"]: r["choice"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}


def save(done):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "kind", "choice", "time"])
        for it in ITEMS:
            if it["id"] in done:
                w.writerow([it["id"], it["kind"], done[it["id"]], time.strftime("%Y-%m-%d %H:%M:%S")])
    os.replace(tmp, OUT)


def view(k):
    done = load_done()
    if k >= len(ITEMS):
        return None, f"### 끝났습니다. {len(done)}/{len(ITEMS)}개 판정. 결과: {OUT}", k
    it = ITEMS[k]
    prev = done.get(it["id"])
    where = f"{it['bag']} {it['t']}초" if "bag" in it else it["frame"]
    n_kind = sum(1 for x in ITEMS if x["kind"] == it["kind"])
    i_kind = sum(1 for x in ITEMS[:k + 1] if x["kind"] == it["kind"])
    info = (f"### {k + 1} / {len(ITEMS)}  ·  [{it['kind']} {i_kind}/{n_kind}]  ·  {where}"
            f"{'  ·  이전 판정: ' + CHOICE[prev] if prev else ''}  ·  완료 {len(done)}개\n\n{HELP[it['kind']]}")
    img = cv2.cvtColor(cv2.imread(os.path.join(D, "view", it["id"] + ".jpg")), cv2.COLOR_BGR2RGB)
    return img, info, k


def decide(k, c):
    if k < len(ITEMS):
        it = ITEMS[k]
        if c == "empty" and it["kind"] in ("wide", "wide_ego"):
            img, info, _ = view(k)  # 사람 라벨(wide)에는 da 비움이 없음
            return img, "### ⚠ wide 후보에는 3(da 비움)을 쓸 수 없습니다. 1 사용 / 2 버림\n\n" + info, k
        done = load_done(); done[it["id"]] = c; save(done)
    return view(min(k + 1, len(ITEMS)))


def first():
    done = load_done()
    return view(next((k for k, it in enumerate(ITEMS) if it["id"] not in done), len(ITEMS)))


KEY_JS = """
() => {
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const m = {'1': 'btn-use', '2': 'btn-drop', '3': 'btn-empty', 'Backspace': 'btn-back'};
    const id = m[e.key];
    if (id) { e.preventDefault(); document.getElementById(id)?.click(); }
  });
}
"""

with gr.Blocks(title="4차 데이터셋 후보 분류", js=KEY_JS) as demo:
    gr.Markdown("**1** 사용 · **2** 버림 · **3** da 비움(edge·off 후보) · **Backspace** 이전  —  "
                "초록 = da, 노란 테두리 = da 경계, 자홍 = 차선(ll)")
    info = gr.Markdown()
    img = gr.Image(show_label=False, height=420)
    idx = gr.State(0)
    with gr.Row():
        b1 = gr.Button("1 사용", elem_id="btn-use", variant="primary")
        b2 = gr.Button("2 버림", elem_id="btn-drop")
        b3 = gr.Button("3 da 비움", elem_id="btn-empty")
        bb = gr.Button("이전", elem_id="btn-back")
    outs = [img, info, idx]
    b1.click(lambda k: decide(k, "use"), idx, outs)
    b2.click(lambda k: decide(k, "drop"), idx, outs)
    b3.click(lambda k: decide(k, "empty"), idx, outs)
    bb.click(lambda k: view(max(0, k - 1)), idx, outs)
    demo.load(first, None, outs)

if __name__ == "__main__":
    demo.queue().launch(server_port=7863, inbrowser=True)
