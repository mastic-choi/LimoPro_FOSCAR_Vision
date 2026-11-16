#!/usr/bin/env python3
"""bag_mt_v1 pseudo-label 을 사람이 검수하는 웹 앱 (Vision 레포 scripts/review_round4.py 를 바탕으로, 박스 판정 추가).

  make  <dataset> : review/view/<name>.jpg ([원본 | 라벨]) 와 review/items.json 을 만든다.
                    순서: test_bag1116 먼저(시험 세트를 공정하게), 그다음 train. bag 안에서는 시간순.
  serve <dataset> : 웹 앱 (http://127.0.0.1:7863). 판정은 즉시 review/decisions.csv 에 저장, 다시 켜면 이어서.
  apply <dataset> : decisions.csv 를 데이터셋에 반영 (원본 라벨은 review/backup/ 에 한 번 복사해 둠)
                    2 버림 → excluded/ 로 옮김 (meta 의 excluded 에 human_drop)
                    3 da·ll 비움 → da_masks·ll_masks 를 0 으로 (트랙 밖, 내 차로 없음)
                    4 박스 무시 → 그 프레임 박스를 전부 무시 박스(-2)로 (da·ll 은 그대로 씀)
                    5 사용 + 정지선 빼기 → da·ll·박스는 그대로, sl_masks·sl_valid 만 지움 (정지선 손실 제외)

키: 1 사용 · 2 버림 · 3 da·ll 비움 · 4 박스 무시 · 5 사용 + 정지선 빼기 · Backspace 이전
  정지선 라벨이 없는 프레임(stop line: no label)은 학습에서 정지선 손실을 안 쓰므로, 정지선이 보여도 1 사용.
  정지선 라벨이 있는데 틀렸거나 "none (reviewed)" 인데 정지선이 보이면 5.

정지선은 따로 검수한다 (모든 데이터셋, sl_candidates.py 다음):
  sl-make  <sl_review> : 후보마다 검수 이미지
  sl-serve <sl_review> : 웹 앱 (http://127.0.0.1:7864), 판정은 <sl_review>/sl_decisions.csv
  (반영은 판정이 끝난 뒤 sl_apply.py 로: 검수한 프레임에만 sl_masks·sl_valid 를 쓴다)
화면 색 (review_round4 와 같음): 초록 = da, 노란 테두리 = da 경계, 자홍 = 차선(ll), 주황 = 정지선(sl, 정지선 검수에서 반영된 것). 박스: 파랑 sign_B, 빨강 sign_A, 노랑 신호등, 회색 = 무시 박스
"""
import csv, json, os, shutil, sys, time

import cv2
import numpy as np

CLASSES = ["pedestrian", "sign_A", "sign_B", "bump", "slow", "parking", "traffic_light",
           "light_red", "light_yellow", "light_green", "limo", "cone"]
BOXCOL = {"sign_A": (0, 0, 255), "sign_B": (255, 128, 0), "traffic_light": (0, 255, 255), "light_red": (0, 0, 255),
          "light_yellow": (0, 200, 255), "light_green": (0, 255, 0)}
CHOICE = {"use": "사용", "drop": "버림", "empty": "da·ll 비움", "nobox": "박스 무시", "use_nosl": "사용 + 정지선 빼기"}
HELP = ("da(초록)가 **지금 달리는 차로**를, ll(자홍)이 **차선**을 맞게 칠했고 박스도 맞으면 **1 사용**.  "
        "da·ll 이 크게 틀렸거나(장애물·사람 위로 칠함, 옆 차로까지 칠함, 차선 빠짐) 박스가 빠진 물체가 있으면 **2 버림**.  "
        "트랙 밖이라 내 차로가 없으면 **3 da·ll 비움**.  da·ll 은 맞는데 박스만 틀렸으면(엉뚱한 물체에 박스) **4 박스 무시**.  \n"
        "정지선(주황): 라벨이 없으면(`no label`) 정지선이 보여도 상관없음 → 1.  "
        "주황이 틀렸거나 `none (reviewed)` 인데 정지선이 보이면 **5 사용 + 정지선 빼기**.")


def read_boxes(p):
    out = []
    for line in open(p):
        c, cx, cy, w, h = line.split(); c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
        out.append((c, int((cx - w / 2) * 640), int((cy - h / 2) * 360), int((cx + w / 2) * 640), int((cy + h / 2) * 360)))
    return out


def render(root, m):
    d = os.path.join(root, m["split"])
    img = cv2.imread(os.path.join(d, "images", m["name"] + ".png"))
    da = cv2.imread(os.path.join(d, "da_masks", m["name"] + ".png"), 0) > 0
    ll = cv2.imread(os.path.join(d, "ll_masks", m["name"] + ".png"), 0) > 0
    o = img.copy()
    o[da] = (0.55 * o[da] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(da.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, (0, 255, 255), 1)
    o[ll] = (255, 0, 255)
    slp = os.path.join(d, "sl_masks", m["name"] + ".png")   # 정지선: 따로 검수해 반영한 것만 있음
    if os.path.exists(slp):
        sl = cv2.imread(slp, 0) > 0
        o[sl] = (0, 140, 255)
        if f"{m['split']}/{m['name']}" in PSEUDO_SL:
            tag = "stop line: pseudo" + ("" if sl.any() else " (none found)")
        else:
            tag = "stop line: reviewed" if sl.any() else "stop line: none (reviewed)"
    else:
        tag = "stop line: no label"
    cv2.rectangle(o, (0, 340), (8 + 9 * len(tag), 360), (0, 0, 0), -1)
    cv2.putText(o, tag, (4, 355), 0, 0.45, (0, 140, 255) if os.path.exists(slp) else (180, 180, 180), 1)
    for c, x1, y1, x2, y2 in read_boxes(os.path.join(d, "labels", m["name"] + ".txt")):
        if c == -2:
            cv2.rectangle(o, (x1, y1), (x2, y2), (170, 170, 170), 1); cv2.putText(o, "ign", (x1, max(10, y1 - 3)), 0, 0.4, (170, 170, 170), 1)
        else:
            col = BOXCOL.get(CLASSES[c], (255, 255, 255))
            cv2.rectangle(o, (x1, y1), (x2, y2), col, 2); cv2.putText(o, CLASSES[c], (x1, max(12, y1 - 3)), 0, 0.5, col, 2)
    return np.hstack([img, np.full((360, 6, 3), 255, np.uint8), o])


PSEUDO_SL = set()


def make(root):
    global PSEUDO_SL
    ps = os.path.join(root, "sl_pseudo_unreviewed.txt")
    PSEUDO_SL = set(open(ps).read().split()) if os.path.exists(ps) else set()
    R = os.path.join(root, "review"); os.makedirs(os.path.join(R, "view"), exist_ok=True)
    meta = [json.loads(l) for l in open(os.path.join(root, "pseudo_meta.jsonl"))]
    order = {"test_bag1116": 0, "train": 1}
    meta = sorted([m for m in meta if m["split"] in order], key=lambda m: (order[m["split"]], m["bag"], m["t"]))
    items = []
    for i, m in enumerate(meta):
        cv2.imwrite(os.path.join(R, "view", m["name"] + ".jpg"), render(root, m), [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append({"id": m["name"], "split": m["split"], "bag": m["bag"], "t": m["t"], "mission": m["mission"],
                      "flags": m["flags"], "labels": m["labels"]})
        if (i + 1) % 200 == 0: print(f"  {i + 1}/{len(meta)}", flush=True)
    json.dump(items, open(os.path.join(R, "items.json"), "w"), ensure_ascii=False)
    print(f"검수 이미지 {len(items)}장 → {R}")


def serve(root):
    import gradio as gr
    R = os.path.join(root, "review")
    ITEMS = json.load(open(os.path.join(R, "items.json")))
    OUT = os.path.join(R, "decisions.csv")

    def load_done():
        if not os.path.exists(OUT): return {}
        return {r["id"]: r["choice"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}

    def save(done):
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "split", "choice", "time"])
            for it in ITEMS:
                if it["id"] in done:
                    w.writerow([it["id"], it["split"], done[it["id"]], time.strftime("%Y-%m-%d %H:%M:%S")])
        os.replace(tmp, OUT)

    def view(k):
        done = load_done()
        if k >= len(ITEMS):
            return None, f"### 끝났습니다. {len(done)}/{len(ITEMS)}장 판정. 결과: {OUT}\n\n반영: `python tools/review_bag_mt.py apply {root}`", k
        it = ITEMS[k]; prev = done.get(it["id"])
        n_sp = sum(1 for x in ITEMS if x["split"] == it["split"]); i_sp = sum(1 for x in ITEMS[:k + 1] if x["split"] == it["split"])
        boxes = ", ".join(c for c, _ in it["labels"] if c != "ignore") or "없음"
        info = (f"### {k + 1} / {len(ITEMS)}  ·  [{it['split']} {i_sp}/{n_sp}]  ·  {it['bag']} {it['t']:.1f}초 (미션 {it['mission']})"
                f"{'  ·  이전 판정: ' + CHOICE[prev] if prev else ''}  ·  완료 {len(done)}장\n\n"
                f"정답 박스: {boxes}" + (f"  ·  규칙 표시: {' '.join(it['flags'])}" if it["flags"] else "") + f"\n\n{HELP}")
        img = cv2.cvtColor(cv2.imread(os.path.join(R, "view", it["id"] + ".jpg")), cv2.COLOR_BGR2RGB)
        return img, info, k

    def decide(k, c):
        if k < len(ITEMS):
            done = load_done(); done[ITEMS[k]["id"]] = c; save(done)
        return view(min(k + 1, len(ITEMS)))

    def first():
        done = load_done()
        return view(next((k for k, it in enumerate(ITEMS) if it["id"] not in done), len(ITEMS)))

    KEY_JS = """
    () => {
      document.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const m = {'1': 'btn-use', '2': 'btn-drop', '3': 'btn-empty', '4': 'btn-nobox', '5': 'btn-nosl', 'Backspace': 'btn-back'};
        const id = m[e.key];
        if (id) { e.preventDefault(); document.getElementById(id)?.click(); }
      });
    }
    """
    with gr.Blocks(title="bag_mt_v1 검수", js=KEY_JS) as demo:
        gr.Markdown("**1** 사용 · **2** 버림 · **3** da·ll 비움 · **4** 박스 무시 · **5** 사용 + 정지선 빼기 · **Backspace** 이전  —  "
                    "왼쪽 원본, 오른쪽 라벨: 초록 = da, 노란 테두리 = da 경계, 자홍 = 차선(ll), 주황 = 정지선(따로 검수해 반영함), 회색 ign = 무시 박스")
        info = gr.Markdown()
        img = gr.Image(show_label=False, height=440)
        idx = gr.State(0)
        with gr.Row():
            b1 = gr.Button("1 사용", elem_id="btn-use", variant="primary")
            b2 = gr.Button("2 버림", elem_id="btn-drop")
            b3 = gr.Button("3 da·ll 비움", elem_id="btn-empty")
            b4 = gr.Button("4 박스 무시", elem_id="btn-nobox")
            b5 = gr.Button("5 사용 + 정지선 빼기", elem_id="btn-nosl")
            bb = gr.Button("이전", elem_id="btn-back")
        outs = [img, info, idx]
        b5.click(lambda k: decide(k, "use_nosl"), idx, outs)
        b1.click(lambda k: decide(k, "use"), idx, outs)
        b2.click(lambda k: decide(k, "drop"), idx, outs)
        b3.click(lambda k: decide(k, "empty"), idx, outs)
        b4.click(lambda k: decide(k, "nobox"), idx, outs)
        bb.click(lambda k: view(max(0, k - 1)), idx, outs)
        demo.load(first, None, outs)
    demo.queue().launch(server_name="127.0.0.1", server_port=7863, inbrowser=True)


def apply(root):
    R = os.path.join(root, "review")
    dec = {r["id"]: r for r in csv.DictReader(open(os.path.join(R, "decisions.csv"), encoding="utf-8"))}
    mp = os.path.join(root, "pseudo_meta.jsonl")
    meta = [json.loads(l) for l in open(mp)]
    bk = os.path.join(R, "backup")
    if not os.path.isdir(bk):   # 처음 반영할 때만 원본 라벨·마스크·meta 를 복사
        for m in meta:
            if m["name"] in dec and dec[m["name"]]["choice"] != "use":
                for sub, ext in (("labels", ".txt"), ("da_masks", ".png"), ("ll_masks", ".png")):
                    os.makedirs(os.path.join(bk, m["split"], sub), exist_ok=True)
                    shutil.copy2(os.path.join(root, m["split"], sub, m["name"] + ext), os.path.join(bk, m["split"], sub))
        shutil.copy2(mp, os.path.join(bk, "pseudo_meta.jsonl"))
    n = {"use": 0, "drop": 0, "empty": 0, "nobox": 0, "use_nosl": 0}
    for m in meta:
        d = dec.get(m["name"])
        if not d or m["split"] == "excluded": continue
        c = d["choice"]; n[c] += 1; m["review"] = c
        src = os.path.join(root, m["split"])
        if c == "empty":
            z = np.zeros((360, 640), np.uint8)
            cv2.imwrite(os.path.join(src, "da_masks", m["name"] + ".png"), z); cv2.imwrite(os.path.join(src, "ll_masks", m["name"] + ".png"), z)
        elif c == "nobox":
            p = os.path.join(src, "labels", m["name"] + ".txt")
            lines = [l.split() for l in open(p) if l.strip()]
            open(p, "w").write("".join(f"-2 {' '.join(x[1:])}\n" for x in lines))
        elif c == "use_nosl":
            for sub in ("sl_masks", "sl_valid"):
                p = os.path.join(src, sub, m["name"] + ".png")
                if os.path.exists(p):
                    os.makedirs(os.path.join(bk, m["split"], sub), exist_ok=True)
                    if not os.path.exists(os.path.join(bk, m["split"], sub, m["name"] + ".png")):
                        shutil.copy2(p, os.path.join(bk, m["split"], sub))
                    os.remove(p)
        elif c == "drop":
            for sub, ext in (("images", ".png"), ("labels", ".txt"), ("da_masks", ".png"), ("ll_masks", ".png"), ("sl_masks", ".png"), ("sl_valid", ".png")):
                if not os.path.exists(os.path.join(src, sub, m["name"] + ext)): continue
                os.makedirs(os.path.join(root, "excluded", sub), exist_ok=True)
                shutil.move(os.path.join(src, sub, m["name"] + ext), os.path.join(root, "excluded", sub, m["name"] + ext))
            m["excluded"] = m.get("excluded", []) + ["human_drop"]; m["split"] = "excluded"
    with open(mp, "w") as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("반영:", n, "| 남은 장수:", {s: len(os.listdir(os.path.join(root, s, "images"))) for s in ("train", "test_bag1116", "excluded")})


# ---------------------------------------------------------------------------
# 정지선(sl) 검수: sl_candidates.py 의 <sl_review>/candidates.json 을 본다.
#   1 모델 맞음  : 주황(모델 확률 > 0.5)을 정지선 라벨로. 믿을 픽셀 = 확률 < 0.3 또는 > 0.7
#   2 정지선 없음: 모델이 잡은 건 오탐 → sl 전부 0, 이미지 전체를 믿을 픽셀로
#   3 애매·틀림  : 이 프레임은 sl 라벨 없음 (학습에서 sl 손실 제외)
#   4 기존 맞음  : 기존 라벨(하늘색 테두리, 11/06 도면 투영)을 그대로
SL_CHOICE = {"model": "모델 맞음", "none": "정지선 없음", "skip": "애매·틀림(안 씀)", "exist": "기존 맞음"}
SL_HELP = ("주황 = 모델이 잡은 정지선(확률 > 0.5), 하늘색 테두리 = 기존 라벨(있을 때만).  "
           "주황이 **정지선·미션 시작선만 빠짐없이** 칠했으면 **1 모델 맞음**.  정지선이 아닌 것(횡단보도 줄무늬, 글자, 반사)만 칠했으면 **2 정지선 없음**.  "
           "정지선을 일부만 칠했거나 다른 것과 섞였으면 **3 애매·틀림**.  하늘색(기존)이 더 맞으면 **4 기존 맞음**.")


def sl_render(it):
    root, sp, stem = it["root"], it["split"], it["rep"]
    img = cv2.imread(os.path.join(root, sp, "images", stem + ".png"))
    p = cv2.imread(os.path.join(SLR, it["ds"], sp, "sl_prob", stem + ".png"), 0) / 255.0
    o = img.copy()
    pos, unc = p > 0.5, (p > 0.3) & (p <= 0.5)
    o[unc] = (0.75 * o[unc] + 0.25 * np.array([0, 140, 255])).astype(np.uint8)
    o[pos] = (0.4 * o[pos] + 0.6 * np.array([0, 140, 255])).astype(np.uint8)   # 반투명: 밑의 흰 페인트가 보이게
    exm = os.path.join(root, sp, "sl_masks", stem + ".png")
    if os.path.exists(exm):
        cs, _ = cv2.findContours((cv2.imread(exm, 0) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (255, 255, 0), 1)
    top = np.hstack([img, np.full((360, 6, 3), 255, np.uint8), o])
    bot = cv2.resize(o[120:], (1280, 480), interpolation=cv2.INTER_NEAREST)   # 라벨의 아래 2/3 를 2배로 (정지선은 화면 아래쪽)
    bot = np.hstack([bot, np.full((480, 6, 3), 255, np.uint8)])
    return np.vstack([top, np.full((6, top.shape[1], 3), 255, np.uint8), bot])


SLR = ""


def sl_make(slr):
    global SLR; SLR = slr
    items = json.load(open(os.path.join(slr, "candidates.json")))
    os.makedirs(os.path.join(slr, "view"), exist_ok=True)
    for i, it in enumerate(items):
        cv2.imwrite(os.path.join(slr, "view", it["id"] + ".jpg"), sl_render(it), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if (i + 1) % 300 == 0: print(f"  {i + 1}/{len(items)}", flush=True)
    print(f"정지선 검수 이미지 {len(items)}장 → {slr}/view")


def sl_serve(slr):
    import gradio as gr
    ITEMS = json.load(open(os.path.join(slr, "candidates.json")))
    OUT = os.path.join(slr, "sl_decisions.csv")

    def load_done():
        if not os.path.exists(OUT): return {}
        return {r["id"]: r["choice"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}

    def save(done):
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "ds", "split", "rep", "n_members", "choice", "time"])
            for it in ITEMS:
                if it["id"] in done:
                    w.writerow([it["id"], it["ds"], it["split"], it["rep"], len(it["members"]), done[it["id"]], time.strftime("%Y-%m-%d %H:%M:%S")])
        os.replace(tmp, OUT)

    def view(k):
        done = load_done()
        if k >= len(ITEMS):
            return None, f"### 끝났습니다. {len(done)}/{len(ITEMS)}개 판정. 결과: {OUT}", k
        it = ITEMS[k]; prev = done.get(it["id"])
        grp = f"  ·  같은 배경 {len(it['members'])}장에 같이 적용" if len(it["members"]) > 1 else ""
        ex = "  ·  기존 라벨 있음(하늘색)" if it["has_ex"] else ""
        info = (f"### {k + 1} / {len(ITEMS)}  ·  [{it['ds']} / {it['split']}]  ·  {it['rep']}{grp}{ex}"
                f"{'  ·  이전 판정: ' + SL_CHOICE[prev] if prev else ''}  ·  완료 {len(done)}개\n\n{SL_HELP}")
        img = cv2.cvtColor(cv2.imread(os.path.join(slr, "view", it["id"] + ".jpg")), cv2.COLOR_BGR2RGB)
        return img, info, k

    def decide(k, c):
        if k < len(ITEMS):
            if c == "exist" and not ITEMS[k]["has_ex"]:
                img, info, _ = view(k)
                return img, "### ⚠ 이 프레임은 기존 라벨이 없습니다. 1 / 2 / 3 중에서 골라 주세요\n\n" + info, k
            done = load_done(); done[ITEMS[k]["id"]] = c; save(done)
        return view(min(k + 1, len(ITEMS)))

    def first():
        done = load_done()
        return view(next((k for k, it in enumerate(ITEMS) if it["id"] not in done), len(ITEMS)))

    KEY_JS = """
    () => {
      document.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const m = {'1': 'btn-model', '2': 'btn-none', '3': 'btn-skip', '4': 'btn-exist', 'Backspace': 'btn-back'};
        const id = m[e.key];
        if (id) { e.preventDefault(); document.getElementById(id)?.click(); }
      });
    }
    """
    with gr.Blocks(title="정지선 검수", js=KEY_JS) as demo:
        gr.Markdown("**1** 모델 맞음 · **2** 정지선 없음 · **3** 애매·틀림(안 씀) · **4** 기존 맞음 · **Backspace** 이전  —  "
                    "위: 원본 | 라벨,  아래: 라벨의 화면 아래쪽 2배.  주황 = 모델 정지선, 옅은 주황 = 애매(0.3~0.5), 하늘색 테두리 = 기존 라벨")
        info = gr.Markdown()
        img = gr.Image(show_label=False, height=680)
        idx = gr.State(0)
        with gr.Row():
            b1 = gr.Button("1 모델 맞음", elem_id="btn-model", variant="primary")
            b2 = gr.Button("2 정지선 없음", elem_id="btn-none")
            b3 = gr.Button("3 애매·틀림", elem_id="btn-skip")
            b4 = gr.Button("4 기존 맞음", elem_id="btn-exist")
            bb = gr.Button("이전", elem_id="btn-back")
        outs = [img, info, idx]
        b1.click(lambda k: decide(k, "model"), idx, outs)
        b2.click(lambda k: decide(k, "none"), idx, outs)
        b3.click(lambda k: decide(k, "skip"), idx, outs)
        b4.click(lambda k: decide(k, "exist"), idx, outs)
        bb.click(lambda k: view(max(0, k - 1)), idx, outs)
        demo.load(first, None, outs)
    demo.queue().launch(server_name="127.0.0.1", server_port=7864, inbrowser=True)


if __name__ == "__main__":
    cmd, root = sys.argv[1], os.path.abspath(sys.argv[2])
    if cmd.startswith("sl-"):
        SLR = root
        {"sl-make": sl_make, "sl-serve": sl_serve}[cmd](root)
    else:
        {"make": make, "serve": serve, "apply": apply}[cmd](root)
