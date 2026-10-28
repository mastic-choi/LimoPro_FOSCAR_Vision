#!/usr/bin/env python3
"""tools/ego_label_review_{mac,radeon,nvidia}.ipynb 생성기.

세 노트북은 GPU 설정 셀만 다르고 나머지는 같다. 노트북을 고칠 땐 이 파일을 고친 뒤 다시 실행:
  python tools/build_review_notebooks.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


INTRO = """
# Ego lane pseudo-label 검수 + SAM 보정 ({title})

파인튜닝한 앙상블(`scripts/make_finetuned_pseudo_labels.py`)이 만든 pseudo-label을 사람이 한 장씩 보며 판정한다.
화면 = **[원본 | 오버레이]**, 파랑 = da(지금 달리는 차선), 빨강 = ll(차선).

| 키 | 동작 |
|---|---|
| **1** | 신뢰 — 그대로 채택하고 다음 |
| **2** | 보정 — SAM 클릭으로 da 또는 ll을 다시 만든다 |
| **3** | 버림 — 학습에 안 씀 (트랙 밖, 벽만 보임, 판단 불가 등) |

- **모델 간 의견이 갈리는 프레임부터** 나온다(`meta.csv`의 `da_agree` 오름차순). 앞쪽이 어려운 장면.
- 판정 즉시 `OUTPUT_DIR`에 저장(`images/ da_masks/ ll_masks/ review.csv`) → 끊겨도 다시 실행하면 이어서.
- 여러 명이 같은 링크(`SHARE = True`)로 들어와도 서로 다른 프레임이 배정된다(찜 + 10분 타임아웃).
- 라벨 관례: 저장되는 da = da ∪ ll (차선은 도로 위에 있다).

**라벨 기준 (팀에서 정한 대로 맞출 것)**
- da = 지금 달리는 차선만. 옆 차선/매트 전체를 칠하면 틀림.
- 커브에서 차선이 한쪽만 보일 때 어디까지 칠할지: **팀 결정 필요** (예: 보이는 차선 ~ 반대쪽 화면 끝 / 차선 폭만큼).
  결정되면 이 칸에 적고 모두 같은 기준으로.

이 노트북은 모델 4개를 돌리지 않는다(라벨은 이미 만들어져 있음). 필요한 건 **SAM2(`sam2.1_b.pt`) 하나**뿐.
"""

SETUP_MD = """
## 1. 환경 설정 ({title}) — 처음 한 번
레포 루트 또는 `tools/`에서 연다. 필요한 패키지가 없으면 설치한다.
gradio는 **5.x로 고정**(6.x는 huggingface-hub ≥ 1.16을 요구해서 transformers와 충돌함).
"""

SETUP_NVIDIA = r'''
import importlib.util, subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

need = [p for p, mod in [("gradio>=5.30,<6", "gradio"), ("ultralytics", "ultralytics"), ("opencv-python", "cv2")]
        if importlib.util.find_spec(mod) is None]
if need:
    pip(*need, "huggingface-hub>=0.34,<1.0")
if importlib.util.find_spec("torch") is None:
    # CUDA 버전은 nvidia-smi 로 확인하고 맞출 것 (예: CUDA 12.1 -> cu121, 12.4 -> cu124)
    pip("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121")

import torch
assert torch.cuda.is_available(), "CUDA 인식 안 됨 - NVIDIA 드라이버 / CUDA용 torch 설치 확인"
DEVICE = "cuda"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
'''

SETUP_MAC = r'''
import importlib.util, os, subprocess, sys

# MPS에 없는 연산은 CPU로 넘기기 (torch import 전에 설정해야 함)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

need = [p for p, mod in [("gradio>=5.30,<6", "gradio"), ("ultralytics", "ultralytics"),
                         ("opencv-python", "cv2"), ("torch", "torch")] if importlib.util.find_spec(mod) is None]
if need:
    pip(*need, "huggingface-hub>=0.34,<1.0")  # 맥용 기본 torch 휠에 MPS가 들어 있음

import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    print("[경고] MPS 사용 불가 -> CPU (Apple Silicon + macOS 12.3 이상인지, arm64 파이썬인지 확인). SAM이 느림")
print("torch", torch.__version__, "| device:", DEVICE)
'''

SETUP_RADEON = r'''
import importlib.util, subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

need = [p for p, mod in [("gradio>=5.30,<6", "gradio"), ("ultralytics", "ultralytics"), ("opencv-python", "cv2")]
        if importlib.util.find_spec(mod) is None]
if need:
    pip(*need, "huggingface-hub>=0.34,<1.0")

# torch는 자동 설치하지 않는다 (라데온은 환경마다 다름):
#   - Windows: 이 레포의 .venv (ZLUDA로 CUDA 흉내) 커널로 실행
#   - Linux  : ROCm용 torch  (pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 등)
import torch

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"  # ROCm torch도 cuda 이름을 그대로 씀
    name = torch.cuda.get_device_name(0)
    if "ZLUDA" in name:
        # ZLUDA: cuDNN 경로가 CUDNN_STATUS_EXECUTION_FAILED로 깨지고,
        # SAM2 어텐션이 flash/mem-efficient(sm80+) 커널을 고르면 "built for sm37"로 실패 -> 끔
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("ZLUDA 감지: cuDNN/flash SDPA 끔. 첫 SAM 실행은 커널 컴파일로 1~2분 걸릴 수 있음(이후 캐시)")
    print("torch", torch.__version__, "| GPU:", name, "| ROCm" if torch.version.hip else "")
else:
    print("[경고] GPU 인식 안 됨 -> CPU로 실행 (SAM이 느림). Windows면 .venv(ZLUDA) 커널인지 확인")
'''

PATHS_MD = """
## 2. 경로 — `LABEL_DIR`을 받은 pseudo-label 폴더로
`make_finetuned_pseudo_labels.py --out` 폴더 그대로(`images/ da_masks/ ll_masks/ meta.csv`). 다른 PC에서 할 땐 그 폴더를 압축해서 받아 풀면 된다.
"""

PATHS = r'''
import csv, os, threading, time
import cv2
import numpy as np

BASE = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "tools" else os.getcwd()  # 레포 루트

LABEL_DIR = os.path.join(BASE, "outputs", "pseudo_ft_v1")     # ↓ 받은 pseudo-label 폴더로 바꿀 것
OUTPUT_DIR = LABEL_DIR + "_review"                            # 검수 결과 (학습에 쓸 것)
SAM_CKPT = os.path.join(BASE, "models", "sam2", "sam2.1_b.pt")
if not os.path.isfile(SAM_CKPT):
    SAM_CKPT = "sam2.1_b.pt"  # 없으면 ultralytics가 처음 실행 때 자동 다운로드 (154MB)
SHARE = False  # True: 팀원과 같이 할 공개 링크 생성 (*.gradio.live, 링크를 아는 누구나 접속 가능)

for sub in ("images", "da_masks", "ll_masks"):
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "review.csv")

with open(os.path.join(LABEL_DIR, "meta.csv"), encoding="utf-8") as f:
    META = list(csv.DictReader(f))  # 이미 의견이 갈리는 순(da_agree 오름차순)으로 정렬돼 있음
FRAMES = [r["frame"] for r in META]
INFO = {r["frame"]: r for r in META}
print(f"대상 {len(FRAMES)}장 ({LABEL_DIR})")
'''

HELPERS_MD = """
## 3. SAM2 로드 + 그리기 함수
"""

HELPERS = r'''
from ultralytics import SAM

sam_model = SAM(SAM_CKPT)
print("SAM2 로드:", SAM_CKPT, "| device:", DEVICE)


def load(stem):
    img = cv2.imread(os.path.join(LABEL_DIR, "images", f"{stem}.png"))
    da = cv2.imread(os.path.join(LABEL_DIR, "da_masks", f"{stem}.png"), 0) > 0
    ll = cv2.imread(os.path.join(LABEL_DIR, "ll_masks", f"{stem}.png"), 0) > 0
    return img, da, ll


def overlay(img, da, ll, points=()):
    """파랑 = da, 빨강 = ll, 점: 초록 = 포함 / 주황 = 제외. RGB 반환."""
    v = img.copy()
    if da is not None:
        v[da] = (0.5 * v[da].astype(np.float32) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    if ll is not None:
        v[ll] = (0, 0, 255)
    for x, y, lbl in points:
        cv2.circle(v, (int(x), int(y)), 5, (0, 220, 0) if lbl == 1 else (0, 140, 255), -1)
        cv2.circle(v, (int(x), int(y)), 5, (255, 255, 255), 1)
    return cv2.cvtColor(v, cv2.COLOR_BGR2RGB)


def side_by_side(img, da, ll):
    return np.hstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), overlay(img, da, ll)])


def run_sam(img, points):
    """점 전부를 '한 물체'에 대한 프롬프트로 넘긴다.
    ultralytics.SAM은 (N,2)로 주면 점마다 다른 물체로 처리해서 포함/제외가 안 섞인다 -> (1,N,2)로 감쌈."""
    if not points:
        return None
    pts = [[[p[0], p[1]] for p in points]]
    lbls = [[p[2] for p in points]]
    r = sam_model(img, points=pts, labels=lbls, device=DEVICE, verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    m = r.masks.data[0].cpu().numpy().astype(bool)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m
'''

STATE_MD = """
## 4. 진행 상태 (이어하기 + 여러 명 동시 작업)
`review.csv`가 있으면 끝난 프레임은 건너뛴다. 누가 보고 있는 프레임은 판정되거나 10분이 지나기 전까지 다른 사람에게 안 나간다.
"""

STATE = r'''
decisions = {}
if os.path.isfile(RESULTS_CSV):
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            decisions[row["frame"]] = row["decision"]
print(f"전체 {len(FRAMES)} / 완료 {len(decisions)} / 남음 {len(FRAMES) - len(decisions)}")

_lock = threading.Lock()
claimed = {}
CLAIM_TIMEOUT_SEC = 600


def claim_next():
    with _lock:
        for s in FRAMES:
            if s in decisions:
                continue
            ts = claimed.get(s)
            if ts is None or time.time() - ts > CLAIM_TIMEOUT_SEC:
                claimed[s] = time.time()
                return s
        return None


def finalize(stem, decision, da=None, ll=None):
    """판정 1건 기록. trust/corrected는 즉시 OUTPUT_DIR에 저장(저장 da = da | ll)."""
    with _lock:
        decisions[stem] = decision
        new = not os.path.isfile(RESULTS_CSV)
        with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["frame", "decision", "source", "da_agree", "time"])
            w.writerow([stem, decision, INFO[stem]["source"], INFO[stem]["da_agree"], time.strftime("%Y-%m-%d %H:%M:%S")])
        claimed.pop(stem, None)
    if decision in ("trust", "corrected"):
        img = cv2.imread(os.path.join(LABEL_DIR, "images", f"{stem}.png"))
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", f"{stem}.png"), img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "da_masks", f"{stem}.png"), (da | ll).astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "ll_masks", f"{stem}.png"), ll.astype(np.uint8) * 255)


def status(stem):
    c = {k: sum(1 for v in decisions.values() if v == k) for k in ("trust", "corrected", "discard")}
    head = (f"완료 {len(decisions)}/{len(FRAMES)} (신뢰 {c['trust']} / 보정 {c['corrected']} / 버림 {c['discard']})"
            f" | 다른 작업자 {max(len(claimed) - 1, 0)}명")
    if stem is None:
        return "더 할 프레임 없음! " + head
    i = INFO[stem]
    return f"{stem}  [{i['source']}, 모델 일치도 {float(i['da_agree']):.2f}]  |  " + head
'''

APP_MD = """
## 5. 검수 앱 실행
브라우저 탭에서 연다(아래 출력의 로컬 주소 또는 공개 링크). Jupyter 출력창 안에서는 키보드 단축키가 안 먹는다.

**보정 모드**
1. `고칠 대상`: da(지금 차선) 또는 ll(차선) 선택
2. `클릭`: 포함 / 제외 고르고 이미지 클릭 → 점 찍기
3. `SAM 실행` → 결과 확인, 이상하면 점 추가/초기화 후 다시
4. `저장하고 다음` (da/ll 둘 다 지금 보이는 상태로 저장). `원래대로`는 pseudo-label로 되돌림, `ll 지우기`는 차선 없음
"""

APP = r'''
import gradio as gr

KEY_JS = """
() => {
  if (window._egoReviewKeys) { return; }
  window._egoReviewKeys = true;
  function find(root, id) {
    if (!root) { return null; }
    if (root.getElementById) { const f = root.getElementById(id); if (f) { return f; } }
    for (const el of (root.querySelectorAll ? root.querySelectorAll('*') : [])) {
      if (el.shadowRoot) { const f = find(el.shadowRoot, id); if (f) { return f; } }
    }
    return null;
  }
  function click(id) { const el = find(document, id); if (el) { (el.querySelector('button') || el).click(); } }
  document.addEventListener('keydown', function (e) {
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') { return; }
    if (e.key === '1') { click('trust_btn'); }
    else if (e.key === '2') { click('fix_btn'); }
    else if (e.key === '3') { click('drop_btn'); }
  });
}
"""

REVIEW, FIX = gr.update(visible=True), gr.update(visible=False)


def show_review(stem):
    if stem is None:
        return None
    img, da, ll = load(stem)
    return side_by_side(img, da, ll)


def next_frame():
    stem = claim_next()
    return stem, "review", [], None, None, REVIEW, FIX, show_review(stem), None, status(stem)


def on_load():
    return next_frame()


def trust(stem, mode):
    if mode != "review" or stem is None:
        return (gr.update(),) * 10
    _, da, ll = load(stem)
    finalize(stem, "trust", da, ll)
    return next_frame()


def drop(stem, mode):
    if stem is None:
        return (gr.update(),) * 10
    finalize(stem, "discard")
    return next_frame()


def enter_fix(stem, mode):
    if mode != "review" or stem is None:
        return (gr.update(),) * 10
    img, da, ll = load(stem)
    return (stem, "fix", [], da, ll, gr.update(visible=False), gr.update(visible=True), None,
            overlay(img, da, ll), "[보정] " + status(stem))


def on_click(stem, mode, points, da, ll, click_mode, evt: gr.SelectData):
    if mode != "fix" or stem is None:
        return points, gr.update()
    x, y = evt.index
    points = points + [(x, y, 1 if click_mode == "포함" else 0)]
    img, _, _ = load(stem)
    return points, overlay(img, da, ll, points)


def sam_apply(stem, points, da, ll, target):
    img, _, _ = load(stem)
    m = run_sam(img, points)
    if m is not None:
        if target.startswith("da"):
            da = m
        else:
            ll = m
    return da, ll, overlay(img, da, ll, points)


def reset_points(stem, da, ll):
    img, _, _ = load(stem)
    return [], overlay(img, da, ll)


def restore(stem):
    img, da, ll = load(stem)
    return [], da, ll, overlay(img, da, ll)


def clear_ll(stem, points, da):
    img, _, _ = load(stem)
    ll = np.zeros(img.shape[:2], bool)
    return ll, overlay(img, da, ll, points)


def save_fix(stem, mode, da, ll):
    if mode != "fix" or stem is None:
        return (gr.update(),) * 10
    finalize(stem, "corrected", da, ll)
    return next_frame()


with gr.Blocks(js=KEY_JS, title="ego lane 검수") as demo:
    gr.Markdown("### Ego lane 검수 — **1** 신뢰 / **2** 보정(SAM) / **3** 버림   (파랑 = da, 빨강 = ll)")
    info = gr.Textbox(label="진행", interactive=False)
    s_stem, s_mode, s_points, s_da, s_ll = gr.State(None), gr.State("review"), gr.State([]), gr.State(None), gr.State(None)

    with gr.Group(visible=True) as review_box:
        review_img = gr.Image(label="원본 | 오버레이", interactive=False)
        with gr.Row():
            drop_btn = gr.Button("✗ 버림 (3)", elem_id="drop_btn")
            fix_btn = gr.Button("✎ 보정 (2)", elem_id="fix_btn")
            trust_btn = gr.Button("✓ 신뢰 (1)", elem_id="trust_btn", variant="primary")

    with gr.Group(visible=False) as fix_box:
        with gr.Row():
            target = gr.Radio(["da (지금 차선)", "ll (차선)"], value="da (지금 차선)", label="고칠 대상")
            click_mode = gr.Radio(["포함", "제외"], value="포함", label="클릭")
        fix_img = gr.Image(label="클릭해서 점 찍기 (초록 = 포함, 주황 = 제외)", interactive=False)
        with gr.Row():
            sam_btn = gr.Button("SAM 실행", variant="secondary")
            reset_btn = gr.Button("점 초기화")
            restore_btn = gr.Button("원래대로")
            clear_ll_btn = gr.Button("ll 지우기")
        with gr.Row():
            drop2_btn = gr.Button("✗ 버림")
            save_btn = gr.Button("저장하고 다음 →", variant="primary")

    outs = [s_stem, s_mode, s_points, s_da, s_ll, review_box, fix_box, review_img, fix_img, info]
    demo.load(on_load, outputs=outs)
    trust_btn.click(trust, [s_stem, s_mode], outs)
    drop_btn.click(drop, [s_stem, s_mode], outs)
    drop2_btn.click(drop, [s_stem, s_mode], outs)
    fix_btn.click(enter_fix, [s_stem, s_mode], outs)
    fix_img.select(on_click, [s_stem, s_mode, s_points, s_da, s_ll, click_mode], [s_points, fix_img])
    sam_btn.click(sam_apply, [s_stem, s_points, s_da, s_ll, target], [s_da, s_ll, fix_img])
    reset_btn.click(reset_points, [s_stem, s_da, s_ll], [s_points, fix_img])
    restore_btn.click(restore, [s_stem], [s_points, s_da, s_ll, fix_img])
    clear_ll_btn.click(clear_ll, [s_stem, s_points, s_da], [s_ll, fix_img])
    save_btn.click(save_fix, [s_stem, s_mode, s_da, s_ll], outs)

demo.queue().launch(share=SHARE, inline=False, inbrowser=True)
'''

AFTER_MD = """
## 6. 끝나면
- 결과: `OUTPUT_DIR/images, da_masks, ll_masks` (신뢰 + 보정만), `review.csv` (버림 포함 전체 판정).
- 이 폴더를 압축해서 학습 담당에게 전달 → 학습 데이터에 추가 → 재학습.
- 앱 종료: 아래 셀 실행 (또는 커널 중지).
"""

CLOSE = "demo.close()\n"

PLATFORMS = {
    "mac": ("맥 Apple Silicon / MPS", SETUP_MAC),
    "radeon": ("라데온 GPU / Windows ZLUDA · Linux ROCm", SETUP_RADEON),
    "nvidia": ("NVIDIA GPU / CUDA", SETUP_NVIDIA),
}

for key, (title, setup) in PLATFORMS.items():
    nb = {
        "cells": [md(INTRO.format(title=title)), md(SETUP_MD.format(title=title)), code(setup),
                  md(PATHS_MD), code(PATHS), md(HELPERS_MD), code(HELPERS), md(STATE_MD), code(STATE),
                  md(APP_MD), code(APP), md(AFTER_MD), code(CLOSE)],
        "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(HERE, f"ego_label_review_{key}.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", path)
