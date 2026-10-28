#!/usr/bin/env python3
"""LimoPro 주행 프레임에 공개 da/ll 모델 7개를 돌려서 4x2 비교 GIF로 만든다.

TwinLiteNet-KMU-finetune의 scripts/montage/make_curve_driving_gif.py(old vs new 2칸)를
가져와서 8칸(원본 + 모델 7개)으로 확장한 버전. 오버레이 규칙은 원본과 동일하다:
파란색 블렌드 = 주행가능영역(da), 빨간색 = 차선(ll).

모델별 전처리는 각 레포의 공식 demo 스크립트를 그대로 따른다(우리 640x480 프레임은
letterbox(640, auto=True) 기준으로 패딩 없이 640x480 그대로 들어감).

  | 칸 | 모델               | 입력 색공간 / 정규화          | 출처 demo                     |
  |----|--------------------|-------------------------------|-------------------------------|
  | 1  | 원본               | -                             | -                             |
  | 2  | YOLOPv2            | RGB, /255                     | YOLOPv2/demo.py (TorchScript) |
  | 3  | TwinLiteNet+ Large | RGB, /255                     | TwinLiteNetPlus/demo.py       |
  | 4  | TwinLiteNet+ Medium| RGB, /255                     | TwinLiteNetPlus/demo.py       |
  | 5  | YOLOP              | BGR, ImageNet mean/std        | YOLOP/tools/demo.py           |
  | 6  | TwinLiteNet+ Small | RGB, /255                     | TwinLiteNetPlus/demo.py       |
  | 7  | HybridNets         | RGB, ImageNet mean/std        | HybridNets/hybridnets_test.py |
  | 8  | TwinLiteNet+ Nano  | RGB, /255                     | TwinLiteNetPlus/demo.py       |

모델 레포/가중치는 models/ 아래에 있어야 한다(.gitignore 대상, models/README.md 참고).

사용 예 (ZLUDA torch가 깔린 Windows python 기준):
  .venv/Scripts/python scripts/make_model_compare_gif.py \
      --frames D:/fine-tune/LimoPro_frames_20261023 --out outputs/montages/model_compare.gif
"""
import argparse
import glob
import importlib
import os
import subprocess
import sys
import time
from argparse import Namespace

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

PANEL_W, PANEL_H = 320, 240
COLS, ROWS = 4, 2
STRIDE = 32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ZLUDA(AMD에서 CUDA 에뮬레이션)는 cuDNN 경로가 CUDNN_STATUS_EXECUTION_FAILED로 깨져서 끈다.
# 순정 NVIDIA 환경에서는 켜둬도 무방.
if "ZLUDA" in (torch.cuda.get_device_name(0) if device.type == "cuda" else ""):
    torch.backends.cudnn.enabled = False
    # SAM2 어텐션(scaled_dot_product_attention)이 flash/mem-efficient(cutlass sm80+) 커널을 고르면
    # ZLUDA에서 "built for sm37"로 실패 -> 일반 math 경로만 사용
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# GPU 사용률 상한(기본 90%). 장시간 100% 부하로 PC가 꺼진 적이 있어서(Kernel-Power 41),
# GPU 작업 구간 시간 t마다 t*(1/duty - 1)만큼 쉬어 평균 점유율을 duty 이하로 맞춘다.
# 환경변수 GPU_DUTY=1.0 이면 제한 없음. 평균만 줄이고 순간 전력 스파이크는 못 막으니,
# 확실히 하려면 AMD Adrenalin > 성능 > 튜닝 > Power Limit도 같이 낮출 것.
GPU_DUTY = float(os.environ.get("GPU_DUTY", "0.9"))


class GpuDutyLimiter:
    def __init__(self, duty=GPU_DUTY):
        self.duty = min(max(duty, 0.05), 1.0)

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if device.type == "cuda":
            torch.cuda.synchronize()
        if self.duty < 1.0:
            time.sleep((time.perf_counter() - self.t0) * (1 / self.duty - 1))
        return False


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def import_from_repo(repo, module, attr):
    """레포마다 utils/model/lib 같은 같은 이름 모듈이 겹쳐서, import 후 sys.modules를 원복한다."""
    sys.path.insert(0, repo)
    try:
        obj = getattr(importlib.import_module(module), attr)
    finally:
        sys.path.remove(repo)
        # 레포 안 파일에서 온 모듈만 지운다(matplotlib 같은 외부 패키지는 재import 시 깨짐)
        prefix = os.path.abspath(repo) + os.sep
        for name, mod in list(sys.modules.items()):
            files = [getattr(mod, "__file__", None) or ""] + list(getattr(mod, "__path__", None) or [])
            if any(os.path.abspath(f).startswith(prefix) for f in files if f):
                del sys.modules[name]
    return obj


def pad_to_stride(img, stride=STRIDE):
    """letterbox(640, auto=True) 동작: 긴 변 640 기준 리사이즈 후 stride 배수로 패딩."""
    h0, w0 = img.shape[:2]
    r = 640 / max(h0, w0)
    if r != 1:
        img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    dh, dw = (-h) % stride, (-w) % stride
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left, h, w)


def to_tensor(img, rgb, imagenet_norm):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    if imagenet_norm:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)


def crop_resize(mask, crop, out_hw):
    top, left, h, w = crop
    mask = mask[top:top + h, left:left + w].astype(np.uint8)
    return cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def prob_resize(prob, crop, out_hw):
    """패딩 크롭 후 원본 해상도로 bilinear 리사이즈한 float32 확률맵."""
    top, left, h, w = crop
    prob = prob[top:top + h, left:left + w].astype(np.float32)
    return cv2.resize(prob, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)


def two_ch_prob(t):
    """[bg, fg] 2채널 출력 -> fg 확률. 채널별 sigmoid(YOLOP)/softmax 유사(YOLOPv2) 둘 다 비율로 정규화."""
    return (t[:, 1] / (t[:, 0] + t[:, 1] + 1e-6))[0].cpu().numpy()


def overlay_masks(img0, da_full, ll_full):
    overlay = img0.copy()
    overlay[da_full] = (0.5 * overlay[da_full].astype(np.float64) + 0.5 * np.array([255, 100, 0])).astype(np.uint8)
    overlay[ll_full] = (0, 0, 255)
    return overlay


def label_panel(img, text):
    panel = cv2.resize(img, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# 모델 로더: 각자 infer(img0_bgr) -> (da_bool, ll_bool) (원본 해상도) 를 돌려준다.
# infer.prep(img0) -> 입력 텐서, infer.model -> forward 대상 (scripts/benchmark_models.py가 씀)
# ---------------------------------------------------------------------------
def load_yolopv2():
    # TorchScript는 두 번째 호출부터 연산을 묶어 NVRTC로 새 커널을 컴파일(NNC fuser)하는데,
    # ZLUDA에서는 여기서 0xC0000005로 크래시한다 -> fuser/profiling executor 끔
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    model = torch.jit.load(os.path.join(MODELS_DIR, "YOLOPv2", "data", "weights", "yolopv2.pt"), map_location=device)
    model = torch.jit.freeze(model.eval())  # conv+bn 접기
    if not torch.backends.cudnn.enabled:
        # TorchScript 안의 batch_norm은 cudnn_enabled=True가 상수로 박혀 있어 전역 설정이 안 먹는다
        graph = model.graph
        false_const = graph.insertConstant(False)
        false_const.node().moveBefore(next(graph.nodes()))
        for node in graph.findAllNodes("aten::batch_norm"):
            node.replaceInput(8, false_const)

    def infer(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            _, seg, ll = model(to_tensor(img, rgb=True, imagenet_norm=False))
        da = torch.argmax(seg, 1)[0].cpu().numpy()
        ll = torch.round(ll)[0, 0].cpu().numpy()
        return crop_resize(da, crop, img0.shape[:2]), crop_resize(ll, crop, img0.shape[:2])

    def probs(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            _, seg, ll = model(to_tensor(img, rgb=True, imagenet_norm=False))
        return prob_resize(two_ch_prob(seg), crop, img0.shape[:2]), prob_resize(ll[0, 0].cpu().numpy(), crop, img0.shape[:2])

    # benchmark_models.py용: 전처리 끝난 입력 텐서 / 순수 forward
    infer.prep = lambda img0: to_tensor(pad_to_stride(img0)[0], rgb=True, imagenet_norm=False)
    infer.probs = probs  # (da_prob, ll_prob) float32, 원본 해상도 — make_ensemble_gif.py용
    infer.model = model
    return infer


def load_twinlitenetplus(config):
    repo = os.path.join(MODELS_DIR, "TwinLiteNetPlus")
    TwinLiteNetPlus = import_from_repo(repo, "model.model", "TwinLiteNetPlus")
    model = TwinLiteNetPlus(Namespace(config=config))
    model.load_state_dict(torch.load(os.path.join(repo, "pretrained", f"{config}.pth"), map_location=device))
    model = model.to(device).eval()

    def infer(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            out_da, out_ll = model(to_tensor(img, rgb=True, imagenet_norm=False))
        da = torch.argmax(out_da, 1)[0].cpu().numpy()
        ll = torch.argmax(out_ll, 1)[0].cpu().numpy()
        return crop_resize(da, crop, img0.shape[:2]), crop_resize(ll, crop, img0.shape[:2])

    def probs(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            out_da, out_ll = model(to_tensor(img, rgb=True, imagenet_norm=False))
        da = torch.softmax(out_da, 1)[0, 1].cpu().numpy()
        ll = torch.softmax(out_ll, 1)[0, 1].cpu().numpy()
        return prob_resize(da, crop, img0.shape[:2]), prob_resize(ll, crop, img0.shape[:2])

    # benchmark_models.py용: 전처리 끝난 입력 텐서 / 순수 forward
    infer.prep = lambda img0: to_tensor(pad_to_stride(img0)[0], rgb=True, imagenet_norm=False)
    infer.probs = probs  # (da_prob, ll_prob) float32, 원본 해상도 — make_ensemble_gif.py용
    infer.model = model
    return infer


def load_yolop():
    repo = os.path.join(MODELS_DIR, "YOLOP")
    get_net = import_from_repo(repo, "lib.models", "get_net")
    model = get_net(False)
    ckpt = torch.load(os.path.join(repo, "weights", "End-to-end.pth"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    def infer(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            _, da_out, ll_out = model(to_tensor(img, rgb=False, imagenet_norm=True))
        da = torch.argmax(da_out, 1)[0].cpu().numpy()
        ll = torch.argmax(ll_out, 1)[0].cpu().numpy()
        return crop_resize(da, crop, img0.shape[:2]), crop_resize(ll, crop, img0.shape[:2])

    def probs(img0):
        img, crop = pad_to_stride(img0)
        with torch.no_grad():
            _, da_out, ll_out = model(to_tensor(img, rgb=False, imagenet_norm=True))
        return prob_resize(two_ch_prob(da_out), crop, img0.shape[:2]), prob_resize(two_ch_prob(ll_out), crop, img0.shape[:2])

    # benchmark_models.py용: 전처리 끝난 입력 텐서 / 순수 forward
    infer.prep = lambda img0: to_tensor(pad_to_stride(img0)[0], rgb=False, imagenet_norm=True)
    infer.probs = probs  # (da_prob, ll_prob) float32, 원본 해상도 — make_ensemble_gif.py용
    infer.model = model
    return infer


def load_hybridnets():
    repo = os.path.join(MODELS_DIR, "HybridNets")
    HybridNetsBackbone = import_from_repo(repo, "backbone", "HybridNetsBackbone")
    # projects/bdd100k.yml 값 그대로
    model = HybridNetsBackbone(
        compound_coef=3,
        num_classes=1,
        ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
        scales=[2**0, 2**0.70, 2**1.32],
        seg_classes=2,
        backbone_name=None,
        seg_mode="multiclass",
    )
    model.load_state_dict(torch.load(os.path.join(repo, "weights", "hybridnets.pth"), map_location=device))
    model = model.to(device).eval()

    def infer(img0):
        # BiFPN이 P3~P7(1/128)까지 내려가서 128 배수가 필요. 공식 demo의 32 배수 패딩으론
        # 640x480이 P7에서 15 vs 16 크기 불일치로 터진다 -> 480을 512로 패딩.
        img, crop = pad_to_stride(img0, stride=128)
        with torch.no_grad():
            seg = model(to_tensor(img, rgb=True, imagenet_norm=True))[-1]
        cls = torch.argmax(seg, 1)[0].cpu().numpy()  # 0=bg, 1=road, 2=lane
        return crop_resize(cls == 1, crop, img0.shape[:2]), crop_resize(cls == 2, crop, img0.shape[:2])

    def probs(img0):
        img, crop = pad_to_stride(img0, stride=128)
        with torch.no_grad():
            p = torch.softmax(model(to_tensor(img, rgb=True, imagenet_norm=True))[-1], 1)[0].cpu().numpy()
        # multiclass라 차선 픽셀은 road가 아니라 lane으로만 찍힘 -> da는 다른 모델처럼 차선 포함(road+lane)
        return prob_resize(p[1] + p[2], crop, img0.shape[:2]), prob_resize(p[2], crop, img0.shape[:2])

    # benchmark_models.py용: 전처리 끝난 입력 텐서 / 순수 forward
    infer.prep = lambda img0: to_tensor(pad_to_stride(img0, stride=128)[0], rgb=True, imagenet_norm=True)
    infer.probs = probs  # (da_prob, ll_prob) float32, 원본 해상도 — make_ensemble_gif.py용
    infer.model = model
    return infer


# key -> (패널 라벨, 로더). README 표 순서대로 왼쪽 위부터 채움, 첫 칸은 원본
MODELS = {
    "yolopv2": ("YOLOPv2 (38.9M)", load_yolopv2),
    "tln_large": ("TwinLiteNet+ Large (1.94M)", lambda: load_twinlitenetplus("large")),
    "tln_medium": ("TwinLiteNet+ Medium (0.48M)", lambda: load_twinlitenetplus("medium")),
    "yolop": ("YOLOP (7.9M)", load_yolop),
    "tln_small": ("TwinLiteNet+ Small (0.12M)", lambda: load_twinlitenetplus("small")),
    "hybridnets": ("HybridNets (12.8M)", load_hybridnets),
    "tln_nano": ("TwinLiteNet+ Nano (0.03M)", lambda: load_twinlitenetplus("nano")),
}

# 앙상블 멤버: TwinLiteNet+ N/S/M/L은 같은 구조의 크기 차이라 Large만 넣는다
# (4개를 다 넣으면 TwinLiteNet+ 계열 혼자 7표 중 과반 4표를 차지함)
ENSEMBLE = ["yolopv2", "tln_large", "yolop", "hybridnets"]


def render_model_panels(key, paths, panel_dir):
    """모델 하나만 로드해서 프레임별 오버레이 패널을 panel_dir/key/ 에 PNG로 저장."""
    name, loader = MODELS[key]
    out_dir = os.path.join(panel_dir, key)
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in paths if not os.path.isfile(os.path.join(out_dir, os.path.basename(p)))]
    print(f"[{key}] device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}, "
          f"남은 프레임 {len(todo)}/{len(paths)}", flush=True)
    if not todo:
        return
    t0 = time.time()
    infer = loader()
    print(f"[{key}] 로드 {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    limiter = GpuDutyLimiter()
    for i, path in enumerate(todo):
        img0 = cv2.imread(path)
        with limiter:
            da, ll = infer(img0)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), label_panel(overlay_masks(img0, da, ll), name))
        if i % 50 == 0:
            print(f"[{key}] {i}/{len(todo)} ({time.time() - t0:.1f}s)", flush=True)
    print(f"[{key}] 완료 {len(todo)}프레임 {time.time() - t0:.1f}s", flush=True)


def run_workers(args, keys):
    """모델별 워커 프로세스를 동시에 띄운다. ZLUDA 첫 커널 컴파일이 CPU 단일 코어 작업이라
    프로세스를 나누면 컴파일이 병렬로 진행된다."""
    base = [sys.executable, "-u", os.path.abspath(__file__), "--frames", args.frames, "--panel-dir", args.panel_dir,
            "--start", str(args.start), "--step", str(args.step)] + (["--end", str(args.end)] if args.end else [])
    os.makedirs(args.panel_dir, exist_ok=True)
    pending, running, failed = list(keys), {}, []
    while pending or running:
        while pending and len(running) < args.jobs:
            key = pending.pop(0)
            log = open(os.path.join(args.panel_dir, f"{key}.log"), "w", encoding="utf-8")
            running[key] = (subprocess.Popen(base + ["--model", key], stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(2)
        for key, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[key]
                print(f"{key}: {'OK' if proc.returncode == 0 else f'실패(exit {proc.returncode})'}", flush=True)
                if proc.returncode != 0:
                    failed.append(key)
    return failed


def compose_gif(args, paths):
    frames = []
    for path in paths:
        fname = os.path.basename(path)
        panels = [label_panel(cv2.imread(path), f"Raw  {fname}")]
        panels += [cv2.imread(os.path.join(args.panel_dir, key, fname)) for key in MODELS]
        grid = np.vstack([np.hstack(panels[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        frames.append(Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"총 {len(frames)}프레임, GIF 저장 중...")
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True
    )
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"저장: {args.out} ({size_mb:.1f} MB, {len(frames)}프레임 @ {args.fps}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frame_NNNNNN.png 가 들어있는 폴더")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "montages", "model_compare.gif"))
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "outputs", "panels"),
                    help="모델별 패널 PNG 저장 위치(이미 있는 프레임은 건너뜀 -> 실패한 모델만 재실행 가능)")
    ap.add_argument("--start", type=int, default=0, help="정렬된 프레임 목록 기준 시작 인덱스")
    ap.add_argument("--end", type=int, default=None, help="끝 인덱스(포함 안 함)")
    ap.add_argument("--step", type=int, default=1, help="프레임 간격(GIF 용량 줄일 때)")
    ap.add_argument("--fps", type=float, default=10.0, help="LimoPro 카메라 기록 주기 10Hz")
    ap.add_argument("--jobs", type=int, default=1,
                    help="동시에 돌릴 모델 워커 수. 2 이상이면 워커끼리 GPU가 겹쳐 GPU_DUTY 상한이 보장 안 됨 "
                         "(첫 ZLUDA 컴파일만 빨라짐, 캐시된 뒤엔 1로 충분)")
    ap.add_argument("--model", choices=list(MODELS), help="(내부용) 이 모델 하나만 패널 렌더링")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")) + glob.glob(os.path.join(args.frames, "frame_*.jpg")))[args.start:args.end:args.step]
    if not paths:
        sys.exit(f"프레임 없음: {args.frames}")

    if args.model:
        render_model_panels(args.model, paths, args.panel_dir)
        return

    print(f"{len(paths)}프레임, 모델 {len(MODELS)}개를 {args.jobs}개씩 병렬 실행 (로그: {args.panel_dir}/<model>.log)")
    failed = run_workers(args, list(MODELS))
    if failed:
        sys.exit(f"실패한 모델: {failed} — 로그 확인 후 같은 명령 재실행하면 남은 프레임만 이어서 돎")
    compose_gif(args, paths)


if __name__ == "__main__":
    main()
