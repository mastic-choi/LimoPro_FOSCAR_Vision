#!/usr/bin/env bash
# LimoPro 다중 작업(차선 분할 + 흰 가로선 + 표지판·신호등·콘 검출) H100 실행 스크립트.
#   bash run.sh            전부: 환경 설치 -> FSOCO 받기·변환 -> Klein 다듬기 -> 학습 -> 결과 zip
#   bash run.sh setup      환경만 (python venv, stable-diffusion.cpp CUDA 빌드, Klein 모델, FSOCO)
#   bash run.sh klein      Klein 다듬기만
#   bash run.sh train      학습만 (추가 인자는 train_multitask.py로 넘어감. 예: bash run.sh train --epochs 60)
#   bash run.sh pack       결과 zip만
# 끊겨도 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 합니다.
# 환경변수: KLEIN_SHARDS(기본 4, 동시에 돌릴 Klein 프로세스 수), BATCH(기본 32), WORKERS(기본 16),
#          SKIP_KLEIN=1(다듬기 생략), SKIP_FSOCO=1(FSOCO 생략, TraCon 콘만), TORCH_INDEX,
#          CUDA_ARCH(Klein 빌드 GPU 세대, 기본 90 = H100. A100 80, RTX 30xx 86, 40xx 89)
set -eo pipefail   # 학습이 실패하면(tee 앞 명령 실패 포함) 결과 묶기까지 가지 않고 멈춤
cd "$(dirname "$0")"
ROOT="$(pwd)"
STAGE="${1:-all}"; [ $# -gt 0 ] && shift
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
KLEIN_SHARDS="${KLEIN_SHARDS:-4}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-16}"
SD_COMMIT=9234ad822a376a4724638467736118ed61d68ee9      # Windows에서 Klein 결과를 확인한 버전
HF=https://huggingface.co/AtomicChat/FLUX-3-Image-Klein-GGUF/resolve/main
FSOCO_URL=http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip
PY="$ROOT/.venv/bin/python"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

dl() {  # dl URL OUT : 이어받기, 다 받은 뒤에만 이름을 바꿔서 반쯤 받은 파일을 완성본으로 착각하지 않음
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -L --fail --retry 10 --retry-delay 10 -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

setup_python() {
  [ -f .venv/.ready ] && return 0
  local P=""
  for c in python3.10 python3.11 python3.12 python3; do command -v "$c" >/dev/null 2>&1 && { P="$c"; break; }; done
  [ -n "$P" ] || { echo "python3이 없습니다: sudo apt install python3 python3-venv"; exit 1; }
  "$P" -m venv .venv || { echo "venv 실패: sudo apt install python3-venv"; exit 1; }
  "$PY" -m pip install -U pip
  "$PY" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
  "$PY" -m pip install -r requirements.txt
  touch .venv/.ready
}

SUDO=""
can_apt() {  # root이거나 sudo를 쓸 수 있으면 0 (비밀번호가 필요하면 터미널에서 한 번 물어봄)
  command -v apt-get >/dev/null 2>&1 || return 1
  [ "$(id -u)" = 0 ] && { SUDO=""; return 0; }
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true 2>/dev/null && { SUDO="sudo"; return 0; }
  [ -t 0 ] && sudo -v && { SUDO="sudo"; return 0; }
  return 1
}

nvcc_ok() {  # nvcc가 있고 버전이 11.8 이상(H100 sm_90 지원)이면 0
  local v; v="$("$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
  [ -n "$v" ] || return 1
  set -- $v
  [ "$1" -gt 11 ] || { [ "$1" -eq 11 ] && [ "$2" -ge 8 ]; }
}

find_nvcc() {
  for c in "$ROOT/tools/cuda_env/bin/nvcc" "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [ -n "$c" ] && [ -x "$c" ] && nvcc_ok "$c" && { echo "$c"; return 0; }
  done
  return 1
}

ensure_build_tools() {  # git, cmake, g++, nvcc(>=11.8) 준비. 성공하면 NVCC 변수 설정 후 0
  local need=""
  for t in git cmake g++; do command -v "$t" >/dev/null 2>&1 || need="$need $t"; done
  if [ -n "$need" ] && can_apt; then
    log "빌드 도구 설치(apt):$need"
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git cmake build-essential >> logs/apt.log 2>&1 || true
  fi
  NVCC="$(find_nvcc || true)"
  if [ -z "$NVCC" ] && ! command -v nvcc >/dev/null 2>&1 && can_apt; then   # nvcc가 이미 있는데 낮으면 apt도 같은 버전이라 건너뜀
    log "CUDA 툴킷 설치 시도(apt nvidia-cuda-toolkit)"
    $SUDO apt-get install -y -qq nvidia-cuda-toolkit >> logs/apt.log 2>&1 || true
    NVCC="$(find_nvcc || true)"
    [ -z "$NVCC" ] && log "apt의 CUDA가 없거나 11.8 미만(Ubuntu 22.04는 11.5) -> micromamba로 CUDA 12.4 설치"
  fi
  if [ -z "$NVCC" ] || ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    # 관리자 권한 없이 작업 폴더 안에 설치 (서버 환경은 안 건드림). g++/git은 서버에 없을 때만 같이 설치
    local had_gxx=0; command -v g++ >/dev/null 2>&1 && had_gxx=1
    if [ ! -x tools/cuda_env/bin/nvcc ]; then
      local pk="cuda-nvcc cuda-cudart-dev libcublas-dev cmake"
      [ "$had_gxx" = 1 ] || pk="$pk gxx_linux-64=12"
      command -v git >/dev/null 2>&1 || pk="$pk git"
      log "micromamba로 CUDA 12.4 컴파일러 등 설치: $pk (tools/cuda_env, 약 3GB)"
      mkdir -p tools/mamba
      [ -x tools/mamba/bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C tools/mamba bin/micromamba
      MAMBA_ROOT_PREFIX="$ROOT/tools/mamba" tools/mamba/bin/micromamba create -y -p "$ROOT/tools/cuda_env" \
        -c nvidia/label/cuda-12.4.1 -c conda-forge $pk >> logs/mamba.log 2>&1 \
        || { log "micromamba 설치 실패 (logs/mamba.log)"; return 1; }
    fi
    export PATH="$ROOT/tools/cuda_env/bin:$PATH" CUDAToolkit_ROOT="$ROOT/tools/cuda_env"
    if [ "$had_gxx" = 0 ]; then   # 서버에 g++가 없을 때만 conda 컴파일러 사용
      local gx; gx="$(ls "$ROOT"/tools/cuda_env/bin/*-g++ 2>/dev/null | head -1)"
      [ -n "$gx" ] && export CXX="$gx" CC="${gx%g++}gcc" CUDAHOSTCXX="$gx"
    fi
    NVCC="$(find_nvcc || true)"
  fi
  [ -n "$NVCC" ] && command -v cmake >/dev/null 2>&1 && command -v git >/dev/null 2>&1
}

setup_sd() {  # 성공하면 0, 빌드 도구를 못 갖추거나 빌드 실패면 1 (Klein만 건너뛰고 학습은 계속)
  [ -x tools/sd/build/bin/sd-cli ] && return 0
  ensure_build_tools || { log "빌드 도구(nvcc>=11.8, cmake, g++, git)를 준비하지 못해 Klein을 건너뜁니다 (logs/apt.log, logs/mamba.log)"; return 1; }
  log "nvcc: $NVCC ($("$NVCC" --version | tail -1))"
  log "stable-diffusion.cpp CUDA 빌드 (10~20분)"
  [ -d tools/sd/.git ] || git clone https://github.com/leejet/stable-diffusion.cpp tools/sd
  ( cd tools/sd && git fetch -q origin "$SD_COMMIT" 2>/dev/null || true; git checkout -q "$SD_COMMIT" && git submodule update --init --recursive -q \
    && cmake -S . -B build -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-90}" -DSD_CUDA_CUTLASS_PATH:PATH= \
    && cmake --build build --config Release -j "$(nproc)" ) > logs/sd_build.log 2>&1 \
    || { log "빌드 실패 -> Klein 건너뜀 (logs/sd_build.log)"; return 1; }
  [ -x tools/sd/build/bin/sd-cli ] || { log "sd-cli가 안 만들어짐 -> Klein 건너뜀 (logs/sd_build.log)"; return 1; }
}

setup_klein_models() {
  log "Klein 모델 받기 (약 9GB)"
  dl "$HF/FLUX-3-Image-Klein-AD-Q5_K.gguf" klein_models/diffusion_models/FLUX-3-Image-Klein-AD-Q5_K.gguf
  dl "$HF/text_encoders/Qwen3-VL-4B-Instruct-Q8_0.gguf" klein_models/text_encoders/Qwen3-VL-4B-Instruct-Q8_0.gguf
  dl "$HF/vae/ae.safetensors" klein_models/vae/flux3_ae.safetensors
}

setup_fsoco() {
  [ -f dataset/cones_v1/.fsoco_done ] && return 0
  [ "${SKIP_FSOCO:-0}" = 1 ] && { log "SKIP_FSOCO=1: FSOCO 생략"; return 0; }
  log "FSOCO 받기 (약 25GB)"
  dl "$FSOCO_URL" dataset/cones_raw/fsoco_dl/fsoco_bounding_boxes_train.zip
  if [ ! -f dataset/cones_raw/fsoco/.extracted ]; then
    log "FSOCO 압축 풀기"
    "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      dataset/cones_raw/fsoco_dl/fsoco_bounding_boxes_train.zip dataset/cones_raw/fsoco
    touch dataset/cones_raw/fsoco/.extracted
  fi
  log "FSOCO 변환 (워터마크 테두리 140px 잘라냄, 5종 콘 -> cone)"
  "$PY" scripts/prepare_cones.py --fsoco --fsoco-border 140 --workers "$(nproc)" | tee logs/prepare_cones.log
  touch dataset/cones_v1/.fsoco_done
}

run_klein() {
  [ "${SKIP_KLEIN:-0}" = 1 ] && { log "SKIP_KLEIN=1: Klein 생략"; return 0; }
  [ -f dataset/synth_signs_v2/train/.klein_done ] && return 0
  setup_sd || return 0
  setup_klein_models
  export KLEIN_SD_CLI="$ROOT/tools/sd/build/bin/sd-cli" KLEIN_MODEL_DIR="$ROOT/klein_models" KLEIN_OFFLOAD=0
  log "Klein 다듬기: 조건에 맞는 train 이미지 전부, 프로세스 $KLEIN_SHARDS개"
  local pids=()
  for i in $(seq 0 $((KLEIN_SHARDS - 1))); do
    "$PY" scripts/synth_klein_refine.py --data dataset/synth_signs_v2 --split train --n 0 --rest 0 \
      --nshard "$KLEIN_SHARDS" --shard "$i" > "logs/klein_s$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  grep -h "^finished" logs/klein_s*.log || true
  touch dataset/synth_signs_v2/train/.klein_done
}

run_train() {
  log "학습 시작 (로그: outputs/train/hybridnets_mt_v1_seed0/train.log)"
  mkdir -p outputs/train/hybridnets_mt_v1_seed0
  "$PY" scripts/train_multitask.py --seg-data dataset/synth_signs_v2 --det-data dataset/cones_v1 \
    --venue-ref-dir dataset/venue_ref_20261106 --init weights/hybridnets_v4_best_da.pt \
    --amp --batch "$BATCH" --workers "$WORKERS" --out outputs/train/hybridnets_mt_v1_seed0 "$@" \
    2>&1 | tee -a outputs/train/hybridnets_mt_v1_seed0/train.log
}

pack() {
  local OUT="results_$(hostname).zip"
  log "결과 묶기 -> $OUT"
  "$PY" - "$OUT" <<'EOF'
import csv, glob, os, sys, zipfile
out = sys.argv[1]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "last.pt", "log.csv", "test.json", "train.log"):
        p = os.path.join("outputs/train/hybridnets_mt_v1_seed0", f)
        if os.path.exists(p) and not (f == "last.pt" and os.path.getsize(p) > 3e8):
            z.write(p)
    base = "dataset/synth_signs_v2/train"
    for f in glob.glob(base + "/klein_refine*.csv") + glob.glob("logs/*.log"):
        z.write(f)
    kept = set()
    for f in glob.glob(base + "/klein_refine*.csv"):
        kept |= {r["image"] for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] == "kept"}
    for n in sorted(kept):   # Klein으로 다듬어 통과한 이미지만 (원본 PC에서 덮어쓰면 됨)
        p = os.path.join(base, "images", n + ".png")
        if os.path.exists(p):
            z.write(p)
print(out, round(os.path.getsize(out) / 1e9, 2), "GB")
EOF
}

# 절전 방지(권한 없으면 무시)
if [ -z "$LIMO_INHIBITED" ] && command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle --who=LimoPro --why=check true >/dev/null 2>&1; then
  export LIMO_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who=LimoPro --why=training bash "$0" "$STAGE" "$@"
fi

case "$STAGE" in
  setup) setup_python; setup_fsoco; setup_sd && setup_klein_models || true ;;
  klein) setup_python; run_klein ;;
  train) setup_python; run_train "$@" ;;
  pack)  setup_python; pack ;;
  all)   # 빌드 도구 설치(sudo 비밀번호를 물을 수 있음)와 빌드를 맨 앞에: 사람이 터미널 앞에 있을 때
         setup_python
         if [ "${SKIP_KLEIN:-0}" != 1 ] && setup_sd; then setup_klein_models; fi
         setup_fsoco; run_klein; run_train "$@"; pack ;;
  *) echo "사용: bash run.sh [all|setup|klein|train|pack]"; exit 1 ;;
esac
log "끝 ($STAGE)"
