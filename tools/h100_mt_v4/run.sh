#!/usr/bin/env bash
# LimoPro 다중 작업 v4 (hybridnets_mt_v4) — 원본 HybridNets(BDD100K)에서 새로 학습. H100 서버용.
#   bash run.sh            전부: 확인 -> 학습 -> 결과 zip
#   bash run.sh check      데이터·가중치·GPU 확인만 (데이터 zip 이 옆에 있으면 여기서 풂)
#   bash run.sh train      학습만 (추가 인자는 train_multitask.py 로)
#   bash run.sh pack       결과 zip 만
# 끊겨도 같은 명령을 다시 실행하면 last.pt 부터 이어서. GPU 메모리가 모자라 죽으면 batch 를 절반으로 줄여 이어서.
#
# H100 에서 느려지지 않게 (README_H100.md "속도"):
#   - GPU_DUTY=1.0 (점유율 제한·매 step 동기화 없음), WORKERS=0 (자동: CPU 코어 수 - 2), PREFETCH=4
#   - CHANNELS_LAST=1, TF32·cudnn.benchmark·fused AdamW (코드에서 켬), COMPILE=1 이면 torch.compile
#   - 대회장 조명 증강의 큰 블러를 저해상도에서 계산 (VENUE_FAST=1)
#   - epoch 마다 "학습 N img/s, 데이터 대기 X%" 출력. 데이터 대기가 30% 를 넘으면 CPU 가 병목 → WORKERS 를 늘리거나 코어 많은 노드로
# 환경변수 (괄호 안이 기본값)
#   EPOCHS(150) / LR(1e-4, 검출 머리·BiFPN 은 x10) / WARMUP(500 step) / ES_PATIENCE(8) val loss 조기 종료 /
#   BATCH(32) / WORKERS(0 = 자동) / PREFETCH(4) / CHANNELS_LAST(1) / COMPILE(0) / REAL_TL_REPEAT(2) / REAL_FORK_REPEAT(2) /
#   MONITOR_P(0.4) DOTS_P(0.5) MONITOR_REAL(all) 신호등 뒤 방해 모니터 / INIT(weights/hybridnets.pth) / PYTHON /
#   DATA_ZIP(자동: ./ 또는 ../LimoPro_dataset_mt_v2.zip, 나눈 파일 .zip.part* 이면 이어 붙임)
set -eo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
STAGE="${1:-all}"; [ $# -gt 0 ] && shift
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
BATCH="${BATCH:-32}"
export GPU_DUTY="${GPU_DUTY:-1.0}"
export VENUE_FAST="${VENUE_FAST:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset PYTHONPATH LD_LIBRARY_PATH
D=dataset/LimoPro_dataset_mt_v2
INIT="${INIT:-weights/hybridnets.pth}"
OUTD=outputs/train/hybridnets_mt_v4_seed0
PY="$ROOT/.venv/bin/python"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

pick_python() {
  for c in "${PYTHON:-}" "$HOME/anaconda3/bin/python3.12" python3.12 python3.11 python3.10; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}

setup_python() {
  [ -f .venv/.ready ] && return 0
  local P
  P="$(pick_python)" || { echo "Python 3.10 이상이 없습니다. PYTHON=/경로/python3.12 bash run.sh 로 지정하세요"; exit 1; }
  log "venv 만들기: $P ($("$P" --version 2>&1))"
  "$P" -m venv .venv || { echo "venv 실패"; exit 1; }
  "$PY" -m pip install -U pip
  "$PY" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
  "$PY" -m pip install -r requirements.txt
  touch .venv/.ready
}

setup_data() {   # LimoPro_dataset_mt_v2.zip (또는 나눈 .zip.part00, part01, ...) -> dataset/LimoPro_dataset_mt_v2
  [ -d "$D/synth_signs_v2/train/images" ] && return 0
  local Z="${DATA_ZIP:-}"
  if [ -z "$Z" ]; then
    for c in "$ROOT/LimoPro_dataset_mt_v2.zip" "$ROOT/../LimoPro_dataset_mt_v2.zip"; do [ -f "$c" ] && { Z="$c"; break; }; done
  fi
  if [ -z "$Z" ]; then   # 나눈 파일이면 이어 붙이기
    for c in "$ROOT" "$ROOT/.."; do
      if ls "$c"/LimoPro_dataset_mt_v2.zip.part* >/dev/null 2>&1; then
        log "나눈 데이터 파일 이어 붙이기 -> $c/LimoPro_dataset_mt_v2.zip"
        cat "$c"/LimoPro_dataset_mt_v2.zip.part* > "$c/LimoPro_dataset_mt_v2.zip"; Z="$c/LimoPro_dataset_mt_v2.zip"; break
      fi
    done
  fi
  [ -n "$Z" ] && [ -f "$Z" ] || { log "학습 데이터가 없습니다: LimoPro_dataset_mt_v2.zip(또는 .zip.part*) 을 이 폴더나 상위 폴더에 두세요"; exit 1; }
  log "데이터 풀기: $Z -> dataset/ (수 분)"
  mkdir -p dataset
  "$PY" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall('dataset'); print('풀기 끝')" "$Z"
}

check() {
  local bad=0
  for d in "$D/synth_signs_v2/train/images" "$D/synth_signs_v2/val/images" "$D/synth_signs_v2/test/images" "$D/bag_mt_v1/train/images" \
           "$D/bag_mt_v1/test_bag1116/images" "$D/real_tl_v1/images/train" "$D/real_fork_v1/images/train" "$D/real_fork_v1/images/test" \
           "$D/venue_ref_20261106" models/HybridNets; do
    [ -d "$d" ] || { log "없음: $d"; bad=1; }
  done
  [ -s "$INIT" ] || { log "없음: $INIT"; bad=1; }
  [ "$bad" = 0 ] || exit 1
  log "합성 train $(ls "$D/synth_signs_v2/train/images" | wc -l), bag train $(ls "$D/bag_mt_v1/train/images" | wc -l), 실물 신호등 train $(ls "$D/real_tl_v1/images/train" | wc -l), 실물 표지판 train $(ls "$D/real_fork_v1/images/train" | wc -l)"
  if command -v sha256sum >/dev/null 2>&1 && [ -f SHA256SUMS ]; then sha256sum -c --quiet SHA256SUMS || exit 1; log "SHA256SUMS OK"; fi
  log "CPU 코어 $(nproc), 메모리 $(free -g | awk '/Mem/{print $2}') GiB"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/  /'
    nvidia-smi --query-gpu=name,mig.mode.current,power.limit,clocks.max.sm --format=csv,noheader | sed 's/^/  /' || true
    nvidia-smi -L | grep -q MIG && log "⚠ MIG 조각이 보입니다. 조각은 H100 의 일부(1/7 단위)만 써서 RTX 보다 느릴 수 있음 — 가능하면 전체 GPU 로"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/  GPU 사용 중: /' || true
  fi
  "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA GPU가 안 보입니다 (nvidia-smi 확인)"
free, total = torch.cuda.mem_get_info()
print(f"GPU {torch.cuda.get_device_name(0)}, 메모리 {total / 2**30:.0f} GiB 중 {free / 2**30:.0f} GiB 비어 있음, bf16 {torch.cuda.is_bf16_supported()}", flush=True)
EOF
}

train_once() {
  "$PY" scripts/train_multitask.py --seg-data "$D/synth_signs_v2" --det-data none \
    --seg-extra "$D/bag_mt_v1" \
    --real-data "$D/real_tl_v1" --real-repeat "${REAL_TL_REPEAT:-2}" \
    --det-extra "$D/real_fork_v1:${REAL_FORK_REPEAT:-2}:all" \
    --venue-ref-dir "$D/venue_ref_20261106" --init "$INIT" --merge-ab \
    --lr "${LR:-1e-4}" --head-lr-mult 10 --det-weight 2 --warmup-steps "${WARMUP:-500}" \
    --epochs "${EPOCHS:-150}" --patience 0 --overfit-drop 0 --es-val-loss "${ES_PATIENCE:-8}" --best-by val_loss --lane-guard 0 \
    --ignore-small 0 --oversample-ab 2 \
    --aug-monitor-p "${MONITOR_P:-0.4}" --aug-monitor-dots "${DOTS_P:-0.5}" --aug-monitor-real "${MONITOR_REAL:-all}" --aug-cable-p 0 \
    --amp --batch "$BATCH" --workers "${WORKERS:-0}" --prefetch "${PREFETCH:-4}" \
    $([ "${CHANNELS_LAST:-1}" = 1 ] && echo --channels-last) $([ "${COMPILE:-0}" = 1 ] && echo --compile) \
    --out "$OUTD" "$@" 2>&1 | tee -a "$OUTD/train.log"
}

run_train() {
  mkdir -p "$OUTD"
  while :; do
    log "학습 hybridnets_mt_v4: $INIT (원본 HybridNets) 에서, 최대 ${EPOCHS:-150} epoch, lr ${LR:-1e-4}, batch $BATCH (로그: $OUTD/train.log)"
    local start_lines; start_lines=$(wc -l < "$OUTD/train.log" 2>/dev/null || echo 0)
    if train_once "$@"; then return 0; fi
    if tail -n +"$((start_lines + 1))" "$OUTD/train.log" | grep -q -i "out of memory" && [ "$BATCH" -gt 2 ]; then
      BATCH=$((BATCH / 2)); log "GPU 메모리 부족 -> batch $BATCH 로 다시"; continue
    fi
    log "학습 실패. $OUTD/train.log 끝을 확인하세요"; exit 1
  done
}

pack() {
  local OUT="results_v4_$(hostname).zip"
  log "결과 묶기 -> $OUT"
  "$PY" - "$OUT" "$OUTD" <<'EOF'
import os, sys, zipfile
out, outd = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "best_val_loss.pt", "best_score.pt", "log.csv", "test.json", "calib.json", "train.log"):
        p = os.path.join(outd, f)
        if os.path.exists(p):
            z.write(p)
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")
EOF
}

if [ -z "$LIMO_INHIBITED" ] && command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle --who=LimoPro --why=check true >/dev/null 2>&1; then
  export LIMO_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who=LimoPro --why=training bash "$ROOT/$(basename "$0")" "$STAGE" "$@"
fi

case "$STAGE" in
  check) setup_python; setup_data; check ;;
  train) setup_python; setup_data; check; run_train "$@" ;;
  pack)  setup_python; pack ;;
  all)   setup_python; setup_data; check; run_train "$@"; pack ;;
  *) echo "사용: bash run.sh [all|check|train|pack]"; exit 1 ;;
esac
log "끝 ($STAGE)"
