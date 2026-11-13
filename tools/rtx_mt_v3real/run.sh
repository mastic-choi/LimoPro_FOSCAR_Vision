#!/usr/bin/env bash
# LimoPro 다중 작업 v3 + 실물 신호등 추가 학습 (RTX 3070 Ti PC, foscar) 실행 스크립트.
# multitask-v3 에서 이어서, 로봇 카메라로 찍은 실물 신호등 사진(dataset/real_tl_v1)을 2배로 섞어 짧게 학습한다.
# 콘(cones_v1)은 쓰지 않는다 (미션 8은 라이다로 처리). 출력 12클래스는 그대로라 TensorRT 변환·C++ 노드는 바꿀 필요 없음.
#   bash run.sh            전부: 확인 -> 학습 -> 실물 신호등 비교 -> 모니터 시험 -> 결과 zip
#   bash run.sh check      데이터·가중치·GPU 확인만 (데이터 zip이 옆에 있으면 여기서 풂)
#   bash run.sh train      학습만 (추가 인자는 train_multitask.py로 넘어감)
#   bash run.sh evalreal   실물 신호등 비교만 (v3 vs 학습 결과 후보들)
#   bash run.sh pack       결과 zip만
# 끊겨도 같은 명령을 다시 실행하면 last.pt부터 이어서 합니다. GPU 메모리가 모자라 죽으면 batch를 절반으로 줄여 이어서 합니다.
# 환경변수 (괄호 안이 기본값)
#   EPOCHS(30) 최대 epoch, lr은 이 epoch 수에 맞춰 줄어듦 / LR(5e-5) / ES_PATIENCE(4) val loss 조기 종료 /
#   LANE_GUARD(0.01) 차선 보호: val da+ll IoU가 시작(v3)보다 이만큼 넘게 낮은 epoch은 best.pt로 안 고름 /
#   REAL_REPEAT(2) 실물 train 사진을 epoch마다 몇 번 넣을지 / BATCH(8, 메모리 부족이면 자동으로 4, 2) / WORKERS(8) /
#   FREEZE_BN(1) BatchNorm 통계를 v3 값으로 고정 (v3는 batch 32로 학습, 여기서는 4~8이라 통계가 흔들리지 않게) /
#   MONITOR_P(0.4) DOTS_P(0.5) MONITOR_REAL(all) 신호등 뒤 방해 모니터 증강 (실물 사진에도 그림) /
#   GPU_DUTY(0.9) GPU 평균 점유율 상한 (공용 PC 보호, 1.0 = 제한 없음) /
#   INIT(weights/hybridnets_mt_v3_best.pt) 시작 가중치 / PYTHON(자동: ~/anaconda3/bin/python3.12 우선) /
#   DATA_ZIP(자동: ./ 또는 ../LimoPro_dataset_mt_v1.zip)
set -eo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
STAGE="${1:-all}"; [ $# -gt 0 ] && shift
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
export GPU_DUTY="${GPU_DUTY:-0.9}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # 이 PC에서 최대 740MiB 절약 (학습환경 노트)
unset PYTHONPATH LD_LIBRARY_PATH                                                       # .bashrc의 ROS·CUDA 12.1 경로가 섞이지 않게
DATA=dataset/synth_signs_v2
REAL=dataset/real_tl_v1
INIT="${INIT:-weights/hybridnets_mt_v3_best.pt}"
OUTD=outputs/train/hybridnets_mt_v3real_seed0
PY="$ROOT/.venv/bin/python"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

pick_python() {   # 이 PC 기본 python3는 3.8이라 최신 PyTorch가 안 깔림 -> Anaconda 3.12
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

setup_data() {   # 데이터 zip(LimoPro_dataset_mt_v1.zip)에서 합성 데이터와 대회장 조명 견본만 풂 (콘은 안 씀)
  [ -d "$DATA/train/images" ] && [ -d dataset/venue_ref_20261106 ] && return 0
  local Z="${DATA_ZIP:-}"
  [ -n "$Z" ] || for c in "$ROOT/LimoPro_dataset_mt_v1.zip" "$ROOT/../LimoPro_dataset_mt_v1.zip"; do [ -f "$c" ] && { Z="$c"; break; }; done
  [ -n "$Z" ] && [ -f "$Z" ] || { log "학습 데이터가 없습니다: LimoPro_dataset_mt_v1.zip 을 이 폴더나 상위 폴더에 두거나 DATA_ZIP=경로 로 지정하세요"; exit 1; }
  log "데이터 풀기: $Z -> dataset/ (synth_signs_v2, venue_ref_20261106만, 수 분 걸림)"
  "$PY" - "$Z" <<'EOF'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
top = "LimoPro_dataset_mt_v1/"
keep = (top + "synth_signs_v2/", top + "venue_ref_20261106/")
names = [n for n in z.namelist() if n.startswith(keep) and not n.endswith("/")]
for i, n in enumerate(names):
    dst = os.path.join("dataset", n[len(top):])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with z.open(n) as s, open(dst, "wb") as d:
        d.write(s.read())
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1}/{len(names)}", flush=True)
print("풀기 끝:", len(names), "files", flush=True)
EOF
}

check() {
  local bad=0
  for d in "$DATA/train/images" "$DATA/train/labels" "$DATA/train/da_masks" "$DATA/val/images" "$DATA/test/images" \
           "$DATA/test_venue/images" dataset/venue_ref_20261106 "$REAL/images/train" "$REAL/images/val" "$REAL/images/test" models/HybridNets; do
    [ -d "$d" ] || { log "없음: $d"; bad=1; }
  done
  [ -s "$INIT" ] || { log "없음: $INIT"; bad=1; }
  [ "$bad" = 0 ] || exit 1
  log "합성 train $(find "$DATA/train/images" -type f | wc -l)장, 실물 train $(ls "$REAL/images/train" | wc -l) / val $(ls "$REAL/images/val" | wc -l) / test $(ls "$REAL/images/test" | wc -l) / test_1m2 $(ls "$REAL/images/test_1m2" 2>/dev/null | wc -l)장"
  if command -v sha256sum >/dev/null 2>&1 && [ -f SHA256SUMS ]; then sha256sum -c --quiet SHA256SUMS || exit 1; log "SHA256SUMS OK"; fi
  "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA GPU가 안 보입니다 (nvidia-smi 확인)"
free, total = torch.cuda.mem_get_info()
print(f"GPU {torch.cuda.get_device_name(0)}, 메모리 {total / 2**30:.2f} GiB 중 {free / 2**30:.2f} GiB 비어 있음, bf16 {torch.cuda.is_bf16_supported()}", flush=True)
EOF
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/  GPU 사용 중: /' || true
}

train_once() {
  "$PY" scripts/train_multitask.py --seg-data "$DATA" --det-data none \
    --real-data "$REAL" --real-repeat "${REAL_REPEAT:-2}" \
    --venue-ref-dir dataset/venue_ref_20261106 --init "$INIT" \
    --lr "${LR:-5e-5}" --head-lr-mult 10 --det-weight 2 --warmup-steps 200 \
    --epochs "${EPOCHS:-30}" --patience 0 --overfit-drop 0 --es-val-loss "${ES_PATIENCE:-4}" --best-by val_loss --lane-guard "${LANE_GUARD:-0.01}" \
    --ignore-small 14 --oversample-ab 2 \
    --aug-monitor-p "${MONITOR_P:-0.4}" --aug-monitor-dots "${DOTS_P:-0.5}" --aug-monitor-real "${MONITOR_REAL:-all}" --aug-cable-p 0 \
    $([ "${FREEZE_BN:-1}" = 1 ] && echo --freeze-bn) \
    --amp --batch "$BATCH" --workers "$WORKERS" --out "$OUTD" "$@" \
    2>&1 | tee -a "$OUTD/train.log"
}

run_train() {
  mkdir -p "$OUTD"
  while :; do
    log "학습: $INIT 에서 이어서, 실물 x${REAL_REPEAT:-2}, 콘 없음, 최대 ${EPOCHS:-30} epoch, lr ${LR:-5e-5}, batch $BATCH (로그: $OUTD/train.log)"
    local start_lines; start_lines=$(wc -l < "$OUTD/train.log" 2>/dev/null || echo 0)
    if train_once "$@"; then return 0; fi
    # 이번 실행에서 메모리 부족으로 죽었으면 batch 절반으로 이어서 (last.pt에서 재개)
    if tail -n +"$((start_lines + 1))" "$OUTD/train.log" | grep -q -i "out of memory" && [ "$BATCH" -gt 2 ]; then
      BATCH=$((BATCH / 2)); log "GPU 메모리 부족 -> batch $BATCH 로 다시"; continue
    fi
    log "학습 실패. $OUTD/train.log 끝을 확인하세요"; exit 1
  done
}

eval_real() {   # 실물 신호등: v3(시작) vs 학습 결과 후보들. 수집 영상 test + 1.2m 12장
  log "실물 신호등 비교 -> $OUTD/eval_real/eval_real.txt"
  local args=(--ckpt "v3=$INIT")
  for f in best.pt best_val_loss.pt best_score.pt; do [ -s "$OUTD/$f" ] && args+=(--ckpt "${f%.pt}=$OUTD/$f"); done
  "$PY" scripts/eval_real_tl.py --real "$REAL" "${args[@]}" --out "$OUTD/eval_real" 2>&1 | tee "$OUTD/eval_real.log" || true
}

eval_monitor() {
  [ -s "$OUTD/best.pt" ] || { log "best.pt가 없어 모니터 시험을 건너뜁니다"; return 0; }
  log "모니터 시험 (합성 test 신호등 뒤 모니터)"
  "$PY" scripts/eval_mt_monitor.py "$OUTD/best.pt" --name v3real --out "$OUTD/eval_monitor" 2>&1 | tee "$OUTD/eval_monitor.log" || true
}

pack() {
  local OUT="results_v3real_$(hostname).zip"
  log "결과 묶기 -> $OUT"
  "$PY" - "$OUT" "$OUTD" <<'EOF'
import os, sys, zipfile
out, outd = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "best_val_loss.pt", "best_score.pt", "log.csv", "test.json", "calib.json", "train.log",
              "eval_monitor.log", "eval_real.log"):
        p = os.path.join(outd, f)
        if os.path.exists(p):
            z.write(p)
    for sub in ("eval_monitor", "eval_real"):
        for d, _, fs in os.walk(os.path.join(outd, sub)):
            for f in fs:
                z.write(os.path.join(d, f))
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")
EOF
}

# 절전 방지(권한 없으면 무시)
if [ -z "$LIMO_INHIBITED" ] && command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle --who=LimoPro --why=check true >/dev/null 2>&1; then
  export LIMO_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who=LimoPro --why=training bash "$ROOT/$(basename "$0")" "$STAGE" "$@"
fi

case "$STAGE" in
  check)    setup_python; setup_data; check ;;
  train)    setup_python; setup_data; check; run_train "$@"; eval_real; eval_monitor ;;
  evalreal) setup_python; eval_real ;;
  pack)     setup_python; pack ;;
  all)      setup_python; setup_data; check; run_train "$@"; eval_real; eval_monitor; pack ;;
  *) echo "사용: bash run.sh [all|check|train|evalreal|pack]"; exit 1 ;;
esac
log "끝 ($STAGE)"
