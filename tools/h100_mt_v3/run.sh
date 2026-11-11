#!/usr/bin/env bash
# LimoPro 다중 작업 v3 (차선 분할 + 흰 가로선 + 표지판·신호등·콘 검출) H100 실행 스크립트.
# v2 학습 때 서버에 풀어 둔 LimoPro_mt 폴더 위에 이 묶음을 덮어 풀고 실행한다.
# 이미지는 새로 만들지 않고 서버에 있는 v2 데이터(dataset/synth_signs_v2, Klein 다듬기 끝난 것)와 cones_v1(FSOCO 포함)을 그대로 쓴다.
# 신호등 뒤 방해 모니터(단색, 땡땡이)는 학습 중 바로 그려 넣는다(scripts/mt_online_aug.py).
#   bash run_v3.sh            전부: 확인 -> 학습(+temperature 보정) -> 모니터 시험 -> 결과 zip
#   bash run_v3.sh check      데이터·가중치·환경 확인만
#   bash run_v3.sh train      학습만 (추가 인자는 train_multitask.py로 넘어감)
#   bash run_v3.sh evalmon    모니터 앞 신호등 시험만 (best.pt)
#   bash run_v3.sh pack       결과 zip만
# 끊겨도 같은 명령을 다시 실행하면 last.pt부터 이어서 합니다.
# 환경변수 (괄호 안이 기본값)
#   증강:  MONITOR_P(0.4) 신호등이 있는 이미지에 뒤 모니터를 그릴 확률 / DOTS_P(0.5) 그중 땡땡이 화면 비율 /
#          MONITOR_REAL(all) 실제 카메라처럼: match(선명도·노이즈),persp(원근),screen(비네팅·반사),jitter(땡땡이 흔들기) /
#          CABLE_P(0 = 끔) 흰 케이블을 그릴 확률 (대회장에는 전선이 없어 기본으로 끔)
#   학습:  EPOCHS(120) 최대 epoch (lr은 이 epoch 수에 맞춰 줄어듦) / ES_PATIENCE(5) val loss 조기 종료: val loss 최저 epoch 뒤로
#          5 epoch 동안 더 낮아지지 않으면(올라가기 시작하면) 멈춤, best.pt = val loss 최저 / ES_MIN_DELTA(0) /
#          LANE_GUARD(0.01) 차선 보호: val da+ll IoU가 시작(v2)보다 0.01 넘게 낮은 epoch은 best.pt로 안 고름 /
#          IGNORE_SMALL(14) 이 px 미만 박스 무시 / OVERSAMPLE_AB(2) / CALIB_LOSS(none; sece|mdca|both) /
#          CALIB_WEIGHT(0 = 학습 중 보정 손실 끔, 학습 후 val temperature만 씀) / BATCH(32) / WORKERS(16) / TORCH_INDEX /
#          GPU_DUTY(1.0 = 제한 없음) GPU 평균 점유율 상한. 로컬 AMD PC 전원 보호용 기본값(0.9)을 H100에서는 풀어 full speed로 학습
#   시작:  INIT(weights/hybridnets_mt_v2_best.pt). v2 서버 폴더의 weights/hybridnets_v4_best_da.pt(4차 차선)도 가능
#          (검출 머리·흰 가로선 채널은 새로 시작, 차선 보호 기준도 그 가중치)
set -eo pipefail   # 학습이 실패하면(tee 앞 명령 실패 포함) 결과 묶기까지 가지 않고 멈춤
cd "$(dirname "$0")"
ROOT="$(pwd)"
STAGE="${1:-all}"; [ $# -gt 0 ] && shift
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-16}"
export GPU_DUTY="${GPU_DUTY:-1.0}"   # H100은 GPU 점유율 제한 해제 (make_model_compare_gif.py 기본 0.9는 로컬 AMD PC 전원 보호용)
DATA=dataset/synth_signs_v2
INIT="${INIT:-weights/hybridnets_mt_v2_best.pt}"   # 4차 차선(ego-lane-v4)에서 시작: INIT=weights/hybridnets_v4_best_da.pt
OUTD=outputs/train/hybridnets_mt_v3_seed0
PY="$ROOT/.venv/bin/python"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_python() {   # v2 때 만든 .venv가 있으면 그대로 사용
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

check() {   # v2 데이터가 서버에 남아 있는지 확인 (없으면 v2 묶음을 먼저 풀어야 함)
  local bad=0 n
  for d in "$DATA/train/images" "$DATA/train/labels" "$DATA/train/da_masks" "$DATA/val/images" "$DATA/val/labels" \
           "$DATA/test/images" "$DATA/test_venue/images" \
           dataset/cones_v1/images models/HybridNets dataset/venue_ref_20261106; do
    [ -d "$d" ] || { log "없음: $d"; bad=1; }
  done
  [ -s "$INIT" ] || { log "없음: $INIT"; bad=1; }
  [ "$bad" = 0 ] || { log "v2 학습 때 쓴 LimoPro_mt 폴더 위에 이 묶음을 풀었는지 확인하세요"; exit 1; }
  # test_bag은 있으면 채점에 쓰고 없으면 train_multitask.py가 알아서 건너뜀 -> 경고만
  [ -d "$DATA/test_bag/images" ] || log "주의: $DATA/test_bag/images 가 없음 -> test_bag 채점은 건너뜀 (test/test_venue로만 채점)"
  n=$(find "$DATA/train/images" -name '*.png' | wc -l)
  log "v2 합성 train 이미지 $n장, sl_masks $(find "$DATA/train/sl_masks" -name '*.png' 2>/dev/null | wc -l)장, Klein $([ -f "$DATA/train/.klein_done" ] && echo 끝 || echo 안 함)"
  [ -f dataset/cones_v1/.fsoco_done ] || log "주의: FSOCO 변환 표시(dataset/cones_v1/.fsoco_done)가 없음 -> TraCon 콘만으로 학습"
  if command -v sha256sum >/dev/null 2>&1 && [ -f SHA256SUMS ]; then sha256sum -c --quiet SHA256SUMS || exit 1; fi
}

run_train() {
  log "학습 시작: $INIT 에서 이어서, 최대 ${EPOCHS:-120} epoch, val loss ${ES_PATIENCE:-5} epoch 안 내려가면 정지, 모니터 p=${MONITOR_P:-0.4}(땡땡이 ${DOTS_P:-0.5}, 효과 ${MONITOR_REAL:-all}) 케이블 p=${CABLE_P:-0} (로그: $OUTD/train.log)"
  mkdir -p "$OUTD"
  "$PY" scripts/train_multitask.py --seg-data "$DATA" --det-data dataset/cones_v1 \
    --venue-ref-dir dataset/venue_ref_20261106 --init "$INIT" \
    --head-lr-mult 10 --det-weight 2 --warmup-steps 500 \
    --epochs "${EPOCHS:-120}" --patience 0 --overfit-drop 0 --es-val-loss "${ES_PATIENCE:-5}" --es-min-delta "${ES_MIN_DELTA:-0}" --best-by val_loss --lane-guard "${LANE_GUARD:-0.01}" \
    --ignore-small "${IGNORE_SMALL:-14}" --oversample-ab "${OVERSAMPLE_AB:-2}" \
    --calib-loss "${CALIB_LOSS:-none}" --calib-weight "${CALIB_WEIGHT:-0}" \
    --aug-monitor-p "${MONITOR_P:-0.4}" --aug-monitor-dots "${DOTS_P:-0.5}" --aug-monitor-real "${MONITOR_REAL:-all}" --aug-cable-p "${CABLE_P:-0}" \
    --amp --batch "$BATCH" --workers "$WORKERS" --out "$OUTD" "$@" \
    2>&1 | tee -a "$OUTD/train.log"
}

eval_monitor() {   # 학습 끝난 best.pt로 모니터 앞 신호등 시험 (v2와 같은 방법: scripts/eval_mt_monitor.py)
  [ -s "$OUTD/best.pt" ] || { log "best.pt가 없어 모니터 시험을 건너뜁니다"; return 0; }
  log "모니터 시험 (기본 모니터, 실제처럼 그린 모니터)"
  "$PY" scripts/eval_mt_monitor.py "$OUTD/best.pt" --name v3 --out "$OUTD/eval_monitor" 2>&1 | tee "$OUTD/eval_monitor.log" || true
  "$PY" scripts/eval_mt_monitor.py "$OUTD/best.pt" --name v3 --real match,persp,screen,jitter --out "$OUTD/eval_monitor" \
    2>&1 | tee -a "$OUTD/eval_monitor.log" || true
}

pack() {
  local OUT="results_v3_$(hostname).zip"
  log "결과 묶기 -> $OUT"
  "$PY" - "$OUT" "$OUTD" <<'EOF'
import os, sys, zipfile
out, outd = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "best_val_loss.pt", "best_score.pt", "last.pt", "log.csv", "test.json", "calib.json", "train.log", "eval_monitor.log"):
        p = os.path.join(outd, f)
        if os.path.exists(p) and not (f == "last.pt" and os.path.getsize(p) > 3e8):
            z.write(p)
    for d, _, fs in os.walk(os.path.join(outd, "eval_monitor")):
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
  check) setup_python; check ;;
  train) setup_python; check; run_train "$@"; eval_monitor ;;
  evalmon) setup_python; eval_monitor ;;
  pack)  setup_python; pack ;;
  all)   setup_python; check; run_train "$@"; eval_monitor; pack ;;
  *) echo "사용: bash run_v3.sh [all|check|train|evalmon|pack]"; exit 1 ;;
esac
log "끝 ($STAGE)"
