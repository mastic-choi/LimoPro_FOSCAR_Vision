#!/usr/bin/env bash
# 사용: bash scripts/run_mt_exp_ab.sh  (tmux 안에서 권장)
# 실험 A(검출 머리·BiFPN lr x10, 검출 손실 x2), B(A + 백본 p3 지름길)를 차례로 학습 -> 채점 -> 비교표.
# 1차 best.pt 에서 이어서 15 epoch. 끊기면 같은 명령으로 다시 실행하면 last.pt 부터 이어서 함.
# 결과: outputs/train/hybridnets_mt_exp{A,B}/ , 비교표 outputs/train/mt_exp_summary.md
cd "$(dirname "$0")/.."   # 저장소(또는 학습 번들) 최상위에서 실행
PY=.venv/bin/python
COMMON=(--seg-data dataset/synth_signs_v2 --det-data dataset/cones_v1 --venue-ref-dir dataset/venue_ref_20261106
        --init outputs/train/hybridnets_mt_v1_seed0/best.pt --amp --batch 32 --workers 16
        --epochs 15 --head-lr-mult 10 --det-weight 2 --warmup-steps 500)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

probe() { [ -s "$2" ] || "$PY" scripts/eval_mt_feature_probe.py . "$1" 2>&1 | grep -v -i "warn\|^init" > "$2"; }

run() {
  local name=$1; shift; local out=outputs/train/$name
  mkdir -p "$out"
  if [ ! -f "$out/test.json" ]; then
    log "$name 학습 시작"
    "$PY" scripts/train_multitask.py "${COMMON[@]}" --out "$out" "$@" 2>&1 | tee -a "$out/train.log"
  fi
  [ -f "$out/test.json" ] || { log "$name 학습 실패 (test.json 없음)"; return 1; }
  log "$name 채점"
  [ -f "$out/eval/ab_stats.json" ] || "$PY" scripts/eval_mt_ab.py . "$out/best.pt" "$out/eval" 2>&1 | grep -v -i warn > "$out/eval.log"
  probe "$out/best.pt" "$out/probe.txt"
  log "$name 끝"
}

V1=outputs/train/hybridnets_mt_v1_seed0
mkdir -p "$V1/eval"; [ -f "$V1/eval/ab_stats.json" ] || cp "$V1/montage/ab_stats.json" "$V1/eval/"
probe "$V1/best.pt" "$V1/probe.txt"
run hybridnets_mt_expA || true
run hybridnets_mt_expB --enc-skip || true
"$PY" -I scripts/summarize_mt_exp.py outputs/train/mt_exp_summary.md \
  "1차=$V1" "A=outputs/train/hybridnets_mt_expA" "B=outputs/train/hybridnets_mt_expB" > /dev/null
touch outputs/train/.mt_exp_done
log "전부 끝. 비교표: outputs/train/mt_exp_summary.md"
