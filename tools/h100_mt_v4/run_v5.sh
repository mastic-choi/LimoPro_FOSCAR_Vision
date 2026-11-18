#!/usr/bin/env bash
# hybridnets_mt_v5: v4 best.pt 에서 이어서, 대회장(11-17) 데이터 + 바닥 반사 증강 (사용자 11-18).
#   새 데이터: rawcam(무대 조명 19:37) da·ll 216장 x3 (v4 + SAM 반사 메우기 / SAM2 추적, 그림자 빼기, ll 잇기, 사람 검수),
#             갈림길 박스(SAM 2.1, 14px 까지): venue1117 B 54장 x3, rawcam A 121장 x3, arall A 184장 x3
#   시험 추가: test_venue1117 (첫 bag da·ll 46장, 학습에서 뺌)
#   증강 추가: --aug-glare-p 0.3 (floor_glare.py: 선 바로 옆 차로에 곧은 흰 띠, 라벨 그대로)
#   나머지 설정은 v4 run.sh 와 같음 (lr 1e-4, 검출 머리·BiFPN x10, warmup 500, val loss 조기 종료)
#   11-18 사용자: 60 → 150 epoch, 조기 종료 val loss 12 epoch 동안 0.005 넘게 안 내려가면 (epoch 11 의 last.pt 에서 이어서)
set -eo pipefail
cd "$(dirname "$0")"
export GPU_DUTY=1.0 VENUE_FAST=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH LD_LIBRARY_PATH
D=dataset/LimoPro_dataset_mt_v2; X=/home/guest6/Desktop/smoony/autorace2026/competition/label_work/datasets
OUTD=outputs/train/hybridnets_mt_v5_seed0; mkdir -p "$OUTD"
.venv/bin/python scripts/train_multitask.py --seg-data "$D/synth_signs_v2" --det-data none \
  --seg-extra "$D/bag_mt_v1" --seg-extra "$X/bag_mt_rawcam1117" --seg-extra "$X/bag_mt_rawcam1117" --seg-extra "$X/bag_mt_rawcam1117" \
  --seg-extra dataset/venue1117_test \
  --real-data "$D/real_tl_v1" --real-repeat 2 \
  --det-extra "$D/real_fork_v1:2:all" --det-extra "$X/real_fork_venue1117:3:all" --det-extra "$X/real_fork_rawcam1117:3:all" --det-extra "$X/real_fork_arall1117:3:all" \
  --venue-ref-dir "$D/venue_ref_20261106" --init outputs/train/hybridnets_mt_v4_seed0/best.pt --merge-ab \
  --lr 1e-4 --head-lr-mult 10 --det-weight 2 --warmup-steps 500 \
  --epochs 150 --patience 0 --overfit-drop 0 --es-val-loss 12 --es-min-delta 0.005 --best-by val_loss --lane-guard 0 \
  --ignore-small 0 --oversample-ab 2 \
  --aug-monitor-p 0.4 --aug-monitor-dots 0.5 --aug-monitor-real all --aug-cable-p 0 --aug-glare-p 0.3 \
  --amp --batch 32 --workers 0 --prefetch 4 --channels-last \
  --out "$OUTD" "$@" 2>&1 | tee -a "$OUTD/train.log"
.venv/bin/python - "$OUTD" <<'PY'
import os, sys, zipfile
outd = sys.argv[1]; out = f"results_v5_{os.uname().nodename}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "best_val_loss.pt", "best_score.pt", "log.csv", "test.json", "calib.json", "train.log"):
        p = os.path.join(outd, f)
        if os.path.exists(p): z.write(p)
print(out)
PY
echo "끝 $(date '+%H:%M:%S')"
