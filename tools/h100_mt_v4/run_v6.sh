#!/usr/bin/env bash
# hybridnets_mt_v6: v4 best.pt 에서 시작 (v5 는 대회장 신호등 장면 DA 꺼짐으로 사용 금지, 사용자 11-18).
#   v5 와 다른 점:
#     - 시작 가중치 v4 (v5 는 '대회장 배경 = 도로 없음'을 이미 배움)
#     - rawcam1117 의 빈 DA 정답 24장 뺌 (검수 'da·ll 비움' = 빈 정답으로 학습되어 v5 사고 원인) -> 192장 x3
#     - 11-18 대회장 bag 자동 라벨 bag_mt_venue1118 996장 x1 (DA·LL = v4 예측, ego lane·앞뒤 일관성으로 고름, 3차선 뺌, 빈 DA 없음)
#       학습에 안 쓴 bag: arall_1118_135742 (신호등까지 감), ar4_1118_135516 -> 릴리즈 전 DA 주행 검사용
#   나머지는 v5 와 같음 (glare 증강 0.3, 갈림길 박스 SAM, lr 1e-4, 150 epoch, val loss 12 epoch 0.005 조기 종료)
set -eo pipefail
cd "$(dirname "$0")"
export GPU_DUTY=1.0 VENUE_FAST=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH LD_LIBRARY_PATH
D=dataset/LimoPro_dataset_mt_v2; X=/home/guest6/Desktop/smoony/autorace2026/competition/label_work/datasets
OUTD=outputs/train/hybridnets_mt_v6_seed0; mkdir -p "$OUTD"
.venv/bin/python scripts/train_multitask.py --seg-data "$D/synth_signs_v2" --det-data none \
  --seg-extra "$D/bag_mt_v1" --seg-extra "$X/bag_mt_rawcam1117" --seg-extra "$X/bag_mt_rawcam1117" --seg-extra "$X/bag_mt_rawcam1117" \
  --seg-extra "$X/bag_mt_venue1118" \
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
outd = sys.argv[1]; out = f"results_v6_{os.uname().nodename}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for f in ("best.pt", "best_val_loss.pt", "best_score.pt", "log.csv", "test.json", "calib.json", "train.log"):
        p = os.path.join(outd, f)
        if os.path.exists(p): z.write(p)
print(out)
PY
echo "끝 $(date '+%H:%M:%S')"
