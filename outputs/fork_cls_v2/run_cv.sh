#!/usr/bin/env bash
# fork_cls_v2: 실물 Ø80 mm A·B 주행(11-16 밤 5개) 추가 학습. 교차 검증 2겹 x (뒤집은 B 를 A 로 씀 / 안 씀)
#   cv1: 실물 220858_A + 215318_B 를 학습에서 빼고 시험 (보통 조명)
#   cv2: 실물 214321_A + 213635_B 를 학습에서 빼고 시험 (빨간·보라 조명이 섞인 주행)
# 공통: --mask-prob 0.5 --tint-prob 0.5 (HANDOFF 2-3), 특징 l3
set -euo pipefail
cd "$(dirname "$0")/../.."
D=dataset/fork_crops_v1
PY=.venv/bin/python
BAGS="$D/bag_030233 $D/bag_030611 $D/bag_1115_234946 $D/bag_1116_014225 $D/bag_1116_014651 $D/bag_1116_181759 $D/bag_1116_183018"
REAL_ALL="$D/real1116_210015_B $D/real1116_213635_B $D/real1116_214321_A $D/real1116_215318_B $D/real1116_220858_A"
run() {   # 이름 빼는A 빼는B 추가인자...
  local name=$1 hoA=$2 hoB=$3; shift 3
  local tr=""
  for r in $REAL_ALL; do [[ $r == *"$hoA"* || $r == *"$hoB"* ]] || tr="$tr $r"; done
  $PY scripts/train_fork_cls.py --mask-prob 0.5 --tint-prob 0.5 "$@" --out outputs/fork_cls_v2/$name \
    --train $D/synth/train $BAGS $tr --real $BAGS $tr \
    --eval venue=$D/synth/test_venue bag=$D/synth/test_bag realA=$D/real1116_${hoA} realB=$D/real1116_${hoB}
}
run cv1_flip   220858_A 215318_B --flip-real-b
run cv1_noflip 220858_A 215318_B
run cv2_flip   214321_A 213635_B --flip-real-b
run cv2_noflip 214321_A 213635_B
