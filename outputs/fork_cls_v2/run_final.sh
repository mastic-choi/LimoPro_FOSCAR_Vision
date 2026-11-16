#!/usr/bin/env bash
# fork_cls_v2 최종: 실물 A·B 5개 주행 + 기존 데이터 전부. 교차 검증(run_cv.sh)에서 --flip-real-b 쪽이 실물 A·B 모두 높아서 켬.
# 내보내기: ONNX + 기준 입력·출력(실물 A 크롭), OpenCV cv::dnn 일치 확인은 check_fork_cls_dnn.py (로봇 OpenCV 4.5.4)
set -euo pipefail
cd "$(dirname "$0")/../.."
D=dataset/fork_crops_v1
PY=.venv/bin/python
BAGS="$D/bag_030233 $D/bag_030611 $D/bag_1115_234946 $D/bag_1116_014225 $D/bag_1116_014651 $D/bag_1116_181759 $D/bag_1116_183018"
REAL="$D/real1116_210015_B $D/real1116_213635_B $D/real1116_214321_A $D/real1116_215318_B $D/real1116_220858_A"
O=outputs/fork_cls_v2/final_v2
$PY scripts/train_fork_cls.py --flip-real-b --mask-prob 0.5 --tint-prob 0.5 --out $O \
  --train $D/synth/train $BAGS $REAL --real $BAGS $REAL \
  --eval venue=$D/synth/test_venue bag=$D/synth/test_bag
$PY scripts/export_fork_cls.py $O/head.pt $O/fork_sign.onnx $D/real1116_214321_A $D/real1116_210015_B
