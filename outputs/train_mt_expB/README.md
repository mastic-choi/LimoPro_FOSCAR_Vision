# 실험 B 결과 (`multitask-v2-skip`, 참고용)

2차(`multitask-v2`, 실험 A)와 같은 학습 설정에, **백본 p3 특징을 1×1 conv로 BiFPN P3에 더해 검출 머리에 넣는 지름길(`det_skip`)**을 추가한 결과입니다. 1차 best.pt에서 이어서 15 epoch 학습했습니다.
가중치는 릴리즈 [`multitask-v2-skip`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/multitask-v2-skip)에 있습니다. **차에 쓸 모델은 `multitask-v2`를 추천합니다.**

## A(v2)와 비교

| | 1차 | A (`multitask-v2`) | B (이것) |
|---|---:|---:|---:|
| 검출 mAP50 (기존 / 대회장 / bag) | 0.651 / 0.714 / 0.751 | 0.895 / 0.937 / 0.952 | 0.898 / 0.948 / 0.960 |
| A/B 2택 (train / 기존 / 대회장 / bag) | 59 / 58 / 57 / 54% | 86 / 84 / 84 / 92% | 84 / 84 / 86 / 92% |
| 18px 미만 A/B 2택 (기존 / 대회장) | - | 61 / 67% | 67 / 58% |
| 차선 da / ll, 대회장 test | 0.934 / 0.493 | 0.935 / 0.486 | 0.938 / 0.492 |
| 흰 가로선 IoU, 대회장 test | 0.578 | 0.604 | 0.603 |
| linear probe A/B, 검출 머리 입력 (기존 test) | 86% | 90% | **95%** |

- 지름길로 검출 머리가 받는 A/B 정보는 백본 수준(95%)까지 올라왔지만, 검출 머리의 A/B 출력은 84~86%로 A와 같습니다.
- mAP와 대회장 차선은 조금 높지만(+0.003~0.011), 시드 1개와 작은 test 세트 기준이라 우연일 수 있는 차이입니다.
- 전체 비교표: `compare_v1_A_B.md`

## 쓸 때 주의

이 가중치에는 `det_skip.weight`, `det_skip.bias`가 더 있습니다.

- `HybridNetsBackbone`에 `load_state_dict(strict=False)`로 읽으면 `det_skip`이 **조용히 빠져서 검출 결과가 틀립니다.**
- `scripts/train_multitask.py`의 `build_model()`로 읽고(`det_skip`을 자동으로 붙임), `forward_mt()`로 추론해야 합니다.
- ONNX·TensorRT로 바꿀 때도 `forward_mt()`처럼 P3에 `det_skip(p3)`를 더하는 forward로 내보내야 합니다.

## 파일

| 파일 | 내용 |
|---|---|
| `hybridnets_mt_v2_skip_test.json`, `hybridnets_mt_v2_skip_log.csv` | 채점 결과, epoch별 기록 (best epoch 15) |
| `ab_stats.json`, `probe.txt` | A/B 표지판별 예측, 층별 linear probe |
| `compare_v1_A_B.md` | 1차·A·B 비교표 |
