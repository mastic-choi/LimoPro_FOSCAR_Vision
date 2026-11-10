# 다중 작업 실험 비교

| | 1차 | 2차(실험 A) |
|---|---|---|
| best epoch | 33 | 14 |
| 기존 test da / ll | 0.952 / 0.819 | 0.955 / 0.816 |
| 기존 test mAP50 | 0.651 | 0.895 |
| 기존 test AP50 sign_A | 0.34 | 0.81 |
| 기존 test AP50 sign_B | 0.37 | 0.89 |
| 기존 test AP50 parking | 0.35 | 0.85 |
| 기존 test AP50 pedestrian | 0.42 | 0.96 |
| 기존 test AP50 bump | 0.61 | 0.97 |
| 대회장 test da / ll | 0.934 / 0.493 | 0.935 / 0.486 |
| 대회장 test mAP50 | 0.714 | 0.937 |
| 대회장 test AP50 sign_A | 0.48 | 0.84 |
| 대회장 test AP50 sign_B | 0.40 | 0.88 |
| 대회장 test AP50 parking | 0.57 | 0.91 |
| 대회장 test AP50 pedestrian | 0.59 | 0.99 |
| 대회장 test AP50 bump | 0.53 | 0.90 |
| test_bag da / ll | 0.981 / 0.820 | 0.981 / 0.826 |
| test_bag mAP50 | 0.751 | 0.952 |
| test_bag AP50 sign_A | 0.48 | 0.96 |
| test_bag AP50 sign_B | 0.47 | 0.88 |
| test_bag AP50 parking | 0.46 | 0.93 |
| test_bag AP50 pedestrian | 0.50 | 0.83 |
| test_bag AP50 bump | 0.90 | 1.00 |
| 대회장 test 흰 가로선 IoU | 0.578 | 0.604 |
| A/B 2택 정답률 (train) | 59% | 86% |
| A/B 최종 예측 정답률 (train) | 38% | 81% |
| A/B 2택 정답률 (test) | 58% | 84% |
| A/B 최종 예측 정답률 (test) | 30% | 77% |
| A/B 2택 정답률 (test_venue) | 57% | 84% |
| A/B 최종 예측 정답률 (test_venue) | 34% | 82% |
| A/B 2택 정답률 (test_bag) | 54% | 92% |
| A/B 최종 예측 정답률 (test_bag) | 23% | 69% |

## linear probe: 1차

```
##### outputs/train/hybridnets_mt_v1_seed0/best.pt
boxes: train 4526, test 260, test_venue 66, test_bag 20
layer         dim  train 3종 | test 3종 / A/B | test_venue 3종 / A/B | test_bag 3종 / A/B
enc_p2        288     0.977 | 0.954 / 0.966 | 0.939 / 0.932 | 1.000 / 1.000
enc_p3        432     0.962 | 0.942 / 0.960 | 0.924 / 0.955 | 0.900 / 0.923
bifpn_P3     1440     0.859 | 0.800 / 0.863 | 0.742 / 0.795 | 0.700 / 1.000
bifpn_P4     1440     0.723 | 0.615 / 0.646 | 0.591 / 0.614 | 0.450 / 0.615
```

## linear probe: 2차(실험 A)

```
##### outputs/train/hybridnets_mt_expA/best.pt
boxes: train 4526, test 260, test_venue 66, test_bag 20
layer         dim  train 3종 | test 3종 / A/B | test_venue 3종 / A/B | test_bag 3종 / A/B
enc_p2        288     0.979 | 0.969 / 0.971 | 0.955 / 0.932 | 1.000 / 1.000
enc_p3        432     0.966 | 0.954 / 0.971 | 0.924 / 0.977 | 0.950 / 1.000
bifpn_P3     1440     0.934 | 0.873 / 0.903 | 0.848 / 0.841 | 0.950 / 1.000
bifpn_P4     1440     0.854 | 0.773 / 0.823 | 0.803 / 0.773 | 0.900 / 0.846
```

