# LimoPro 다중 작업 v3 학습 (H100 서버용)

v2 학습 때 서버에 풀어 둔 `LimoPro_mt` 폴더를 그대로 씁니다. **이미지는 이 zip에 없습니다.**
`dataset/synth_signs_v2`(Klein 다듬기 끝난 것), `dataset/cones_v1`(FSOCO 포함), `models/HybridNets`, `.venv`가 그 폴더에 남아 있어야 합니다.

## 실행

```bash
cd <v2 때 LimoPro_mt를 푼 폴더>      # LimoPro_mt 의 상위 폴더
unzip -o LimoPro_mt_v3_h100.zip       # scripts/ 일부를 새 버전으로 덮어씀 (v2 run.sh는 그대로)
cd LimoPro_mt
bash run_v3.sh check                  # 데이터·가중치 확인 (SHA256SUMS 검사 포함)
tmux new -s limo3
bash run_v3.sh                        # 학습 -> 모니터 시험 -> results_v3_<서버이름>.zip
```

- 끊기면 `bash run_v3.sh`를 다시 실행하면 last.pt부터 이어서 합니다.
- 끝나면 `LimoPro_mt/results_v3_<서버이름>.zip`만 보내 주시면 됩니다.
- 새로 받을 것은 없습니다(인터넷 불필요). 결과는 `outputs/train/hybridnets_mt_v3_seed0/`.
- 시간: v2가 H100에서 epoch당 약 190초였습니다. 최대 120 epoch(약 6시간 반)이고, val loss 조기 종료로 보통 더 일찍 끝납니다.

## v2와 다른 점

| 항목 | 내용 |
|---|---|
| 시작 가중치 | multitask-v2 best (`weights/hybridnets_mt_v2_best.pt`) |
| 방해 모니터 | 신호등이 있는 이미지에 확률 0.4로 신호등 뒤 16:9 화면을 그림 (학습 중, 라벨 없는 음성 샘플). 그중 50%는 땡땡이 화면: 어두운/흰 바탕에 신호등 불빛 크기의 원, 색은 켜진 불과 다른 색(초록불이면 빨강·노랑). 신호등 몸체와 불빛은 항상 모니터 앞에 남음 |
| 모니터를 실제처럼 | 주변 사진의 선명도·노이즈에 맞춰 흐림·노이즈, 좌우 ±25° 원근, 비네팅·반사 줄·밝은 색 날아감(땡땡이는 약하게), 땡땡이 원 위치·크기·간격 흔들기. `MONITOR_REAL`로 고름 |
| 흰 케이블 | 기본으로 끔 (대회장에는 전선이 없음). `CABLE_P=0.15`로 켤 수 있음 |
| 작은 박스 | √넓이 14px 미만은 무시. 불빛(light_red/yellow/green)은 원래 작아서 제외 |
| A/B/주차 | 들어간 이미지를 2배 자주 뽑음 |
| 보정 | 학습 중 보정 손실은 끔. 학습 후 val로 클래스별 temperature를 맞춰 `calib.json`에 저장 (차에서 p' = sigmoid(logit(p) / T)) |
| 조기 종료 | 최대 120 epoch. 매 epoch val에서 학습과 같은 손실(val loss)을 재고, val loss 최저 epoch 뒤로 5 epoch 동안 더 낮아지지 않으면(올라가기 시작하면) 멈춤. est.pt = val loss 최저 epoch, est_score.pt = 점수(da+ll IoU+mAP50) 최고 epoch (참고) |
| 차선 보호 | 학습 전에 시작 가중치(v2)를 val로 재 두고, val da+ll IoU가 그보다 0.01 넘게 낮은 epoch은 val loss가 최저여도 est.pt로 고르지 않음. 끝까지 그러면 시작 가중치가 est.pt로 남음 |
| 모니터 시험 | 학습 후 test·test_venue 신호등 사진에 원본/단색/땡땡이 모니터를 같은 자리에 그려 가짜 불빛·불빛 색을 잼(`eval_monitor/`). v2와 같은 seed라 바로 비교 가능 |

값은 환경변수로 바꿀 수 있습니다(`run_v3.sh` 맨 위 설명). 예: `EPOCHS=150 ES_PATIENCE=8 DOTS_P=0.3 MONITOR_REAL=match,persp bash run_v3.sh`
