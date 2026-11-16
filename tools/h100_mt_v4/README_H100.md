# hybridnets_mt_v4 학습 (H100 서버용, 2026-11-16)

**원본 HybridNets(BDD100K 공개 가중치)에서 새로 학습**합니다. v3real 이나 차선 계보에서 이어받지 않습니다.

## 준비물 (같은 폴더에)
| 파일 | 내용 |
|---|---|
| `LimoPro_mt_v4_h100.zip` | 코드, 원본 HybridNets 가중치(`weights/hybridnets.pth`), 실행 스크립트 |
| `LimoPro_dataset_mt_v2.zip` (3.7 GB) 또는 `LimoPro_dataset_mt_v2.zip.part00`, `part01`, … | 학습 데이터. 나눈 파일은 `run.sh` 가 이어 붙임 |

```bash
unzip LimoPro_mt_v4_h100.zip && cd LimoPro_mt_v4
bash run.sh check      # venv(처음만, 인터넷 필요: PyTorch 등) · 데이터 풀기 · GPU/MIG/CPU 확인
tmux new -s limo4
bash run.sh            # 학습 -> results_v4_<서버이름>.zip
```
- 끊기면 `bash run.sh` 다시 → `last.pt` 부터 이어서. 메모리 부족이면 batch 절반으로 자동.
- 결과: `outputs/train/hybridnets_mt_v4_seed0/` (`best.pt` = val loss 최저, `best_score.pt`, `best_val_loss.pt`, `test.json`, `calib.json`, `log.csv`). `results_v4_<서버이름>.zip` 만 보내 주면 됩니다.

## 출력
- 분할: da(내 차로), ll(차선), 정지선(흰 가로선)
- 검출 12칸(로봇 TensorRT·C++ 구조 그대로): 0 pedestrian, **1 fork_sign (A·B 합침, A/B 는 갈림길 분류기가 정함)**, 2 비움, 3 bump, 4 slow, 5 parking,
  6 traffic_light, 7~9 light_red/yellow/green, 10 limo(데이터 없음), 11 cone(데이터 없음)

## 데이터 (`LimoPro_dataset_mt_v2`)
| 폴더 | 학습 | 시험 | 라벨 |
|---|---:|---|---|
| `synth_signs_v2` | 9,304 | val 326, test 542, test_venue 143, test_bag 42 | 사람 검수 da·ll + 합성 박스. **정지선 사람 검수로 train 554 → 2,500장** |
| `bag_mt_v1` | 834 | test_bag1116 184 | USB 실차 bag, v3real pseudo-label 을 사람이 한 장씩 검수 (da·ll·정지선·박스) |
| `real_tl_v1` | 961 ×2 | test | 실물 신호등 (박스만) |
| `real_fork_v1` | 641 ×2 (빈 프레임 포함) | test 462 (A 1 + B 1 주행) | 실물 Ø80 mm A·B M7 주행, SAM 2.1 추적 박스 (박스만). 아주 먼 표지판(10~17 px)도 정답 |
| `venue_ref_20261106` | | | 대회장 조명 견본 (증강) |

콘(`cones_v1`)은 v3real 부터 안 써서 뺐습니다.

## 학습 설정 (`run.sh`)
- lr 1e-4 (검출 머리·BiFPN ×10), 워밍업 500 step, 최대 150 epoch, val loss 8 epoch 조기 종료, 차선 보호 없음(원본에서 시작)
- `--ignore-small 0` (작은 박스도 학습), A/B 합침 → 표지판 이미지도 좌우 반전, 표지판·주차 이미지 2배, 방해 모니터 0.4(땡땡이 50%)
- 대회장 조명 증강, batch 32, bf16

## 속도 (H100 이 RTX 보다 느려지는 원인과 대응)
v3 는 H100 에서 epoch 173 s 로 초당 약 80장이었습니다 — H100 이 이 모델에서 낼 수 있는 속도보다 훨씬 낮습니다.
H100 은 작은 커널이 많은 모델(HybridNets: 깊이별 합성곱, BiFPN)에서 GPU 연산보다 CPU 쪽 커널 실행·동기화·데이터 준비가 병목이 되기 쉽습니다.

| 원인 (이 PC 에서 잼) | 대응 |
|---|---|
| 데이터 준비: 학습 이미지 1장 CPU 51 ms, 그중 대회장 조명 증강(v3·v4) 1번에 75 ms | 큰 블러를 저해상도에서 계산 → v4 73→15 ms, v3 75→39 ms (결과 픽셀 차이 평균 0.5 이하). 장당 약 51 → 31 ms |
| 워커 16개 고정 | `WORKERS=0` = 코어 수 - 2 (최대 64), `PREFETCH=4`, persistent workers, pin_memory + non_blocking |
| 매 step `torch.cuda.synchronize()`(점유율 제한기, GPU_DUTY=1 에서도) + 손실 6개 `float()` | H100 에선 제한기 안 씀, 손실은 GPU 에 모아 100 step 마다만 꺼냄 |
| 커널 실행 부담 | batch 32, channels_last, TF32·cudnn.benchmark, fused AdamW. `COMPILE=1` 로 torch.compile 시도 가능 |
| MIG 조각 | `run.sh check` 가 `nvidia-smi -L` 로 MIG 를 알려 줌. 조각이면 H100 의 일부만 씀 |

**학습 중 보기**: epoch 줄 끝 `학습 N img/s, 데이터 대기 X%` (`log.csv` 의 `ips`, `data_wait`).
데이터 대기가 30% 를 넘으면 CPU(증강)가 병목 → `WORKERS` 를 코어 수까지, 또는 코어 많은 노드. 대기가 낮은데 img/s 가 낮으면 GPU 쪽 → `BATCH=64 LR=2e-4` 나 `COMPILE=1`.
