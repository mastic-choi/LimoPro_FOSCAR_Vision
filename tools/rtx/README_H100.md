# LimoPro 차선 인식 3차 학습 (H100 서버용)

AutoRace 2026 FOSCAR팀 LIMO PRO의 차선 인식 모델을 학습하는 묶음입니다. 압축을 풀고 명령 한 줄이면 환경 설치부터 학습, 결과 정리까지 됩니다.

**2차 학습 가중치에서 이어서 학습합니다(기본).** 2차 모델에 11/06 대회장 조명 사진과 대회장 조명 증강을 더해 3차 모델을 만듭니다.

## 빠른 실행

```bash
unzip LimoPro_train_v3_h100.zip && cd LimoPro_train
bash run_train.sh --models hybridnets           # HybridNets만
bash run_train.sh                               # 4개 모델 전부 (hybridnets, tln_large, yolop, yolopv2 순서)
```

- 처음 실행하면 `.venv`를 만들고 PyTorch 2.11(CUDA 12.8) 등을 설치합니다. 인터넷이 필요하고 약 3GB를 받습니다.
- 끊겨도 같은 명령을 다시 실행하면 끝난 모델은 건너뛰고 마지막 epoch부터 이어서 합니다.
- 끝나면 `LimoPro_train/results_<서버이름>.zip`이 생깁니다. **이 파일만 보내 주시면 됩니다.**
- 오래 걸리니 `tmux`나 `nohup` 안에서 돌리는 것을 권합니다. 로그는 `outputs/train/<모델>_v3_seed0/train.log`에 쌓입니다.

## 필요한 것

| 항목 | 조건 |
|---|---|
| OS | Linux x86_64 (glibc 2.28 이상) |
| GPU | NVIDIA, 드라이버가 CUDA 12.x를 지원하면 됨 (RTX 3070 Ti, 드라이버 560에서 확인) |
| Python | 3.10, 3.11, 3.12 중 하나가 PATH에 있어야 함. 없으면 `sudo apt install python3 python3-venv` |
| 디스크 | 약 12GB (코드와 데이터 2GB + venv 7GB + 결과) |

## 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models` | 4개 전부 | 학습할 모델 |
| `--init` | `v2` | 시작 가중치. `v2` = 2차에서 이어서(기본), `v1` = 1차, `none` = BDD100K 사전학습부터 |
| `--venue-p 0.7` | 0.7 | 대회장 조명 증강을 적용할 비율. `--venue-ref-dir none`이면 끔 |
| `--epochs 40` | 40 | 최대 epoch. 8 epoch 동안 val이 안 오르면 조기 종료 |
| `--batch 8` | 8 | 시작할 때 메모리를 시험해 모자라면 4, 2로 줄임. 지난 라운드와 비교하려고 8로 둠 |
| `--workers 4` | 4 | 데이터 로더 수. 대회장 조명 증강이 CPU를 써서, 코어가 많으면 8~16 권장 |
| `GPU_DUTY` 환경변수 | 1.0 | GPU 사용률 상한. 공용 서버라 낮추려면 `GPU_DUTY=0.8 bash run_train.sh` |

예: `bash run_train.sh --models hybridnets --workers 16`

## 데이터 (`dataset/ego_v3`)

LIMO PRO 카메라(1280×720)를 640×360으로 줄인 사진과 마스크입니다. 마스크는 0/255 PNG입니다.

| split | 장수 | 내용 |
|---|---:|---|
| train | 3,988 | 10/27 일반 조명 3,524 + 11/06 대회장 조명 464 |
| val | 324 | 10/27, best 가중치 선택용 |
| test | 542 | 10/27, 지난 라운드와 같은 test |
| test_venue | 143 | 11/06 대회장 조명, 코스 구간 5곳에서 30장씩 |

- **da (주행가능영역):** 지금 달리는 차선만 칠합니다. 3차선 구간과 A/B 갈림길도 예외 없이 내 차로만입니다.
- **ll (차선):** 차선 위치입니다. 라벨 관례상 da는 ll을 포함합니다.
- 연속 프레임이라 test 구간 앞뒤 프레임은 학습에서 뺐습니다. 무작위로 나누면 점수가 부풀려집니다.
- `dataset/venue_ref_20261106`: 대회장 조명 증강의 색 견본(라벨 없음). test_venue 사진은 들어 있지 않습니다.

## 학습 방식

- 입력 640×384 (위아래 12px 패딩), 손실 BCE + soft Dice (ll 가중치 2.0), AdamW, poly LR
- 모델별 LR: HybridNets 0.0001, TwinLiteNet+ Large 0.00005, YOLOP 0.0001(채널별 BCE), YOLOPv2 0.00001
- 증강: 대회장 조명(실제 대회장 사진의 조명 분포를 옮김) 70%, 좌우 반전, 회전·확대, 색, 블러, 노출 과다, 반사, 차선 끊김, 매트 주름

## 결과 (`results_<서버이름>.zip`)

모델마다 `best_da.pt`, `best_ll.pt`, `log.csv`(epoch별 val), `test.json`, `train.log`가 들어갑니다.
`test.json`의 `test`는 기존 test, `test_venue`는 대회장 test(구간별 점수 포함)입니다.

지난 라운드 결과 (기존 test, da / ll):

| 모델 | 1차 | 2차 |
|---|---|---|
| HybridNets | 0.967 / 0.809 | 0.958 / 0.802 |
| TwinLiteNet+ Large | 0.954 / 0.784 | 0.956 / 0.783 |
| YOLOP | 0.940 / 0.749 | 0.946 / 0.769 |
| YOLOPv2 | 0.962 / 0.810 | 0.959 / 0.806 |

## 문제가 생기면

- `CUDA를 쓸 수 없습니다`: `nvidia-smi`가 되는지 확인
- `venv 실패`: `sudo apt install python3-venv`
- 설치가 중간에 끊김: `.venv` 폴더를 지우고 다시 실행
- 메모리 부족으로 멈춤: `bash run_train.sh --batch 4`로 다시 실행하면 이어서 합니다

폴더 안의 `CLAUDE.md`, `README_RTX.txt`는 동아리 PC에서 Claude Code로 돌릴 때 쓰는 설명이라 무시하셔도 됩니다.
