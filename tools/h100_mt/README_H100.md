# LimoPro 다중 작업 학습 (H100 서버용)

AutoRace 2026 FOSCAR팀 LIMO PRO의 인식 모델을 학습하는 묶음입니다.
HybridNets 하나로 **차선(주행 영역·차선) + 흰 가로선(정지선·미션 시작선) 분할**과
**표지판·신호등·라바콘·LIMO 검출**을 같이 학습합니다. 4차 차선 모델에서 이어서 시작합니다.

## 빠른 실행

```bash
unzip LimoPro_mt_h100.zip && cd LimoPro_mt
tmux new -s limo          # 오래 걸리니 tmux(또는 nohup) 안에서
bash run.sh               # 환경 설치 -> FSOCO 받기 -> Klein 다듬기 -> 학습 -> 결과 zip
```

- 끊겨도 `bash run.sh`를 다시 실행하면 끝난 단계는 건너뛰고 이어서 합니다(학습도 마지막 epoch부터).
- 끝나면 `LimoPro_mt/results_<서버이름>.zip`이 생깁니다. **이 파일만 보내 주시면 됩니다.**
- 인터넷이 필요합니다. 받는 양: PyTorch 등 약 3GB, FSOCO 약 25GB, Klein 모델 약 9GB.

## 단계와 예상 시간 (H100 1장 기준, 대략)

| 단계 | 하는 일 | 시간 |
|---|---|---|
| 1. 환경 | `.venv`에 PyTorch 2.11(CUDA 12.8) 설치 | 5~10분 |
| 2. FSOCO | 공개 라바콘 데이터 받기, 압축 풀기, 변환(`scripts/prepare_cones.py`) | 받는 속도에 따라 30분~2시간 |
| 3. Klein | stable-diffusion.cpp를 CUDA로 빌드 → 합성 표지판 사진을 이미지 편집 모델(FLUX 3 Klein)로 실물처럼 다듬기 | 빌드 10~20분 + 다듬기 수 시간 |
| 4. 학습 | `scripts/train_multitask.py`, 최대 40 epoch, 8 epoch 동안 안 좋아지면 조기 종료 | 수 시간 |
| 5. 결과 | `results_<서버이름>.zip` | 몇 분 |

단계별로만 돌리려면 `bash run.sh setup | klein | train | pack`.

## 필요한 것

| 항목 | 조건 |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA, 드라이버가 CUDA 12.x 지원 |
| Python | 3.10~3.12 (`sudo apt install python3 python3-venv`) |
| Klein 빌드 도구 | `nvcc`(CUDA 11.8 이상), `cmake`, `git`, `g++`. **없으면 run.sh가 알아서 설치합니다**(아래) |
| 디스크 | 약 75GB (FSOCO 원본 25GB + 압축 푼 것 25GB + Klein 모델 9GB + 나머지) |

### Klein 빌드 도구 자동 설치 (run.sh 맨 처음에 함)

1. `git`, `cmake`, `g++`가 없으면 `apt-get install git cmake build-essential` (root이거나 sudo 가능할 때. **비밀번호를 물으면 한 번 입력**)
2. `nvcc`가 없거나 11.8 미만이면 `apt-get install nvidia-cuda-toolkit` 시도
   (Ubuntu 22.04의 apt CUDA는 11.5라 H100용 빌드가 안 됩니다. 24.04는 12.0이라 됨)
3. 그래도 안 되면 **micromamba로 CUDA 12.4 컴파일러·cmake·g++를 `tools/cuda_env`에 설치**(관리자 권한 불필요, 약 3GB, 서버 환경은 안 건드림)
4. 그래도 실패하면 Klein만 건너뛰고 학습은 그대로 진행. 기록: `logs/apt.log`, `logs/mamba.log`, `logs/sd_build.log`

sudo 비밀번호 입력을 피하려면 미리 `sudo apt install git cmake build-essential`을 해 두거나, 그냥 Enter 없이 기다리지 말고 Ctrl+C 후 `SKIP_KLEIN=1 bash run.sh`로 Klein 없이 진행할 수 있습니다.

## 옵션 (환경변수)

| 변수 | 기본 | 설명 |
|---|---|---|
| `KLEIN_SHARDS` | 4 | Klein 프로세스를 동시에 몇 개 돌릴지. 하나에 VRAM 약 10~12GB |
| `BATCH` | 32 | 학습 batch (bf16). 메모리가 모자라면 16 |
| `WORKERS` | 16 | 데이터 로더 수. 대회장 조명 증강이 CPU를 많이 씀 |
| `SKIP_KLEIN=1` | | Klein 다듬기 생략 |
| `SKIP_FSOCO=1` | | FSOCO 생략(TraCon 콘 270장만) |

학습 옵션은 뒤에 붙이면 넘어갑니다: `bash run.sh train --epochs 60 --lr 5e-5`

## 데이터

| 폴더 | 내용 |
|---|---|
| `dataset/synth_signs_v2` | 차선 학습 데이터(ego_v4) 프레임에 표지판·신호등을 합성한 것. 한 장에 분할 마스크와 박스가 같이 있음. train 9,304 / val 326 / test 542 / test_venue 143 / test_bag 42 |
| `dataset/synth_signs_v2/*/sl_masks` | 흰 가로선 라벨. 11/06 대회장 프레임 중 트랙 도면으로 위치를 찾은 것만(train 554장). `sl_valid`는 라벨을 믿을 픽셀 |
| `dataset/cones_v1` | 라바콘: TraCon(포함) + FSOCO(서버에서 받음). 클래스는 cone 하나 |
| `dataset/venue_ref_20261106` | 대회장 조명 증강 색 견본 |
| `weights/hybridnets_v4_best_da.pt` | 시작 가중치(4차 차선 모델) |

검출 클래스(12): pedestrian, sign_A, sign_B, bump, slow, parking, traffic_light(몸체), light_red, light_yellow, light_green(켜진 불), limo, cone

## 학습 방식 (요약)

- 분할 출력 4채널: 배경 / 도로 / 차선 / 흰 가로선. 주행 영역 = 도로+차선+가로선이라 기존 da 정의가 유지됩니다.
- 이미지마다 가진 라벨의 손실만 씁니다: 콘 사진은 검출만, 가로선 라벨이 없는 사진은 가로선 손실 제외.
- A/B 표지판이 있는 사진은 좌우 반전하지 않습니다(화살표 방향이 바뀌어 라벨이 틀어짐).
- 매 epoch val에서 da/ll IoU와 클래스별 AP50을 재고, da+ll+mAP50이 가장 좋은 것을 `best.pt`로 저장합니다.
- 끝나면 best로 test / test_venue(가로선 IoU 포함) / test_bag을 채점해 `test.json`에 씁니다.

## 문제가 생기면

- 로그: `logs/`(FSOCO 변환, Klein, 빌드), `outputs/train/hybridnets_mt_v1_seed0/train.log`(학습)
- 메모리 부족: `BATCH=16 bash run.sh train`
- Klein만 다시: `rm dataset/synth_signs_v2/train/.klein_done && bash run.sh klein` (이미 한 것은 건너뜀)
