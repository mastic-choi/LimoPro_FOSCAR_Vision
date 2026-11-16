# 갈림길 A/B 표지판 크롭 분류기 (fork_cls_v1 → v2, 2026-11-16, 진행 중)

> **상태: 사전 릴리즈 [`fork-sign-v0.2`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/fork-sign-v0.2) (Pre-release).**
> v0.1 에 **실물 크기(Ø80 mm 인쇄) A·B 로 돌린 M7 주행 5개**(A 2, B 3)를 더해 다시 학습한 `fork_cls_v2/final_v2` 입니다.
> 학습에 안 쓴 실물 주행으로 교차 검증해 A↔B 혼동 0, 재현율 A 98~100% / B 98~100% 입니다([`outputs/fork_cls_v2/README.md`](../outputs/fork_cls_v2/README.md)).
> 실물 A·B bag 으로 가상 주행(확정 시각)을 아직 못 했습니다. 통과하면 정식 `fork-sign-v1` 로 올립니다.
>
> ```bash
> # 로봇: 모델 받기 (비공개 레포라 gh 로그인 필요). ar_fork_sign 노드 기본 경로가 ~/ar_models/fork_sign.onnx
> mkdir -p ~/ar_models && gh release download fork-sign-v0.2 -R mastic-choi/LimoPro_FOSCAR_Vision -p 'fork_sign.onnx*' -D ~/ar_models --clobber
> # TensorRT 엔진은 ONNX 를 바꿀 때마다 그 로봇에서 다시 (노드가 엔진을 먼저 읽음)
> /usr/src/tensorrt/bin/trtexec --onnx=$HOME/ar_models/fork_sign.onnx --saveEngine=$HOME/ar_models/fork_sign_fp16.engine --fp16 \
>   --minShapes=input:1x1x96x96 --optShapes=input:5x1x96x96 --maxShapes=input:5x1x96x96
> ```
>
> 이전: [`fork-sign-v0.1`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/fork-sign-v0.1) (`final_l3_mask`, 원 바깥 가림, 실물 A 없음),
> [`fork-sign-v0`](https://github.com/mastic-choi/LimoPro_FOSCAR_Vision/releases/tag/fork-sign-v0) (`final_l3`, 배경 지름길 있음). 아래 본문은 v1 때 기록입니다.

## 배경 지름길 (11-16 추가)

실차 B를 좌우로 뒤집어 A로 쓰면 화살표뿐 아니라 **글자(거울 B)와 배경(노트북 흰 막대)**도 뒤집힙니다. 그래서 "막대가 왼쪽이면 A"를 배울 수 있습니다.
`scripts/eval_fork_cls_mask.py`로 표지판 원 바깥을 회색으로 가리고 시험했습니다(학습에 안 쓴 bag의 B 크롭, 대회 판은 무늬 없는 인쇄라 가린 쪽이 실전에 가까움).

| 원 바깥 가린 뒤집은 B → A (반대쪽 오답) | 가림 증강 없음 (`cv_l3_flip_tr*`) | **가림 증강 0.5 (`cv_l3_flip_mask_tr*`)** |
|---|---:|---:|
| 030611 시험 | 90% (1%) | **97% (0%)** |
| 030233 시험 | 71% (7%) | **96% (1%)** |

`train_fork_cls.py --mask-prob 0.5`: 이 확률로 원(반지름 ≈ 크롭 한 변 / 3.2 × 1.0~1.25, 중심 ±3%) 바깥을 회색으로 가립니다.
가상 주행(030233, 학습에 안 쓴 모델): 가림 없음 B 6.79 s / 거울 A 10.50 s → 가림 B 10.20 s / 거울 A 10.44~10.50 s.
가림 모델은 B 쪽 치우침이 사라지고 먼 거리에서 신중해집니다. 문턱 0.5 / 0.1이면 B 5.52 s(인지만 쓴 방식 5.56 s), 거울 A 10.44 s, 방향 모두 맞음.

## 왜

- 갈림길 방향은 로봇의 M7(3차선, `ar_static_obstacle`)이 정합니다. 인지 검출에서 sign_A/sign_B(점수 0.3 이상)를 프레임마다 세서,
  먼저 6프레임이 된 쪽으로 확정합니다.
- 인지(HybridNets multitask)는 1280×720을 640×360으로 줄여서 보고, 표지판은 규정집 그림을 합성해서 배웠습니다.
- 11-16 실차 bag(`arall_1116_030233`, B)의 M7 구간에서 같은 B 표지판이 **sign_B 138회, slow 114회, sign_A 34회**로 흔들렸습니다.
- 교수님 제안: 검출 문턱을 낮춰 후보를 늘리고, 후보를 원본에서 잘라 2단계 분류기로 거른다.

## 구조

```
인지 검출 (문턱 0.05, 표지판류 6클래스 후보)  ┐
원본 영상 1280×720 (같은 stamp)              ├→ 후보 크롭 → 흑백·96px·CLAHE → 분류기 [A, B, 없음] → 프레임당 최대 1개 → M7
미션 상태 (M7 동안만)                        ┘
```

- 로봇 쪽은 팀 저장소 `ar_fork_sign` 패키지입니다(C++, `cv::dnn` CPU). 이 레포는 학습·내보내기만 합니다.
- **크롭·전처리 규칙은 로봇과 같아야 합니다.** `scripts/fork_crop_rule.py`와 C++ `crop_rect`·`preprocess`가 같은 숫자를 내는지
  양쪽 테스트(`scripts/test_fork_crop_rule.py`, `ar_fork_sign/test`)가 확인합니다.

## 분류기: 고정 ResNet18 특징 + 선형층

- ImageNet ResNet18은 고정하고 마지막 선형층만 학습합니다(linear probe). 첫 conv를 1채널로 접어 흑백 입력을 받습니다
  (`W' = Σ_c W[:,c]/std_c`, `b' = −Σ W·mean/std`). 학습·ONNX·로봇이 모두 접힌 모델을 씁니다.
- **특징은 layer3 공간 맵(256×6×6 = 9216)입니다.** 처음엔 전역 평균(GAP, 512)을 썼는데 A/B를 못 갈랐습니다. ImageNet 학습은
  좌우 반전 증강을 써서 특징이 좌우 대칭에 둔감하고, 평균을 내면 화살표가 어느 쪽인지라는 위치 정보가 사라지기 때문입니다.
  실차 B만으로 학습한 GAP 모델은 좌우를 뒤집은 B(가짜 A)를 99~100% **B**라고 했습니다. 노트북 화면처럼 생긴 것을 B로 외운 것입니다.
- 실물 A가 없어서 **실차 B 크롭을 좌우로 뒤집어 A로도** 씁니다(`--flip-real-b`). 화살표 방향은 맞고 글자는 거울 B입니다.
- 증강: 위치·크기 흔들기, 색 조명(채널 배율), 밝기·대비, 흐림, 축소 후 확대, JPEG. 좌우 반전 증강은 하지 않습니다(A↔B가 바뀜).
- 고르기(로봇과 같음): max(pA, pB) ≥ 0.6 이고 |pA − pB| ≥ 0.2 일 때만 그 클래스, 아니면 없음.

## 데이터

| 출처 | A | B | 없음 | 버림 |
|---|---:|---:|---:|---:|
| 합성 `synth_signs_v2` train (640 사진, 배율 1 크롭) | 1,509 | 1,511 | 13,331 (학습은 4,000만) | |
| 실차 `bag_030233` (1280 원본 크롭) | 0 | 187 | 118 | 78 |
| 실차 `bag_030611` | 0 | 156 | 98 | 36 |

- 실차 라벨은 bag 단위입니다(한 주행 안에서는 표지판이 안 바뀜). 검출을 IoU로 이은 트랙 몽타주를 보고 트랙이 표지판인지만 정했습니다.
- **A/B 모양이 안 보이는 크롭은 버렸습니다**(멀리서 하얗게 날아간 화면, 화면 가장자리, 작은 표지판이 섞인 원통 크롭).
  B라고 붙이면 "흰 원 = B"를 배워 A 주행에서 B라고 할 것이기 때문입니다.

## 결과

교차 검증: 실차 bag 하나로 학습하고 다른 bag으로 시험(둘 다 `--flip-real-b`). 고르기 규칙(0.6 / 0.2) 기준.

| 시험 | GAP 512 | layer4 4608 | **layer3 9216** |
|---|---:|---:|---:|
| 합성 대회장 `test_venue`: A↔B 혼동 / 재현율 | 2 / 39~55% | 1~2 / 68% | **0 / 98%** |
| 합성 `test_bag`: A↔B / 재현율 | 1~3 / 23% | 2 / 46~54% | **0 / 92%** |
| 실차 B (학습에 안 쓴 bag): A↔B / 재현율 | 1~2 / 91~96% | 1 / 99% | **0 / 99%** |
| 실차 B 좌우 반전 → A | 82~92% | 97~100% | **99%** |
| 없음을 A/B로 | 0~7% | 1~5% | **0~2%** |

최종(`final_l3`, 두 bag 모두 학습): 합성 val / test / 대회장 / bag의 A↔B 0 / 2 / 0 / 0, 재현율 91 / 93 / 98 / 92%.
ONNX(opset 11, 11.2 MB)를 로봇과 같은 OpenCV 4.5.4 `cv2.dnn`으로 돌려 torch와 최대 차이 2e-6입니다.

### 가상 주행 (실차 bag → 로봇 노드 → M7 규칙)

팀 저장소 `ar_fork_sign`의 C++ `fork_sign_replay`로 bag을 1배속으로 재생해, 같은 M7 규칙으로 분류기 방식과 예전 방식(인지 그대로)을
비교했습니다. 모델은 시험 bag을 학습에 안 쓴 교차 검증 모델입니다. 거울은 영상·박스를 좌우로 뒤집은 가짜 A 주행입니다.
예전 방식의 거울 결과는 인지가 뒤집힌 세상에서 sign_A/B를 맞바꿔 냈다고 가정한 것입니다(유리한 가정).

결과는 [`outputs/fork_cls_v1/README.md`](../outputs/fork_cls_v1/README.md)에 있습니다. 요약: **방향은 모두 맞고, 확정은 예전 방식보다 0.1~1.3초 늦습니다.**
예전 방식이 빠른 이유는 멀리서 하얗게 날아간 화면을 인지가 sign_B로 낸 프레임을 세기 때문입니다. 같은 bag에서 인지는 같은 원을
sign_A 0.11로도 냈습니다. 실물 A 주행에서 인지가 멀리 있는 밝은 원을 B라고 내면 예전 방식은 빨리, 그러나 틀리게 확정할 수 있습니다.
실물 크기 A bag이 있어야 확인됩니다.


### 고르기 문턱 (030233, 학습에 안 쓴 모델)

| min_prob / min_margin | B | A (거울) |
|---|---|---|
| 0.6 / 0.2 (기본) | B 6.82 s (+1.27) | A 10.67 s (+5.12) |
| 0.5 / 0.1 | B 5.39 s (−0.17) | A 10.44 s (+4.88) |
| 0.4 / 0.1 | B 5.35 s (−0.20) | **B 오답** |

괄호는 예전 방식 대비. 0.4에서 거울(A) 주행을 틀려서 기본은 0.6 / 0.2로 둡니다.

## 재현

```bash
python -m venv .venv && .venv/bin/pip install torch torchvision onnx opencv-python rosbags numpy
python scripts/fork_crops_synth.py dataset/synth_signs_v2 dataset/fork_crops_v1/synth
python scripts/fork_crops_bag.py extract <bag> <팀레포>/src/ar_msgs/msg dataset/fork_crops_v1/bag_<이름>
python scripts/fork_crops_bag.py label dataset/fork_crops_v1/bag_<이름> --sign A|B --tracks 3,7 --drop 5   # tracks_overview.png 보고
python scripts/train_fork_cls.py --flip-real-b --out outputs/fork_cls_v1/final_l3 \
    --train dataset/fork_crops_v1/synth/train dataset/fork_crops_v1/bag_030233 dataset/fork_crops_v1/bag_030611 \
    --real dataset/fork_crops_v1/bag_030233 dataset/fork_crops_v1/bag_030611 \
    --eval venue=dataset/fork_crops_v1/synth/test_venue bag=dataset/fork_crops_v1/synth/test_bag
python scripts/export_fork_cls.py outputs/fork_cls_v1/final_l3/head.pt outputs/fork_cls_v1/final_l3/fork_sign.onnx dataset/fork_crops_v1/bag_030233
python3 scripts/check_fork_cls_dnn.py outputs/fork_cls_v1/final_l3/fork_sign.onnx   # 로봇과 같은 OpenCV 에서
python scripts/test_fork_crop_rule.py && python scripts/test_fork_cls_fold.py
```

## 이어서 할 일

1. ~~대회용 크기 A·B 로 M7 녹화~~, ~~크롭 → 재학습(뒤집기 있/없 비교) → 내보내기 → 일치 확인~~: 11-16 밤 `fork_cls_v2` 로 함 (`fork-sign-v0.2`).
2. 실물 A·B 를 **bag 으로** 다시 녹화(`tools/ar.bash rec`, 검출·영상 포함) — 이번 5개는 M7 영상만 받아서 가상 주행을 못 함.
3. 새 bag으로 가상 주행. 실물 A·B 모두 방향이 맞고 확정 시각이 예전 방식과 비슷하거나 빠르면 릴리즈(`fork-sign-v1`).
4. 늦게 확정되면 `min_prob`·`min_margin`(0.6 / 0.2)을 낮춰 가상 주행으로 비교.
