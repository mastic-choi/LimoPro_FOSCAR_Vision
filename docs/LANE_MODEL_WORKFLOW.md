# 차선/주행가능영역 인식 모델 제작 워크플로우

> 출처: [mastic-choi/TwinLiteNet-KMU-finetune](https://github.com/mastic-choi/TwinLiteNet-KMU-finetune)
> (제9회 국민대 자율주행 경진대회, `README.md` + `PROGRESS.md` 기준)
>
> TwinLiteNet-KMU 팀이 실제로 거친 과정을 단계별 워크플로우로 정리했다.
> 시행착오는 "함정" 항목으로 따로 묶어서, LimoPro/AutoRace2026에 그대로 옮겨 쓸 수 있게 했다.

---

## 0. 한눈에 보기

```
[1] 문제 정의 ── 기존 모델이 우리 트랙에서 어디서 실패하는지 확인
     │
[2] 원본 수집·정리 ── 실차 주행 프레임 → 중복 제거 → 실패/커브/콘 위주로 다이어트
     │
[3] 초안 라벨 ── 기존 모델 출력을 초안 마스크로 → CVAT로 사람이 보정
     │
[4] 부트스트랩 학습 ── 사람이 검증한 소량 데이터로 미니 파인튜닝
     │
[5] 자동 라벨(pseudo-label) 파이프라인
     │   ├─ da(주행가능영역): 딥앙상블(medium×5 + 기존 모델) soft-vote
     │   └─ ll(차선): YOLOPv2(사전학습) 추론 → skeleton → polyline → 고정 두께
     │
[6] 사람 신뢰 리뷰 ── 자동 라벨을 신뢰/비신뢰로 빠르게 분류 + SAM 클릭 보정
     │        (검증 코퍼스 확장 → 앙상블 재학습 → 라벨 재생성: 반복)
     │
[7] 약점 데이터 추가 수집 ── 지그재그 주행 등 → 자동 라벨 → 육안 검수
     │
[8] 최종 배포 모델 학습 ── TwinLiteNetPlus medium, ll 손실·LR 가중
     │
[9] 검증 ── 사람 GT 기준 IoU/FP/FN + 약점 프레임 개별 확인 + 주행 GIF
     │
[10] 배포 ── ONNX export → 실차 PC에서 로드·속도 측정
```

**최종 결과 (v1.2.0, 사람 GT 1016장 기준)**

| 모델 | da IoU | ll IoU |
|---|---|---|
| TwinLiteNetPlus 사전학습(BDD100K, 파인튜닝 없음) | 0.815 | 0.132 |
| v1.0.0 (medium, pseudo_dataset_v2 1068장) | 0.945 | 0.577 |
| **v1.2.0 (medium, bootstrap_v2 + lap_005 = 3446장)** | **0.957** | **0.599** |

---

## 1. 문제 정의

- 기존에 쓰던 원조 [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)(da/ll 듀얼헤드
  세그멘테이션)은 실차 영상에서 다음 문제가 있었다.
  1. 노란 차선을 거의 못 잡음
  2. 역광/글레어 구간에서 미검출
  3. 커브·좌회전에서 da가 트랙 경계 밖까지 칠해짐
- 같은 트랙, 같은 차량으로만 달리는 좁은 도메인이므로 범용 모델을 찾는 것보다
  **우리 데이터로 직접 파인튜닝**하기로 결정했다.
- 베이스 모델은 [TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)
  (CAAM 어텐션, nano/small/medium/large)를 썼고, 최종적으로 **medium(약 47.9만 파라미터)**을 채택했다.
- **실패 판정 기준은 실차 제어 코드에서 그대로 가져왔다**: 실제 threshold(`DL_FG_THRESHOLD=0.5`,
  `DL_LL_FG_THRESHOLD=0.7`)와 ROI(`y=250:390`)를 사용했다.

## 2. 원본 데이터 수집 및 다이어트

| 단계 | 방법 | 스크립트 |
|---|---|---|
| 원본 | 실차 주행 프레임 2123장 (640×480, 170° 어안) | — |
| triage | 기존 모델로 전체 추론 → ROI 기준 실패 후보 CSV | `data_prep/triage_dataset.py` |
| 차선 개수 검사 | ROI를 좌/중/우로 나눠 3선 중 몇 개 잡히는지 | `data_prep/find_missing_line_frames.py` |
| 콘 검출 | YOLOv8 콘 검출기로 라바콘 프레임 표시 (과대표집 확인) | `data_prep/detect_cones.py` |
| 콘 축소 | 구간별 stride=5 + 구간 최고 신뢰도 프레임 보존 | `data_prep/build_cone_reduced_dataset.py` |
| 중복 제거 | dHash 근접 중복 제거, 쉬운 성공 프레임 다운샘플 | `data_prep/diet_dataset.py` |
| 결과 | **470장** (실패후보 133 / 콘 71 / 일반 266) | — |

추가로 수집한 데이터(lap_005, 지그재그 주행 3223장)는 다음 순서로 정리했다.
1. 연속 프레임 dHash 해밍거리 ≤2가 3장 이상 이어지면 정지 구간으로 보고 제거 → 2903장
2. YOLOPv2가 차선을 하나도 못 찾는 프레임 제거(`filter_no_lane_yolopv2.py`) → 2734장
3. 파일명에 `lap005_` 같은 **배치 접두어**를 붙인다. 기존 데이터와 `frame_NNNNNN` 파일명이 799개 겹쳤기 때문이다.

> ⚠️ 함정: 실패 후보 보존 로직이 CSV 하나만 참조해서 "진짜 실패 23장" 중 20장이 다이어트에서
> 빠졌다. 보존 조건은 **모든 판정 소스의 합집합**으로 걸어야 한다.

## 3. 초안 라벨 + CVAT 협업 라벨링

- 오버레이 이미지의 블렌드 공식을 역산해서 초안 da/ll 마스크를 복원했다(`extract_draft_masks.py`).
  재추론이 필요 없다.
- CVAT(app.cvat.ai)에서 쓴 라벨은 `drivable_area`, `lane_line`, `background`이다.
  **background도 실제 라벨로 등록해야 import 에러가 나지 않는다.**
- pre-annotation import 방식은 세 가지를 비교했다.
  1. Segmentation mask PNG: 폴리곤 점이 너무 많음 ❌
  2. COCO polygon(approxPolyDP): 객체가 여러 개로 쪼개짐 ❌
  3. **COCO RLE(iscrowd=1)**: 이미지당 da 1개 + ll 1개, 브러시로 편집 가능 ✅ (`prepare_coco_rle_import.py`)
- export는 Jobs CSV에서 **State=completed인 job의 프레임만** 가져왔다(`export_cvat_bootstrap.py`).
  미완료 프레임에는 원래 모델의 실패가 "정답"으로 남아 있기 때문이다.
- CVAT 무료 플랜의 SAM 호출에는 월 한도가 있다. 소진되면 로컬 SAM(ultralytics `SAM('sam2.1_b.pt')`)으로 대체한다.

## 4. 부트스트랩(Active learning) 미니 파인튜닝

- 사람이 완료한 74장으로 small config를 40epoch 학습했다: da mIoU 0.93, ll IoU 0.28~0.31.
- old/new 모델 비교 몽타주와 ROI 기준 정량 비교를 함께 봤다. 원래 완전히 실패하던 프레임은 극적으로
  좋아졌지만, 일부 잘 되던 프레임은 오히려 나빠졌다. **프레임별 편차를 반드시 같이 봐야 한다.**

## 5. 자동 라벨(pseudo-label) 파이프라인

### 5.1 ll(차선): YOLOPv2 + skeleton 정제

**모델 선정은 우리 GT로 직접 검증해서 했다.** 공개 벤치마크 순위는 우리 도메인 성능을 예측하지 못했다.

| 후보 | 결과 |
|---|---|
| **YOLOPv2 (BDD100K, 파인튜닝 0회)** | ✅ ll IoU 0.412, 당시 우리 모델의 7배 이상 |
| UFLD-v2 | ❌ 20장 중 17장 미검출 (CULane 전용) |
| CLRNet / GANet / CondLaneNet | ❌ CUDA 전용 커스텀 op |
| PETRv2 / LaneGAP | ❌ 멀티카메라 전제 |
| SegFormer / Mask2Former | ❌ "차선" 클래스 없음, da는 30~50% 과다예측 |
| Grounding DINO + SAM2 | ❌ 엉뚱한 위치를 두껍게 잡음 |

**후처리** (`pseudo_label/skeleton_polyline_utils.py`, `refine_ll_masks.py`)

1. 컴포넌트별 `skimage.morphology.skeletonize`
2. BFS 두 번으로 최장 경로를 구해 곁가지 제거 (tree diameter)
3. `cv2.approxPolyDP`로 polyline 단순화
4. `cv2.polylines`로 **고정 두께 8px** 재래스터화
5. 면적 1% 미만 또는 15px 미만인 노이즈 컴포넌트 제거

이렇게 하면 사람 브러시 두께 편차(4~48px, std 8.4px)가 std 2.4px로 줄어든다.
업계 표준(TuSimple/CULane/BDD100K)도 차선은 polyline으로 라벨링한다.

### 5.2 da(주행가능영역): 딥앙상블 soft-vote

- **라벨링은 큰(또는 여러) 모델로, 배포는 작은 단일 모델로**(이현기 교수님 피드백).
  처음에는 small 모델로 라벨을 만들어 medium을 학습하는 역순이었고, 그 결과 small 모델의 편향이 그대로 대물림됐다.
- 구성
  - 사람이 검증한 코퍼스로 **medium config 5개를 seed 0~4로 독립 학습**한다. 각각 BDD100K pretrained에서 시작한다.
  - 여기에 다양한 장면을 이미 본 **기존 배포 모델을 6번째 멤버**로 추가한다.
  - 확률을 평균한 뒤 0.5 threshold를 적용한다.
- 6개 조합을 고른 이유(156장 코퍼스 기준)

| | da IoU | FP | FN | 하단 고정물체 FP |
|---|---|---|---|---|
| 기존 단일 모델 | 0.904 | 6.2% | 4.2% | 25.8% |
| 신규 5개 | 0.901 | 4.5% | 5.9% | 15.9% |
| **6개(5 + 기존)** | **0.904** | **4.5%** | **5.6%** | **16.7%** |

- 앙상블은 **모델마다 다르게 틀릴 때만** 편향을 상쇄한다. 데이터가 부족해서 모두가 비슷하게 틀리는 경우는 잡지 못한다.
  그래서 코퍼스를 156장에서 1016장으로 늘려 v2 앙상블을 재학습했고, 약점 프레임이 개선됐다
  (frame_001766: 0.897 → 0.930, frame_001791: 0.901 → 0.970).
- 라벨 관례: `da_final = da | ll` (차선은 도로 위에 있다).
- 사람이 이미 검증한 프레임은 **자동 라벨로 덮어쓰지 않는다**(`merge_human_labels*.py`).

## 6. 사람 신뢰 리뷰 루프 (`da_trust_review.ipynb`)

CVAT 브러시 편집 대신 Gradio 기반 Anki식 리뷰 도구를 만들었다.

- 앙상블 da 오버레이를 보고 키보드로 판정한다: **1 = 신뢰(채택)**, **2 = 비신뢰 → SAM 클릭 보정**(포함/제외 점).
- 판정 즉시 마스크와 CSV를 저장하므로, 재실행하면 이어서 진행된다.
- 여러 명이 공개 링크로 동시에 작업할 수 있다: `gr.State` 세션 분리, `threading.Lock` 기반 claim, 10분 타임아웃.
- 실적: 자동 라벨 925장 → 797 신뢰 + 63 보정 = **860장 채택**, 65장 폐기.
  이 결과로 검증 코퍼스가 156장에서 **1016장(bootstrap_v2)**으로 늘었다.

**반복 구조:** 검증 코퍼스 → 앙상블 학습 → 자동 라벨 → 신뢰 리뷰 → 검증 코퍼스 확장 → 앙상블 재학습 → …

> 구현 함정:
> - `gr.HTML`의 `<script>`는 실행되지 않는다. `launch(js=...)`와 `inline=False`를 써야 한다.
> - ultralytics SAM에 점을 `(N,2)`로 넘기면 점마다 별도 객체로 처리된다. `(1,N,2)`로 감싸야 포함/제외 점이 한 객체로 결합된다.

## 7. 약점 데이터 추가 수집

- 팀 피드백: **차량이 차선과 평행하지 않을 때 da가 흔들린다.** 기존 데이터가 정상 주행 위주였기 때문이다.
- 대응: **지그재그 주행으로 lap_005를 수집**했다. 정리 과정은 §2에 있다.
- 앙상블 v2로 2734장에 자동 라벨을 만든 뒤, 사람이 이미지 폴더를 직접 훑으며 304장을 삭제했다
  (트랙을 벗어난 시점에 YOLOPv2가 벽이나 가구 윤곽을 차선으로 오검출한 경우). 남은 것은 **2430장**이다.
- 일반 원칙: "안 되는 상황 위주로 모아서 다음 학습에 반영한다." 커브, 글레어, 콘, 비평행 각도 등이 여기에 해당한다.

## 8. 최종 모델 학습

| 항목 | 값 |
|---|---|
| 베이스 | TwinLiteNetPlus `pretrained/medium.pth` (BDD100K) |
| 데이터 | bootstrap_v2 1016 + lap_005 2430 = **3446장**, train/val = 85/15 (2930/516) |
| 증강 | 이미지당 1개(데이터가 많아져서 3에서 줄임): 글레어, 모션블러, 바닥 반사 스트릭, 가짜 얇은 선(hard negative) |
| 입력 | **640×384 plain resize** (letterbox 사용 안 함) |
| 손실 | Tversky, **ll 손실 ×2.0** |
| LR | ll 디코더(`*_ll` 파라미터)만 **×2.5** (`poly_lr_scheduler`가 `lr_mult`를 반영하도록 패치) |
| epoch / batch | 40 / 4 (GPU 과부하로 PC가 강제종료된 뒤 배치마다 65ms throttle 추가) |
| 체크포인트 | 매 epoch `checkpoint.pth.tar`, `best.pth`(da 기준) + `best_ll.pth`(ll 기준) 별도 저장 |
| 환경 | Colab(무료 GPU, 할당량 부족) → **WSL2 Ubuntu + ROCm (RX 9070 XT)**. Windows 네이티브 ROCm은 MIOpen JIT가 크래시함 |

TwinLiteNetPlus 원본 코드에 적용한 패치(멱등 문자열 치환)

1. `BDD100K.py`: `letterbox()` 대신 `cv2.resize`를 쓰고, 라벨 높이를 360에서 384로 통일
2. `loss.py`, `utils.py`: `[12:-12]` 크롭 제거
3. `.cuda()` 하드코딩을 `resolve_device()`(cuda > mps > cpu)로 교체. AMP는 CUDA에서만 사용
4. `--seed` 인자 추가(앙상블용). `torch.load(..., weights_only=False)`
5. `best_da_miou`/`best_ll_iou`를 checkpoint에 저장해서 resume 시 best가 초기화되지 않게 함
6. 완료 검증 래퍼 스크립트: checkpoint epoch이 max_epochs에 도달했는지 확인하고, 아니면 자동으로 재시도

## 9. 검증 원칙

- **학습 로그 숫자만 믿지 않는다.** 항상 별도 스크립트로 실제 이미지에 추론해서 확인한다.
- 지표는 사람 GT 대비 **IoU + FP/GT(과다) + FN/GT(과소)**를 쓴다. 커버리지나 면적 비율만으로는 착시가 생긴다
  (SegFormer의 커버리지는 0.87이었지만 IoU는 0.75였다).
- ROI 밖의 시각 정보에 속지 않는다. 실차 ROI 기준 숫자로 판단한다.
- 전체 평균 대신 **카테고리별(커브/직선/역광/콘)과 알려진 약점 프레임**을 따로 확인한다.
- **GT 정의가 바뀌면**(예: ll 두께 재정의) 전후 로그 숫자를 직접 비교하지 않는다.
- **자기참조 주의:** 모델 A의 pseudo-label을 사람이 검수만 해서 만든 GT로 모델 A를 평가하면 점수가 부풀려진다.
- 시각 자료: old/new 3-way 몽타주와 연속 프레임 주행 GIF(직진 대조군, 커브, 복잡한 장면).
  관련 스크립트는 `scripts/eval/*`, `scripts/montage/*`에 있다.

## 10. 배포

- `scripts/export/export_best_onnx.py`: 입력 `images`, 출력 `da`/`ll`, 640×384.
  최신 torch exporter를 쓰려면 `onnxscript`가 필요하고, 가중치가 `.onnx.data`로 분리 저장된다.
- ⚠️ 파일명을 바꾸면 `.onnx` 내부의 external-data `location`도 고쳐야 한다(`onnx`로 재작성).
  그렇지 않으면 레포를 새로 clone한 사람은 모델을 로드하지 못한다.
- 실차 코드의 입력 높이(`DL_INPUT_H`)를 360에서 384로 바꿔야 한다.
- **실배포 하드웨어에서 속도를 반드시 직접 측정한다.**

| HW | 원조 TwinLiteNet | KMU medium |
|---|---|---|
| RX 9070 XT (ROCm, 학습 PC) | 46.3 fps | **91.5 fps** |
| Ryzen 7 5700U (CPU) | **12.4 fps** | 10.8 fps |
| Jetson Orin NX (PyTorch CUDA) | **31.8 fps** | 27.1 fps |

CAAM 어텐션 때문에 conv 레이어가 58개에서 133개로 늘었고, 이 커널 launch 오버헤드가 엣지 기기에서 크게 드러났다.
**학습 PC에서 빠르다고 엣지에서도 빠르다고 가정하면 안 된다.** 공정하게 비교하려면 두 모델을 같은 런타임으로 측정해야 한다.

---

## 11. 교훈 요약 (다시 겪지 않기 위해)

1. **이미지와 라벨의 전처리를 반드시 일치시킨다.** 원본 TwinLiteNetPlus는 이미지에 letterbox를, 라벨에는
   plain resize를 썼다. 16:9 BDD에서는 우연히 맞았지만, 4:3 카메라에서는 좌우 64px가 어긋났다.
   학습 전처리를 실차 추론 전처리와 대조해서 확인한다.
2. **pseudo-label은 생성 모델의 결함을 대물림한다.** 라벨을 만드는 모델부터 카테고리별로 검증한다.
3. **SAM 자동 적용은 이 도메인과 맞지 않는다.** da/ll 경계는 시각적 경계가 아니라 의미적 경계라서,
   완벽한 GT를 프롬프트로 줘도 IoU가 1.0에서 0.75로 떨어졌다. 사람이 클릭하는 보정 용도로만 쓴다.
4. 공개 벤치마크 순위는 도메인 전이 성능을 예측하지 못한다. 직접 검증한다.
5. 노트북 셀은 **멱등**하게 만든다. 패치 셀에는 "이미 패치됨" 가드를 넣고, 증강 셀은 `_aug*` 파일을 먼저 삭제한다.
   그렇지 않으면 셀을 재실행할 때마다 400장이 5200장으로 불어난다.
6. 장시간 무인 학습을 하기 전에는 Windows Update 자동 재부팅을 끄고 GPU power limit을 설정한다.
   WSL은 Windows 재부팅에서 살아남지 못한다.
7. `pkill -f '패턴'`은 자기 자신을 실행한 래퍼 셸까지 죽일 수 있다. PID 기반으로 종료한다.
8. 데이터 배치마다 파일명 접두어(`lapNNN_`)를 붙여 충돌을 막는다.

---

## 12. LimoPro(AutoRace2026)에 적용할 체크리스트

- [ ] 현재 쓰는 모델의 실패 유형 정의 (실차 ROI/threshold 기준)
- [ ] 실차 주행 원본 수집: 정상 주행 + **지그재그/비평행 각도** + 역광 + 장애물
- [ ] dHash 중복 제거, 정지 프레임 제거, 실패 프레임 우선 보존 다이어트
- [ ] 카메라 해상도/종횡비 확인 → **학습 전처리와 실차 전처리 일치 여부 검증**
- [ ] 소량(약 100~150장) 사람 라벨: CVAT COCO RLE 방식
- [ ] ll 소스 후보(YOLOPv2 등)를 우리 GT로 IoU 비교한 뒤 선정, skeleton/polyline 정제
- [ ] 사람 GT로 medium×5 앙상블 학습 → da soft-vote pseudo-label
- [ ] 신뢰 리뷰 도구로 검수 → 코퍼스 확장 → 앙상블 재학습 (1~2회 반복)
- [ ] 최종 배포 모델 학습 (ll 가중치, ll LR 배수, best/best_ll 분리 저장, resume 안전장치)
- [ ] 사람 GT 기준 IoU/FP/FN + 약점 프레임 + 주행 GIF로 검증
- [ ] ONNX export → LimoPro 실차 PC에서 로드와 FPS를 직접 측정
