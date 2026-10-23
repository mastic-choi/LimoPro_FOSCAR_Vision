# Ego lane pseudo-label 재현 가이드

> 대상: 이 레포를 처음 보는 사람 또는 LLM 에이전트.
> 이 문서만 따라 하면 `outputs/montages/ego_lane_final.gif`와 `outputs/ego_labels_*/` 라벨을
> 같은 방법으로 다시 만들 수 있게 썼다. 결정마다 "왜 그렇게 했는지"를 같이 적었으니,
> 다른 트랙/데이터에 적용할 때도 그 근거를 보고 판단하면 된다.

---

## 0. 한 줄 요약

공개 da/ll 모델 7개를 **앙상블**해서 차선을 얻고 → **노란색 HSV**로 빠진 차선을 보완 → 화면 아래부터
위로 **중심선 기준 좌/우 가장 가까운 차선**을 추적해 그 사이를 "지금 달리는 차선(ego lane)" da로 채우고 →
**SAM2** 결과를 그 범위 안으로만 잘라 다듬는다. 자동 판정 **FAIL** 프레임만 사람이 검수한다.

```
프레임 ─▶ [1] 7모델 앙상블 ─▶ [2] +노란색 HSV ─▶ [3] 차선 정리 ─▶ [4] 좌/우 경계 추적
                                                                       │
        라벨 저장 ◀─ [7] OK/FAIL 판정 ◀─ [6] SAM2 ∩ ego 범위 ◀─ [5] 두 차선 사이 채우기
```

LimoPro 2026-10-23 데이터(175장) 결과: **OK 90장(51%)**, FAIL 85장.

---

## 1. 준비

### 1.1 데이터

- 연속 주행 프레임 폴더, 파일명 `frame_NNNNNN.png` (예: `D:\fine-tune\LimoPro_frames_20261023\`, 640×480, 10Hz).
- 정지 구간(차가 서 있는 앞부분)은 `--start`로 건너뛴다. 2026-10-23 데이터는 frame 0~84가 정지 화면이라 `--start 85`.

### 1.2 모델 (`models/README.md`에 명령 있음)

| 용도 | 모델 | 위치 |
|---|---|---|
| 앙상블 멤버 | YOLOPv2, TwinLiteNet+ Nano/Small/Medium/Large, YOLOP, HybridNets (전부 BDD100K 사전학습, 파인튜닝 없음) | `models/<레포>/...` |
| 경계 다듬기 | SAM2.1-base (`sam2.1_b.pt`, ultralytics 배포본) | `models/sam2/sam2.1_b.pt` |

SAM2 가중치:
```bash
mkdir -p models/sam2
curl -L -o models/sam2/sam2.1_b.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt
```

### 1.3 파이썬 환경

```bash
python -m venv --system-site-packages .venv      # 전역 torch를 그대로 상속
.venv/Scripts/python -m pip install "timm==0.5.4" efficientnet_pytorch yacs prefetch_generator \
    tensorboardX webcolors pretrainedmodels ultralytics
```

- `timm==0.5.4` 고정: HybridNets가 `timm.models.layers.activations`를 import한다(최신 timm엔 없음).
- `ultralytics` 설치 전 `pip install --dry-run ultralytics`로 **torch가 바뀌지 않는지** 확인할 것
  (ZLUDA용 torch가 덮어써지면 GPU를 못 씀). 2.1.2+cu118 환경에선 ultralytics 8.3.90이 깔렸고 torch는 그대로였다.

### 1.4 GPU: NVIDIA면 그대로, AMD면 ZLUDA

NVIDIA(CUDA)에서는 추가 설정 없이 돈다. **AMD + ZLUDA**(우리 환경: RX 9070 XT, Windows)에서는 아래가 필요하고,
전부 `scripts/make_model_compare_gif.py` 상단에서 ZLUDA를 감지하면 자동 적용된다.

| 증상 | 원인 | 조치 |
|---|---|---|
| 9070 XT가 아니라 내장 그래픽만 잡힘 | 구버전 ZLUDA가 HIP 5(`amdhip64.dll`)에 링크 — RDNA4(gfx1201) 미지원 | lshqqytiger/ZLUDA 최신 릴리즈(`amdhip64_7.dll` 링크)로 교체 + HIP SDK 7.x 설치 |
| `CUDNN_STATUS_EXECUTION_FAILED` | torch에 들어있는 진짜 NVIDIA cuDNN은 ZLUDA에서 안 돎 | `torch.backends.cudnn.enabled = False` |
| YOLOPv2만 계속 cuDNN 에러 | TorchScript 그래프의 `batch_norm`에 `cudnn_enabled=True`가 상수로 박힘 | `torch.jit.freeze` 후 남은 batch_norm 노드의 8번째 입력을 False 상수로 교체 |
| YOLOPv2 두 번째 프레임에서 `0xC0000005` 크래시 | TorchScript NNC fuser가 NVRTC로 커널 컴파일 | `_jit_set_profiling_executor(False)`, `_jit_set_texpr_fuser_enabled(False)`, `_jit_override_can_fuse_on_gpu(False)` |
| SAM2 `kernel fmha_cutlassF... is for sm80-sm100, but was built for sm37` | SDPA가 flash/mem-efficient 커널 선택 | `enable_flash_sdp(False)`, `enable_mem_efficient_sdp(False)`, `enable_math_sdp(True)` |
| 첫 실행이 수 분씩 걸리고 GPU 0%, CPU 1코어만 바쁨 | ZLUDA가 처음 보는 커널을 CPU로 JIT 컴파일(정상) | 기다리면 됨. 캐시 `%LOCALAPPDATA%\ZLUDA\ComputeCache\zluda.db`에 남아 다음부턴 빠름(SAM2 첫 호출 119s → 이후 0.3s) |
| 멈춘 python이 안 죽고 `nvcuda.dll`을 잡고 있음 | GPU 호출 중 hang | venv 런처 PID까지 `taskkill /F /T /PID <pid>` |

GPU 사용률은 기본 **90%로 제한**된다(`GpuDutyLimiter`, 환경변수 `GPU_DUTY`, 1.0이면 무제한). 장시간 100% 부하로
PC가 꺼진 적(Kernel-Power 41)이 있어서다. 평균만 낮추는 장치라 순간 전력 스파이크까지 막으려면
AMD Adrenalin의 Power Limit도 낮춰 둘 것.

---

## 2. 실행

```bash
.venv/Scripts/python scripts/make_ego_lane_labels.py \
    --frames D:/fine-tune/LimoPro_frames_20261023 \
    --start 85 --step 2 --fps 5 \
    --save-labels outputs/ego_labels_20261023
```

| 옵션 | 의미 |
|---|---|
| `--start / --end / --step` | 정렬된 프레임 목록 슬라이스(정지 구간 건너뛰기, GIF 용량 조절) |
| `--fps` | GIF 재생 속도. 10Hz 데이터를 step 2로 뽑았으면 5가 실시간 |
| `--save-labels DIR` | `DIR/images/`, `DIR/da_masks/`, `DIR/ll_masks/`, `DIR/status.csv` 저장 |
| `--no-hsv` | HSV 보완 끄기(앙상블 차선만) |
| `--no-sam` | SAM2 생략, 최종 = 기하 결과 |
| `--out` | GIF 경로(기본 `outputs/montages/ego_lane_final.gif`) |

소요 시간(RX 9070 XT + ZLUDA, 컴파일 캐시 있는 상태, 90% 제한): 175장에 약 3분 + GIF 저장 1분.

출력 GIF 4칸: `원본 | 1. 앙상블 | 2. +HSV ego lane(초록 = 선택된 좌/우 경계, 상단 OK/FAIL) | 3. 최종(SAM2 ∩ ego)`.

---

## 3. 단계별 상세

640×480 기준 값. 다른 해상도면 비례해서 조정.

### [1] 7모델 앙상블

- 각 모델의 da/ll 출력을 **0~1 확률**로 맞춘다(모델마다 출력 형식이 다름):

  | 모델 | 출력 | 확률 변환 |
  |---|---|---|
  | TwinLiteNet+ | 2ch logit | softmax 후 채널 1 |
  | HybridNets | 3ch logit (bg/road/lane) | softmax; da = road + lane, ll = lane (multiclass라 차선 픽셀은 road에 안 찍힘) |
  | YOLOPv2 | da 2ch 확률, ll 1ch 시그모이드 | da = c1/(c0+c1), ll 그대로 |
  | YOLOP | 채널별 시그모이드 2ch | c1/(c0+c1) |

- 전처리는 각 레포 공식 demo를 따른다. 640×480은 `letterbox(640, auto=True)`에서 패딩 없이 통과.
  예외: YOLOP는 **BGR** 입력(공식 demo가 그렇게 넣음), HybridNets는 BiFPN 때문에 **128 배수**가 필요해 640×512로 패딩.
- **ll = Voting**: 픽셀별로 확률 ≥ 0.5인 모델이 4개(과반) 이상이면 차선.
- **도로 = da Averaging**: 7개 확률 평균 ≥ 0.5. ego 추적의 종료 조건/클리핑 범위로만 쓴다.

> 왜: Voting과 Averaging은 결과가 거의 같았다(`ensemble_vote_vs_avg.gif`). 앙상블 차선의 결정적 장점은
> **끊긴 흰 점선을 연속된 선으로 이어 준다**는 것(BDD100K 학습 덕). HSV/허프로는 조각만 나와 다시 이어야 한다.
> 반대로 모델 da는 "도로 전체" 개념이라 매트 전체를 칠하므로 ego lane 라벨로 그대로 쓸 수 없다.

### [2] 노란색 HSV 보완

- `H 15–40, S 70–255, V 90–255` (OpenCV HSV) → 3×3 opening → 도로(앙상블 da)를 25px 팽창한 범위 안만.
- 앙상블 ll과 OR.
- **흰색은 쓰지 않는다.**

> 왜: 앙상블이 지그재그 노란 선 같은 걸 놓친다. 흰색까지 넣으면 횡단보도, 마름모, 노면 글자, 화살표가 다
> 차선으로 잡혀 12장 테스트에서 OK가 7→5로 줄었다. 흰 점선 차선은 [1]이 이미 잘 잡는다.

### [3] 차선 정리 (`clean_ll`)

1. **화면 아래 15행**에서 가로로 40px 넘게 이어진 구간을 행 단위로 지운다.
2. 연결 성분별로 제거: 면적 < 150px / 화면 맨 아래에 닿고 높이 ≤ 15px / 높이 < 25px이면서 높이 < 0.15×폭(납작한 가로선).

> 왜: TwinLiteNet+ 등 여러 모델이 **화면 맨 아래 가장자리에 가짜 차선 줄**을 찍고, 이게 과반 Voting에서도 살아남는다.
> 바닥까지 내려온 진짜 차선과 한 덩어리로 붙으면 성분 필터로는 안 걸러져서 추적이 맨 아래에서 잘못 시작됐다
> (1번 처리 추가 전 OK 4/12 → 후 7/12). 정지선/횡단보도 같은 가로선은 ego 경계가 아니므로 뺀다.

### [4] 좌/우 경계 추적 (`track_ego`)

- 화면 **맨 아래 행부터 위로** 한 행씩. 각 행에서 차선 픽셀 구간의 중심 x 목록을 구한다.
- 처음: 중심선(x = W/2) 기준 왼쪽에서 가장 가까운 것 = 왼쪽 경계, 오른쪽에서 가장 가까운 것 = 오른쪽 경계.
  **좌/우 둘 다 찾은 첫 행에서 추적 시작.**
- 이후: 각 경계는 최근 10개 점의 직선 외삽 위치 **±14px** 안의 후보 중 가장 가까운 것만 받는다.
  못 찾으면 외삽값으로 버티되 **연속 40행** 넘게 못 찾으면 "lane lost"로 종료.
- 중심선은 매 행 (왼쪽+오른쪽)/2로 갱신 → 커브를 따라간다.
- 정상 종료: 경계 간격 < 8px(소실점 근처) 또는 중심점이 앙상블 도로 밖(매트 끝/벽).
- 시작 행보다 아래(차선이 화면 옆으로 빠져나간 구간)는 처음 찾은 30행으로 직선 외삽해 화면 끝까지 채운다(0~W-1로 클립).

| 파라미터 | 값 | 의미 |
|---|---|---|
| `ROW_TOL` | 14 | 한 행 사이 경계 이동 허용 px |
| `MAX_GAP` | 40 | 외삽으로 버티는 최대 행 수 |
| `MIN_WIDTH` | 8 | 이보다 좁아지면 추적 종료 |
| `MIN_TRACKED_ROWS` | 60 | 양쪽을 실제로 찾은 행이 이보다 적으면 FAIL |

### [5] 기하 ego da

- 행마다 왼쪽~오른쪽 경계 사이를 채움 → 앙상블 도로를 15px 팽창한 범위로 AND.

> OK 프레임은 이것만으로도 깔끔하다. 이게 라벨의 뼈대다.

### [6] SAM2 보정 (범위 클리핑)

- 프롬프트(ultralytics `SAM`): ego 세로 범위의 25% / 50% / 80% 높이에서
  - 포함 점: 그 행의 ego 중심
  - 제외 점: 왼쪽 경계 − off, 왼쪽 경계 − 2.5·off, 오른쪽 경계 + off, 오른쪽 경계 + 2.5·off (off = max(20, 0.25×차선 폭)), 화면 밖이면 생략
  - → 포함 3 + 제외 최대 12
- **점 전체를 `[[...]]`(1, N, 2) 형태로 넘긴다.** 평평한 (N, 2)로 주면 ultralytics가 점마다 별도 물체로 처리해 포함/제외가 무시된다.
- SAM 마스크 → 15×15 closing(구멍 메우기) → **기하 ego를 9px 팽창한 범위와 AND**.
- 클리핑 결과가 기하 ego 면적의 50% 미만이면 기하 결과로 폴백(`SAM2 weak -> geometric`).
- FAIL 프레임에는 SAM을 돌리지 않고 기하 결과를 그대로 둔다.

> 왜: **SAM2를 단독으로 쓰면 제외 점을 찍어도 노란 선을 넘어 매트 전체로 번진다**(`ego_lane_compare.gif`).
> 노란 선 양쪽이 같은 검은 매트라 SAM은 한 물체로 본다. TwinLiteNet-KMU에서도 같은 이유로 자동 SAM이 실패했다
> (da/ll 경계가 시각적 경계가 아니라 차선이 정의하는 의미적 경계). 범위 안으로 자르면 번짐은 무해해지고,
> 차선 위의 사람/장애물을 잘라내는 효과만 남는다. 제외 점을 6→12개로 늘려도 번짐 자체는 막지 못했다.

### [7] OK / FAIL 판정

| 판정 | 조건 | 2026-10-23 175장 |
|---|---|---|
| OK | 아래에 해당 없음 | **90** |
| FAIL: no lane pair | 좌/우 차선을 동시에 찾은 행이 하나도 없음(한쪽 차선 누락) | 43 |
| FAIL: lane lost | 추적 중 한쪽 경계를 40행 넘게 못 찾음 | 27 |
| FAIL: few rows (n) | 양쪽을 실제로 찾은 행이 60 미만(급커브 등) | 15 |

FAIL은 보수적이다(경계 그림은 맞는데 FAIL인 경우도 있음). FAIL만 사람이 본다.

---

## 4. 출력 형식

```
outputs/ego_labels_<날짜>/
  images/frame_NNNNNN.png    원본
  da_masks/frame_NNNNNN.png  0/255, 최종 ego ∪ ll (차선은 도로 위라 da에 포함 — KMU 관례)
  ll_masks/frame_NNNNNN.png  0/255, [3]까지 정리된 차선(앙상블 ∪ 노란색 HSV)
  status.csv                 frame,status,tracked_rows,final_source
```

`final_source`: `SAM2 in ego` / `SAM2 weak -> geometric` / `geometric`(FAIL 또는 `--no-sam`).
`outputs/ego_labels_*/`는 `.gitignore` 대상(데이터).

---

## 5. 검증 방법 (결과를 믿기 전에 할 것)

1. GIF를 직접 열어 OK 프레임 몇 장의 초록 경계가 실제 지금 차선 양쪽 선 위에 있는지 본다.
2. `status.csv`로 OK/FAIL 분포를 센다. FAIL 사유가 한쪽에 몰리면 해당 단계 파라미터를 의심.
3. 파라미터를 바꿀 땐 `--step 30` 정도로 12장만 돌려 비교(1~2분)한 뒤 전체 실행.
4. **숫자로 품질을 말하려면 사람 GT가 필요하다.** 30~50장을 직접 라벨링해 IoU를 재기 전까지
   "OK = 정답"이라고 가정하지 말 것(TwinLiteNet-KMU 교훈: 커버리지 같은 인상 지표는 자주 틀렸다).

---

## 6. 알려진 한계 / 다음 할 일

- 흐린 주황빛 노란 선은 앙상블과 HSV 둘 다 못 잡는다(예: frame_000295 오른쪽 선) → HSV 범위 확장 여지.
- 흰 T자 선만 있는 구간처럼 좌/우 차선이 아예 없는 장면은 이 방법으로 불가 → 사람 라벨.
- 차선 안 작은 구멍이 SAM 클리핑에서 남는 경우가 있다.
- 사람 검수: `tools/da_trust_review.ipynb`(TwinLiteNet-KMU에서 복사)로 1 = 신뢰 / 2 = SAM 클릭 보정.
  아직 KMU 경로 설정 그대로라 초안 da를 `da_masks/`에서, SAM 가중치를 `models/sam2/sam2.1_b.pt`에서 읽도록 고쳐야 한다.
- 이후 흐름: 검수된 라벨로 작은 모델(TwinLiteNet+ Nano/Small/Medium) 파인튜닝 → LIMO PRO(Jetson)에서 속도 실측
  (`scripts/benchmark_models.py`).

## 7. 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/make_ego_lane_labels.py` | 이 파이프라인 전체 |
| `scripts/make_model_compare_gif.py` | 모델 로더/전처리/확률 변환, ZLUDA 설정, `GpuDutyLimiter` (다른 스크립트가 import) |
| `scripts/make_ensemble_gif.py` | Voting vs Averaging 비교 GIF |
| `scripts/benchmark_models.py` | 모델별 추론 속도 |
| `models/README.md` | 모델/가중치 다운로드, 추론 설정 |
| `docs/LANE_MODEL_WORKFLOW.md` | 참고 프로젝트(TwinLiteNet-KMU) 전체 워크플로우 |
