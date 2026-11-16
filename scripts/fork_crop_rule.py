"""갈림길 표지판 분류기의 크롭·전처리 규칙.

로봇 노드(autorace2026 `src/ar_fork_sign`, `crop_rect`·`preprocess`)와 **같아야** 한다. 학습 입력과 실행 입력이 다르면
분류기가 실물에서 떨어진다. 같은 숫자를 `scripts/test_fork_crop_rule.py`와 C++ gtest가 확인한다.

- 크롭: 검출 박스(det_w x det_h 좌표, 보통 640x360) 중심을 원본(img_w x img_h)으로 키우고,
  한 변 = max(w·sx, h·sy)·(1 + 2·margin), 최소 min_px, 영상보다 크면 min(img_w, img_h).
  크기를 유지한 채 영상 안으로 밀어 넣은 정사각형.
- 전처리: 흑백(RGB/BGR 순서대로) → size x size INTER_AREA → CLAHE(clip 2.0, 타일 4x4). uint8.
  분류기 입력은 이것을 /255 한 1x1xHxW.
"""
import math

import cv2


def lround(v):
    """C++ std::lround 와 같다 (0.5 는 0에서 먼 쪽). 파이썬 round 는 짝수 쪽이라 쓰지 않는다."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def crop_rect(box, det_w, det_h, img_w, img_h, margin=0.3, min_px=64):
    """box = (x1, y1, x2, y2) 검출 좌표 → (x, y, w, h) 원본 좌표 정사각형."""
    x1, y1, x2, y2 = box
    sx, sy = img_w / det_w, img_h / det_h
    cx, cy = 0.5 * (x1 + x2) * sx, 0.5 * (y1 + y2) * sy
    side = max((x2 - x1) * sx, (y2 - y1) * sy) * (1.0 + 2.0 * margin)
    side = max(side, float(min_px))
    full = lround(side)
    s = min(full, img_w, img_h)
    if s < full:   # 영상보다 커서 줄였으면 줄인 크기로 다시 가운데
        x, y = lround(cx - s / 2.0), lround(cy - s / 2.0)
    else:
        x, y = lround(cx - side / 2.0), lround(cy - side / 2.0)
    x = min(max(x, 0), img_w - s)
    y = min(max(y, 0), img_h - s)
    return (x, y, s, s)


def preprocess(crop, rgb, size=96):
    """3채널 uint8 크롭 → size x size uint8 흑백 (CLAHE)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(small)
