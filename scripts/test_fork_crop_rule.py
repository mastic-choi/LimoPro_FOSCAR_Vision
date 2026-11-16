#!/usr/bin/env python3
"""fork_crop_rule 이 로봇 노드(autorace2026 src/ar_fork_sign)와 같은 크롭·전처리를 하는지.
숫자는 ar_fork_sign/test/test_fork_sign_logic.cpp 의 CropRect 테스트와 같다.

사용: python -I scripts/test_fork_crop_rule.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fork_crop_rule as r


def test_scales_center_and_margin():
    assert r.crop_rect((300, 100, 320, 120), 640, 360, 1280, 720) == (588, 188, 64, 64)


def test_min_size():
    assert r.crop_rect((300, 100, 305, 105), 640, 360, 1280, 720) == (573, 173, 64, 64)


def test_shifts_inside_at_corner():
    assert r.crop_rect((630, 0, 640, 10), 640, 360, 1280, 720) == (1216, 0, 64, 64)


def test_same_size_image_scale_one():
    assert r.crop_rect((300, 100, 340, 140), 640, 360, 640, 360, min_px=32) == (288, 88, 64, 64)


def test_larger_than_image_shrinks():
    assert r.crop_rect((100, 0, 500, 300), 640, 360, 1280, 720) == (240, 0, 720, 720)


def test_preprocess_shape_and_channel_order():
    img = np.zeros((96, 96, 3), np.uint8); img[:, :] = (0, 0, 255); img[:, :48] = (255, 0, 0)
    g_rgb, g_bgr = r.preprocess(img, rgb=True), r.preprocess(img, rgb=False)
    assert g_rgb.shape == (96, 96) and g_rgb.dtype == np.uint8
    assert g_rgb[:, :48].mean() > g_rgb[:, 48:].mean()   # RGB 로 읽으면 빨강(왼쪽)이 밝다
    assert g_bgr[:, :48].mean() < g_bgr[:, 48:].mean()


if __name__ == "__main__":
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            try: f(); print("PASS", k)
            except Exception as e: fails += 1; print("FAIL", k, repr(e))
    sys.exit(1 if fails else 0)
