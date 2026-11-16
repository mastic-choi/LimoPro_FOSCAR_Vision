#!/usr/bin/env python3
"""train_fork_cls.folded_backbone: 접힌 1채널 첫 conv 가 '흑백 3채널 복사 + ImageNet 정규화 + 원래 conv' 와 같은지.
가장자리 3픽셀은 패딩 값이 달라(정규화 공간 0 ↔ 흑백 0) 비교에서 뺀다. 학습·ONNX·로봇이 모두 접힌 모델을 쓰므로 그것으로 충분하다.

사용: python -I scripts/test_fork_cls_fold.py
"""
import os, sys
import torch, torchvision
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fork_cls as t


def test_fold_conv1_matches_interior():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 96, 96)
    ref = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    mean = torch.tensor(t.MEAN).view(1, 3, 1, 1); std = torch.tensor(t.STD).view(1, 3, 1, 1)
    want = ref.conv1((x.repeat(1, 3, 1, 1) - mean) / std)
    got = t.folded_backbone().m.conv1(x)
    assert got.shape == want.shape
    assert torch.allclose(got[..., 3:-3, 3:-3], want[..., 3:-3, 3:-3], atol=1e-4), (got - want)[..., 3:-3, 3:-3].abs().max()


def test_backbone_is_frozen_and_512():
    b = t.folded_backbone()
    assert not any(p.requires_grad for p in b.parameters())
    with torch.no_grad():
        assert b(torch.rand(3, 1, 96, 96)).shape == (3, 512)


def test_spatial_features_keep_left_right():
    """GAP 는 좌우 반전에 거의 같은 특징을 낸다(ImageNet 학습이 반전 증강). 공간 특징(l4·l3)은 반전하면 달라야 한다."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 96, 96); xf = torch.flip(x, dims=[3])
    dims = {"gap": 512, "l4": 512 * 3 * 3, "l3": 256 * 6 * 6}
    with torch.no_grad():
        for feat, n in dims.items():
            b = t.folded_backbone(feat)
            f, ff = b(x), b(xf)
            assert f.shape == (1, n), (feat, f.shape)
            if feat != "gap":
                rel = (f - ff).norm() / f.norm()
                assert rel > 0.3, (feat, float(rel))


if __name__ == "__main__":
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            try: f(); print("PASS", k)
            except Exception as e: fails += 1; print("FAIL", k, repr(e))
    sys.exit(1 if fails else 0)
