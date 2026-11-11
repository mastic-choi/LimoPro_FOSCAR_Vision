#!/usr/bin/env python3
"""다중 작업 HybridNets 검출 손실.

HybridNets의 FocalLoss(hybridnets/loss.py)와 같은 계산에 두 가지를 더했다. 옵션을 끄면 원래와 같은 값이 나온다.
  1) 무시 박스: 정답 클래스 칸이 -2인 박스(너무 작은 먼 물체 등). 이 박스와 IoU >= ignore_iou로 겹치는 앵커는
     양성으로도 음성으로도 학습하지 않는다(분류 손실 0). 맞혀도 틀려도 벌을 받지 않음.
  2) 학습 중 보정 손실(calibration). 이미지마다 양성 앵커 + 점수가 가장 높은 음성 앵커 k개(양성의 3배, 최소 20)를 모아 계산.
     분류 점수(classification)에만 걸고, 박스 위치(regression)와 분할에는 걸지 않는다. calib 값으로 고른다.
     - "mdca": 클래스별 |평균 점수 - 평균 정답| 의 평균 (Hebbalaguppe et al., CVPR 2022 MDCA를 검출용으로)
     - "sece": 미분 가능한 ECE (soft-binned ECE, Karandikar et al., NeurIPS 2021).
               앵커마다 확신도 p = 가장 높은 클래스 점수, 맞음 = 그 클래스가 정답인지(1/0).
               구간을 딱 자르는 대신 구간 중심과의 거리로 부드럽게 나눠(softmax(-(p-c_m)^2/τ)) ECE를 계산해 미분 가능하게 함.
               ECE = Σ_m (w_m/N)|맞음 비율_m - 평균 확신도_m|
     - "both": 둘의 합
입력 형식은 원래와 같다: classifications (B,A,C) sigmoid 값, regressions (B,A,4), anchors (1,A,4) [y1,x1,y2,x2],
annotations (B,N,5) [x1,y1,x2,y2,cls], cls = -1 빈칸, -2 무시 박스.
반환: (분류 손실, 회귀 손실 x 50, 보정 손실) 각각 shape (1,)
"""
import torch
import torch.nn as nn

ALPHA, GAMMA = 0.25, 2.0


def calc_iou(a, b):
    """a: 앵커 (A,4) [y1,x1,y2,x2], b: 박스 (N,4) [x1,y1,x2,y2] -> (A,N)"""
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(a[:, 3:4], b[:, 2]) - torch.max(a[:, 1:2], b[:, 0])
    ih = torch.min(a[:, 2:3], b[:, 3]) - torch.max(a[:, 0:1], b[:, 1])
    iw, ih = iw.clamp(min=0), ih.clamp(min=0)
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + area - iw * ih
    return iw * ih / ua.clamp(min=1e-8)


def soft_ece(p, correct, bins=15, tau=0.01):
    """미분 가능한 ECE. p (N,) 확신도, correct (N,) 0/1 (상수)."""
    c = torch.linspace(0.5 / bins, 1 - 0.5 / bins, bins, device=p.device, dtype=p.dtype)
    u = torch.softmax(-(p[:, None] - c[None]) ** 2 / tau, 1)          # (N, M) 구간 소속 정도
    w = u.sum(0).clamp(min=1e-6)
    acc = (u * correct[:, None]).sum(0) / w
    conf = (u * p[:, None]).sum(0) / w
    return ((w / p.shape[0]) * (acc - conf).abs()).sum()


class DetLoss(nn.Module):
    def __init__(self, ignore_iou=0.3, calib="none", calib_neg_ratio=3, calib_min_neg=20):
        super().__init__()
        if calib is True:
            calib = "mdca"
        self.ignore_iou, self.calib = ignore_iou, (calib or "none")
        self.neg_ratio, self.min_neg = calib_neg_ratio, calib_min_neg

    def forward(self, classifications, regressions, anchors, annotations):
        B = classifications.shape[0]
        anchor = anchors[0]
        dtype, dev = anchors.dtype, anchors.device
        aw, ah = anchor[:, 3] - anchor[:, 1], anchor[:, 2] - anchor[:, 0]
        acx, acy = anchor[:, 1] + 0.5 * aw, anchor[:, 0] + 0.5 * ah
        cls_l, reg_l, cal_l = [], [], []
        for j in range(B):
            cl = classifications[j].clamp(1e-4, 1 - 1e-4)
            rg = regressions[j]
            ann = annotations[j]
            boxes = ann[ann[:, 4] >= 0]
            ign = ann[ann[:, 4] == -2]
            targets = torch.zeros_like(cl)
            pos = torch.zeros(cl.shape[0], dtype=torch.bool, device=dev)
            if boxes.shape[0]:
                iou = calc_iou(anchor, boxes[:, :4])
                iou_max, iou_arg = iou.max(1)
                asg = boxes[iou_arg]
                big = (asg[:, 2] - asg[:, 0]) * (asg[:, 3] - asg[:, 1]) > 100
                pos = (big & (iou_max >= 0.5)) | (~big & (iou_max >= 0.15))
                targets[pos, asg[pos, 4].long()] = 1
            npos = pos.sum()
            # 무시 박스와 겹치는 앵커(양성이 아닌 것)는 분류 손실에서 뺌
            keep = torch.ones(cl.shape[0], dtype=torch.bool, device=dev)
            if ign.shape[0]:
                keep = ~((calc_iou(anchor, ign[:, :4]).max(1)[0] >= self.ignore_iou) & ~pos)
            af = torch.where(targets == 1, torch.full_like(cl, ALPHA), torch.full_like(cl, 1 - ALPHA))
            fw = torch.where(targets == 1, 1 - cl, cl)
            bce = -(targets * torch.log(cl) + (1 - targets) * torch.log(1 - cl))
            loss = af * fw.pow(GAMMA) * bce * keep[:, None]
            cls_l.append(loss.sum() / npos.to(dtype).clamp(min=1.0))
            if npos > 0:
                a = asg[pos]
                gw, gh = (a[:, 2] - a[:, 0]).clamp(min=1), (a[:, 3] - a[:, 1]).clamp(min=1)
                gcx, gcy = a[:, 0] + 0.5 * (a[:, 2] - a[:, 0]), a[:, 1] + 0.5 * (a[:, 3] - a[:, 1])
                t = torch.stack(((gcy - acy[pos]) / ah[pos], (gcx - acx[pos]) / aw[pos],
                                 torch.log(gh / ah[pos]), torch.log(gw / aw[pos])), 1)
                d = (t - rg[pos]).abs()
                reg_l.append(torch.where(d <= 1 / 9, 0.5 * 9 * d ** 2, d - 0.5 / 9).mean())
            else:
                reg_l.append(torch.zeros((), dtype=dtype, device=dev))
            if self.calib != "none":
                cand = keep & ~pos
                k = int(min(cand.sum().item(), max(self.min_neg, self.neg_ratio * int(npos.item()))))
                sel = pos.clone()
                if k > 0:
                    sc = cl.max(1)[0].detach().masked_fill(~cand, -1)
                    sel[sc.topk(k).indices] = True
                p, y = cl[sel], targets[sel]
                v = torch.zeros((), dtype=dtype, device=dev)
                if self.calib in ("mdca", "both"):
                    v = v + (p.mean(0) - y.mean(0)).abs().mean()
                if self.calib in ("sece", "both"):
                    conf, arg = p.max(1)
                    v = v + soft_ece(conf, y.gather(1, arg[:, None])[:, 0].detach())
                cal_l.append(v)
            else:
                cal_l.append(torch.zeros((), dtype=dtype, device=dev))
        return (torch.stack(cls_l).mean(0, keepdim=True), torch.stack(reg_l).mean(0, keepdim=True) * 50,
                torch.stack(cal_l).mean(0, keepdim=True))
