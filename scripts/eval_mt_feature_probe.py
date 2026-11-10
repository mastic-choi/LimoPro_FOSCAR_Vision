#!/usr/bin/env python3
"""HybridNets 내부 특징에 A/B/P 정보가 있는가 (linear probe).

학습된 가중치를 고정하고, 정답 표지판 박스 자리의 특징(roi_align 3x3)을 뽑아 선형 분류기만 학습한다.
- probe가 높다  -> 특징에 정보는 있음. 검출 분류 머리 / 학습 방식이 못 꺼낸 것 (학습 설정으로 고칠 수 있음)
- probe가 낮다  -> 특징 자체에 정보가 없음 (해상도·구조 쪽을 바꿔야 함)
층: encoder p2(stride 4), encoder p3(stride 8), BiFPN P3(stride 8, 검출 머리가 보는 층), BiFPN P4(stride 16)

사용: python scripts/eval_mt_feature_probe.py . <가중치.pt> [<가중치2.pt> ...]
"""
import os, sys, collections
root = os.path.abspath(sys.argv[1]); ckpts = [os.path.abspath(p) for p in sys.argv[2:]]
sys.path.insert(0, os.path.join(root, "scripts"))
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
import train_multitask as tm
import make_model_compare_gif as m
os.chdir(m.ROOT)

CLS = {tm.CLASSES.index("sign_A"): 0, tm.CLASSES.index("sign_B"): 1, tm.CLASSES.index("parking"): 2}
dev = m.device


def items(split):
    its = tm.seg_items(os.path.join(root, "dataset/synth_signs_v2"), split)
    return [it for it in its if os.path.exists(it["lab"]) and any(int(l.split()[0]) in CLS for l in open(it["lab"]) if l.strip())]


@torch.no_grad()
def extract(model, split):
    dl = torch.utils.data.DataLoader(tm.Items(items(split), False), batch_size=32, num_workers=8, collate_fn=tm.collate)
    feats, ys, sz = collections.defaultdict(list), [], []
    for x, *_, ann, names in dl:
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model.encoder(x)[-4:]
            bif = model.bifpn(tuple(enc[1:]))
            if hasattr(model, "det_skip"):   # 실험 B: 검출 머리가 실제로 받는 P3
                bif = (bif[0] + model.det_skip(enc[1]), *bif[1:])
        boxes = []
        for k in range(len(names)):
            a = ann[k]
            for b in a[a[:, 4] >= 0]:
                c = int(b[4])
                if c in CLS:
                    boxes.append([k, *b[:4].tolist()]); ys.append(CLS[c])
                    sz.append(float(((b[2] - b[0]) * (b[3] - b[1])).sqrt()))
        if not boxes:
            continue
        bx = torch.tensor(boxes, device=dev, dtype=torch.float32)
        for nm, f, s in (("enc_p2", enc[0], 4), ("enc_p3", enc[1], 8), ("bifpn_P3", bif[0], 8), ("bifpn_P4", bif[1], 16)):
            r = roi_align(f.float(), bx, output_size=3, spatial_scale=1 / s, sampling_ratio=2, aligned=True)
            feats[nm].append(r.flatten(1).cpu())
    return {k: torch.cat(v) for k, v in feats.items()}, np.array(ys), np.array(sz)


def probe(Xtr, ytr, Xte_list):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr = ((Xtr - mu) / sd).to(dev); ytr = torch.as_tensor(ytr, device=dev)
    W = torch.nn.Linear(Xtr.shape[1], 3).to(dev)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(400):
        loss = F.cross_entropy(W(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr = (W(Xtr).argmax(1) == ytr).float().mean().item()
        return tr, [F.softmax(W(((X - mu) / sd).to(dev)), 1).cpu().numpy() for X in Xte_list]


def score(P, y):
    ab = y < 2
    return (P.argmax(1) == y).mean(), (P[ab][:, :2].argmax(1) == y[ab]).mean()


for ck in ckpts:
    model = tm.build_model(m, ck).to(dev).eval()
    print(f"\n##### {os.path.relpath(ck, root)}", flush=True)
    tr = extract(model, "train")
    tests = {sp: extract(model, sp) for sp in ("test", "test_venue", "test_bag")}
    print(f"boxes: train {len(tr[1])}, " + ", ".join(f"{k} {len(v[1])}" for k, v in tests.items()), flush=True)
    print(f"{'layer':10s} {'dim':>6s} {'train 3종':>9s} | " + " | ".join(f"{k} 3종 / A/B" for k in tests), flush=True)
    for layer in tr[0]:
        trn, Ps = probe(tr[0][layer], tr[1], [v[0][layer] for v in tests.values()])
        cells = [f"{a:.3f} / {b:.3f}" for (a, b) in (score(P, v[1]) for P, v in zip(Ps, tests.values()))]
        print(f"{layer:10s} {tr[0][layer].shape[1]:6d} {trn:9.3f} | " + " | ".join(cells), flush=True)
    del model; torch.cuda.empty_cache()
