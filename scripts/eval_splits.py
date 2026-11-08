#!/usr/bin/env python3
"""가중치 하나를 데이터셋의 여러 split에서 채점 (train_seg.py의 evaluate와 같은 방식).

사용 예:
  .venv/Scripts/python scripts/eval_splits.py --model hybridnets --weights outputs/train/hybridnets_v3_seed0/best_da.pt \
      --data dataset/ego_v4 --splits test test_venue test_bag --out outputs/train_v4/eval_v3_on_v4.json
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_seg as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(T.ADAPTERS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "test_venue", "test_bag"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import make_model_compare_gif as m
    model, rgb, norm, fwd, _ = T.ADAPTERS[args.model](m)
    if args.model == "yolopv2":
        model = T.load_yolopv2_trainable(args.weights, m.device)
    else:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(m.device).eval()
    res = {}
    for s in args.splits:
        if not os.path.isdir(os.path.join(args.data, s, "images")):
            continue
        loader = torch.utils.data.DataLoader(T.EgoDataset(args.data, s, False, rgb, norm), batch_size=8, num_workers=0)
        with torch.no_grad(), m.GpuDutyLimiter():
            r = T.evaluate(model, fwd, loader, m.device)
        res[s] = {**r["all"], "n": len(loader.dataset)}
        print(s, res[s], flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"model": args.model, "weights": args.weights, "data": args.data, **res}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
