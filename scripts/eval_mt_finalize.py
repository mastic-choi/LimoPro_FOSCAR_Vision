#!/usr/bin/env python3
"""후보 가중치(best_val_loss.pt, best_score.pt 등)로 학습 끝 처리를 다시 한다: val로 temperature를 맞춰 calib.json을 만들고,
그 temperature로 test 3종을 채점해 test.json을 만든다. train_multitask.py 학습 끝 부분과 같은 데이터·같은 계산이다.
(train_multitask.py는 best.pt로만 이 처리를 하므로, 다른 후보를 릴리즈할 때 쓴다.)

사용: python scripts/eval_mt_finalize.py <ckpt> <출력 접두사> [--device mps] [--calib-only]
  -> <접두사>_calib.json, <접두사>_test.json
val 326장(synth_signs_v2/val)과 콘 val 앞 300장(seed 0 순서), test에는 콘 test + 콘 val 나머지가 들어간다.
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_mt_compare_montage as cm

tm, util = cm.tm, cm.util
mc = tm.mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("prefix")
    ap.add_argument("--seg-data", default="dataset/synth_signs_v2")
    ap.add_argument("--det-data", default="dataset/cones_v1")
    ap.add_argument("--cone-val-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="", help="비우면 cuda가 있으면 cuda, 없으면 cpu (예: mps)")
    ap.add_argument("--calib-only", action="store_true")
    a = ap.parse_args()
    ckpt, pre = os.path.abspath(a.ckpt), os.path.abspath(a.prefix)
    os.chdir(cm.m.ROOT)
    dev = torch.device(a.device) if a.device else cm.m.device
    mk = lambda items: torch.utils.data.DataLoader(tm.Items(items, False), batch_size=a.batch, shuffle=False,
                                                   num_workers=0, collate_fn=tm.collate)

    model = tm.build_model(cm.m, ckpt).to(dev).eval()
    va_ = tm.seg_items(a.seg_data, "val") + tm.det_items(a.det_data, "val", a.cone_val_n, a.seed)
    print("val", len(va_), flush=True)
    vres = tm.evaluate(model, mk(va_), dev, util, False)
    s_, c_, y_ = vres["_raw"]
    T_all, T_cls = mc.fit_T_per_class(s_, y_, c_, tm.CLASSES)
    calib = {"T_all": T_all, "T_cls": T_cls, "how": "p_calibrated = sigmoid(logit(p) / T_cls[class])",
             "val_raw": {k: v for k, v in mc.metrics(s_, y_).items() if k != "rel"},
             "val_T": {k: v for k, v in mc.metrics(mc.apply_T_cls(s_, c_, T_cls, tm.CLASSES), y_).items() if k != "rel"}}
    json.dump(calib, open(pre + "_calib.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("calib:", json.dumps(calib, ensure_ascii=False), flush=True)
    vres.pop("_raw", None)
    res = {"ckpt": os.path.basename(ckpt), "val": vres["all"], "calib": calib}
    if not a.calib_only:
        for split in ("test", "test_venue", "test_bag"):
            items = tm.seg_items(a.seg_data, split)
            if split == "test":
                items += tm.det_items(a.det_data, "test", 0, a.seed) + tm.det_items(a.det_data, "val", 0, a.seed)[a.cone_val_n:]
            res[split] = tm.evaluate(model, mk(items), dev, util, False, T_cls=T_cls)
            res[split].pop("_raw", None)
            print(split, len(items), json.dumps(res[split]["all"], ensure_ascii=False), flush=True)
    json.dump(res, open(pre + "_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
