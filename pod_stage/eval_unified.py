"""Per-sample frame + beat macro-F1 on a cache, for BOTH 2-head and unified
(8-class) models — the regression test for merging frame+beat into one head.

A unified model's frame head emits 8 classes
(0 other,1 P,2 T,3 sinus,4 vpc,5 paced,6 fusion,7 unknown); we map it back to
the 4-class frame (other/P/QRS/T) and 6-class beat (none/sinus/vpc/paced/fusion/
unknown) and score the SAME per-sample metric as the 2-head model.

    python3 eval_unified.py B_s0.pt A_s0.pt codec_v5.pt --cache real_ml_v2_val.npz
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from kgpu_train import _MH
from eval_lydus_rhythm import load_model

IGN = 255
# unified frame-class (8) -> 4-class frame target space (0 other,1 P,2 QRS,3 T)
U2FRAME = {0: 0, 1: 1, 2: 3, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
# unified -> 6-class beat (0 none,1 sinus,2 vpc,3 paced,4 fusion,5 unknown)
U2BEAT = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}


def macro_f1(pred, true, classes):
    f1s = []
    for c in classes:
        if int((true == c).sum()) == 0:
            continue
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(ckpt, cache, device):
    model = load_model(ckpt, device)
    is_unified = model.head_sample.out_features == 8
    dl = DataLoader(_MH(cache), batch_size=64, shuffle=False, num_workers=2)
    fp_, ft_, bp_, bt_ = [], [], [], []
    with torch.no_grad():
        for sig, lead, fr, rh, bt in dl:
            frame, beat, _ = model(sig.to(device), lead.to(device))
            fa = frame.argmax(-1).cpu().numpy()
            if is_unified:
                fpred = np.vectorize(U2FRAME.get)(fa)
                bpred = np.vectorize(U2BEAT.get)(fa)
            else:
                fpred = fa
                bpred = beat.argmax(-1).cpu().numpy()
            fr = fr.numpy(); bt = bt.numpy()
            mf = fr != IGN
            fp_.append(fpred[mf]); ft_.append(fr[mf])
            mb = (bt != IGN) & (bt != 0)          # beat scored on real (QRS) beats only
            bp_.append(bpred[mb]); bt_.append(bt[mb])
    fp_ = np.concatenate(fp_); ft_ = np.concatenate(ft_)
    bp_ = np.concatenate(bp_); bt_ = np.concatenate(bt_)
    frame_f1 = macro_f1(fp_, ft_, [1, 2, 3])              # P/QRS/T
    beat_f1 = macro_f1(bp_, bt_, [1, 2, 3, 4, 5])         # sinus/vpc/paced/fusion/unknown
    beat_fus = macro_f1(bp_, bt_, [4])                    # fusion only
    return is_unified, frame_f1, beat_f1, beat_fus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cache", default="real_ml_v2_val.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== per-sample frame/beat on {args.cache} ===")
    print(f"{'ckpt':22s} {'type':8s} {'frame_f1':>9s} {'beat_f1':>8s} {'fusion':>7s}")
    for ck in args.ckpts:
        try:
            uni, ff, bf, fus = evaluate(ck, args.cache, device)
            print(f"{ck:22s} {'unified' if uni else '2-head':8s} {ff:9.4f} {bf:8.4f} {fus:7.3f}")
        except Exception as e:
            print(f"{ck:22s} ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
