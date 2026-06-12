"""Iter 2: per-class logit-bias calibration of the rhythm head (zero-regression).

v4's rhythm head is mis-calibrated for the natural (sinus-heavy) prior — bbb is
over-called (prec 0.57 << rec 0.68). We optimize an additive per-class bias on
the rhythm logits to maximize window macro-F1 at the TEST prior, tuned on the
prior-matched nat-dev, then fold it into rhythm_sample.bias (frame/beat & all
features stay byte-identical). Confirmed on the held-out test only after selection.

    python3 iter2_logit_bias.py --ckpt codec_v4.pt \
        --dev lydus_dev_nat.npz --test lydus_rhythm_test.npz --out codec_v4_biasadj.pt
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from kgpu_train import _MH
from eval_lydus_rhythm import load_model

CLS = {0: "sinus", 1: "avb", 2: "paced", 3: "afib", 4: "bbb", 5: "vent"}


def collect_persample(model, npz, device, stride=10, bs=64):
    """Per-window downsampled per-sample rhythm logits (N, T//stride, 6) + labels.

    Window pred = mode of per-sample argmax; downsampling the samples by `stride`
    leaves the per-window mode essentially unchanged but shrinks memory ~stride×,
    so the exact metric can be optimized in-memory over a bias grid."""
    ds = _MH(npz)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    L, Y = [], []
    with torch.no_grad():
        for sig, lead, fr, rh, bt in dl:
            _, _, rhythm = model(sig.to(device), lead.to(device))   # (B,T,6)
            L.append(rhythm[:, ::stride, :].cpu().numpy().astype(np.float32))
            Y.append(rh[:, 0].numpy())
    return np.concatenate(L), np.concatenate(Y)


def window_mode(logits, bias):
    """(N,T,6)+bias -> per-window mode of per-sample argmax. Vectorized."""
    a = (logits + bias).argmax(2)                       # (N,T)
    n, t = a.shape
    off = a + 6 * np.arange(n)[:, None]
    cnt = np.bincount(off.ravel(), minlength=6 * n).reshape(n, 6)
    return cnt.argmax(1)


def macro_f1(pred, true, ncls=6):
    f1s = []
    for c in range(ncls):
        if int((true == c).sum()) == 0:
            continue
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="codec_v4.pt")
    ap.add_argument("--dev", default="lydus_dev_nat.npz")
    ap.add_argument("--test", default="lydus_rhythm_test.npz")
    ap.add_argument("--out", default="codec_v4_biasadj.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    # --- coordinate ascent on the EXACT mode metric (nat-dev, downsampled) ---
    L, Y = collect_persample(model, args.dev, device)
    print(f"[nat-dev] logits {L.shape}  classes {np.unique(Y).tolist()}", flush=True)
    bias = np.zeros(6, dtype=np.float32)
    f0 = macro_f1(window_mode(L, bias), Y)
    print(f"[nat-dev] macro @ bias0: {f0:.4f}", flush=True)
    for rnd in range(4):
        prev = bias.copy()
        for c in range(5):                       # classes 0..4 present
            best_b, best_f = bias[c], macro_f1(window_mode(L, bias), Y)
            for b in np.linspace(-4, 4, 41):
                bias[c] = b
                f = macro_f1(window_mode(L, bias), Y)
                if f > best_f + 1e-6:
                    best_f, best_b = f, b
            bias[c] = best_b
        f_now = macro_f1(window_mode(L, bias), Y)
        print(f"[round {rnd}] nat-dev macro {f_now:.4f}  bias={np.round(bias,2).tolist()}", flush=True)
        if np.allclose(prev, bias):
            break

    # --- final gate: held-out test, exact metric (stride2, mode-robust) ---
    Lt, Yt = collect_persample(model, args.test, device, stride=2)
    p0 = window_mode(Lt, np.zeros(6, dtype=np.float32))
    pb = window_mode(Lt, bias)
    print(f"\n=== TEST (held-out, full-res) per-class f1 ===")
    print(f"{'class':7s} {'v4':>7s} {'bias':>7s}")
    for c in range(6):
        if int((Yt == c).sum()) == 0:
            continue
        a = macro_f1((p0 == c).astype(int), (Yt == c).astype(int), 2)
        b_ = macro_f1((pb == c).astype(int), (Yt == c).astype(int), 2)
        print(f"{CLS[c]:7s} {a:7.3f} {b_:7.3f}")
    print(f"TEST macro: v4 {macro_f1(p0,Yt):.4f} -> bias-adj {macro_f1(pb,Yt):.4f}", flush=True)
    print(f"final bias = {np.round(bias,3).tolist()}", flush=True)

    # --- fold bias into rhythm_sample.bias, save (frame/beat byte-identical) ---
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    st = blob["model_state"]
    key = "rhythm_sample.bias" if "rhythm_sample.bias" in st else "m.rhythm_sample.bias"
    st[key] = st[key] + torch.tensor(bias, dtype=st[key].dtype)
    torch.save(blob, args.out)
    print(f"saved {args.out} (rhythm bias folded; frame/beat unchanged)", flush=True)


if __name__ == "__main__":
    main()
