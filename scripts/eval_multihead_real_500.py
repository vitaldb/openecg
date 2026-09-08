"""v59b (500 Hz sample-res MH) — REAL-data eval of the rhythm + beat heads.

The synth-val numbers are on-distribution (memorised generator patterns).
The honest test of "synth-trained rhythm/beat transfers to real patients" is
to read the per-sample heads on real annotated records:

  * BEAT  — MIT-BIH Arrhythmia DS2 (gold beat benchmark). At each expert
            R-peak, read the predicted beat class from the per-sample output
            at that sample (mode over a ±tol-sample window for robustness);
            compare to the AAMI symbol mapped to our BEAT scheme:
              N/L/R/e/j/A/a/J/S/n -> sinus,  V/E -> vpc,  F/f -> fusion,
              / -> paced,  Q/? -> unknown.
  * RHYTHM — real windows with a known rhythm: AFDB '(AFIB' episodes -> afib,
            MIT-BIH NSR-DB / mitdb '(N' -> sinus, VFDB/CUDB '(VT|(VFL|(VF' ->
            ventricular. Majority-vote the per-sample rhythm head over the
            window; compare to the window's true rhythm.

This differs from the 250 Hz version: the model is sample-resolution at
500 Hz, so the output length == window length (5000) and there is NO patch
indexing — we read the head at the sample directly. All signals resampled to
500 Hz, lead II (or first lead).

    python scripts/eval_multihead_real_500.py \
        --ckpt data/checkpoints/v59b_multihead_500.pt --root data/rare_arrhythmia
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch
import wfdb

from openecg.dsp import rank_normalize
from openecg.stage2.model import load_model_from_ckpt

FS = 500
WINDOW = 5000          # 10 s @ 500 Hz
RTOL = 10              # ±10 samples (20 ms) mode window around an R-peak
RHYTHM = ["sinus", "avb", "paced", "afib", "bbb", "vent"]
BEAT = ["none", "sinus", "vpc", "paced", "fusion", "unknown"]
R_SINUS, R_AVB, R_PACED, R_AFIB, R_BBB, R_VENT = range(6)
B_NONE, B_SINUS, B_VPC, B_PACED, B_FUSION, B_UNK = range(6)

BEAT_SYM = {"N": B_SINUS, "L": B_SINUS, "R": B_SINUS, "e": B_SINUS, "j": B_SINUS,
            "A": B_SINUS, "a": B_SINUS, "J": B_SINUS, "S": B_SINUS, "n": B_SINUS,
            "V": B_VPC, "E": B_VPC, "F": B_FUSION, "f": B_FUSION,
            "/": B_PACED, "Q": B_UNK, "?": B_UNK}
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]


def _load(ck, device):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    st = blob.get("model_state", blob)
    if any(k.startswith("m.") for k in st):
        blob["model_state"] = {(k[2:] if k.startswith("m.") else k): v for k, v in st.items()}
        ck = str(ck) + ".stripped.pt"; torch.save(blob, ck)
    return load_model_from_ckpt(ck, device=device)


def _resample_500(sig, fs):
    if fs == FS:
        return sig.astype(np.float32)
    if FS % fs == 0:                       # upsample integer factor
        return ss.resample_poly(sig, FS // fs, 1).astype(np.float32)
    if fs % FS == 0:                       # downsample integer factor
        return ss.decimate(sig, fs // FS, zero_phase=True).astype(np.float32)
    n = int(round(len(sig) * FS / fs))
    return ss.resample(sig, n).astype(np.float32)


def prf(pred, true, n, names):
    pred = np.asarray(pred); true = np.asarray(true)
    rows = {}
    for c in range(n):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        sup = int((true == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows[names[c]] = {"precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f, 4), "support": sup}
    return rows


def _unpack(out):
    """(beat, rhythm) from MH forward — 3-tuple (frame,beat,rhythm) for the
    conv-tok sample-res MH, or 4-tuple (frame,reg,beat,rhythm) for old."""
    if len(out) == 3:
        return out[1], out[2]
    return out[2], out[3]


@torch.no_grad()
def _infer(model, sig, device):
    """Return (beat[L], rhythm[L]) per-sample argmax for one window."""
    x = torch.from_numpy(rank_normalize(sig.astype(np.float32))).unsqueeze(0).to(device)
    beat, rhythm = _unpack(model(x, torch.zeros(1, dtype=torch.long, device=device)))
    return beat.argmax(-1)[0].cpu().numpy(), rhythm.argmax(-1)[0].cpu().numpy()


def _mode_around(arr, idx, tol):
    lo = max(0, idx - tol); hi = min(len(arr), idx + tol + 1)
    seg = arr[lo:hi]
    if seg.size == 0:
        return int(arr[min(idx, len(arr) - 1)])
    vals, cnts = np.unique(seg, return_counts=True)
    return int(vals[np.argmax(cnts)])


def eval_beat(model, root, device):
    """MIT-BIH DS2 real beats -> predicted beat class at each R-peak sample.

    Many beats fall in the same 10 s window; we group beats by window and run
    one forward per window (≈12× fewer inferences than per-beat)."""
    pred_all, true_all = [], []
    d = Path(root) / "mitdb"
    for rid in DS2:
        rp = str(d / f"{rid}")
        try:
            rec = wfdb.rdrecord(rp); ann = wfdb.rdann(rp, "atr")
        except Exception:
            continue
        fs = int(rec.fs)
        leadidx = 0
        names = [s.upper() for s in rec.sig_name]
        for cand in ("II", "MLII"):
            if cand in names:
                leadidx = names.index(cand); break
        sig = _resample_500(rec.p_signal[:, leadidx].astype(np.float64), fs)
        scale = FS / fs
        # bucket beats by window index
        by_win = {}
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym not in BEAT_SYM:
                continue
            r500 = int(samp * scale)
            w = r500 // WINDOW
            lo = w * WINDOW
            if lo + WINDOW > len(sig):
                continue
            by_win.setdefault(w, []).append((r500 - lo, BEAT_SYM[sym]))
        for w, beats in by_win.items():
            lo = w * WINDOW
            beat_s, _ = _infer(model, sig[lo:lo + WINDOW], device)
            for off, t in beats:
                pred_all.append(_mode_around(beat_s, off, RTOL)); true_all.append(t)
        print(f"  beat rec {rid}: {len(by_win)} win, {len(pred_all)} beats", flush=True)
    return prf(pred_all, true_all, 6, BEAT)


def eval_rhythm(model, root, device, max_win=80, test_only=True):
    """Real rhythm windows from aux_note: afib (afdb), sinus (nsrdb/mitdb),
    ventricular (vfdb/cudb). Majority-vote per-sample rhythm over the window.

    Caps windows kept *per rhythm class per record* at `max_win` so long
    Holter records (nsrdb 24 h, afdb 10 h) don't dominate / blow up runtime.

    test_only=True restricts to the held-out TEST records (afdb/vfdb/nsrdb TE
    lists + MIT-BIH DS2) so the v60 sweep's training/val records are excluded.
    """
    AUX = {"(AFIB": R_AFIB, "(AFL": R_AFIB, "(N": R_SINUS, "(NSR": R_SINUS,
           "(VT": R_VENT, "(VFL": R_VENT, "(VF": R_VENT, "(B": R_VENT}
    if test_only:
        from scripts.build_real_multilayer_cache import AFDB_TE, VFDB_TE, NSRDB_TE, DS2
        rec_filter = {"afdb": set(AFDB_TE), "vfdb": set(VFDB_TE),
                      "nsrdb": set(NSRDB_TE), "mitdb": set(str(r) for r in DS2)}
    else:
        rec_filter = None
    sources = [("afdb", None), ("nsrdb", R_SINUS), ("vfdb", None), ("mitdb", None)]
    pred_all, true_all = [], []
    n_win = 0
    for sub, default in sources:
        d = Path(root) / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.dat")):
            rid = f.stem
            if rec_filter is not None and rid not in rec_filter.get(sub, set()):
                continue
            rp = str(d / rid)
            try:
                rec = wfdb.rdrecord(rp); ann = wfdb.rdann(rp, "atr")
            except Exception:
                continue
            fs = int(rec.fs)
            sig = _resample_500(rec.p_signal[:, 0].astype(np.float64), fs)
            scale = FS / fs
            events = []
            for s, aux in zip(ann.sample, ann.aux_note):
                if not aux:
                    continue
                tok = aux.strip().rstrip("\x00").split()[0]
                if tok in AUX:
                    events.append((int(s * scale), AUX[tok]))
            if not events and default is None:
                continue
            cur = default if default is not None else (events[0][1] if events else None)
            if cur is None:
                continue
            ev_i = 0
            kept = {}                          # per-class window count this record
            for w in range(len(sig) // WINDOW):
                lo = w * WINDOW
                while ev_i < len(events) and events[ev_i][0] <= lo:
                    cur = events[ev_i][1]; ev_i += 1
                if w % 6 != 0:                 # subsample to keep it fast
                    continue
                if kept.get(cur, 0) >= max_win:
                    continue
                seg = sig[lo:lo + WINDOW]
                if seg.std() < 1e-4:
                    continue
                _, rhythm_s = _infer(model, seg, device)
                vals, cnts = np.unique(rhythm_s, return_counts=True)
                pred_all.append(int(vals[np.argmax(cnts)])); true_all.append(cur)
                kept[cur] = kept.get(cur, 0) + 1; n_win += 1
    print(f"  rhythm eval windows: {n_win}")
    return prf(pred_all, true_all, 6, RHYTHM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/rare_arrhythmia")
    ap.add_argument("--out", default="out/v59b_real_eval.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load(args.ckpt, device); model.eval()

    print("=== BEAT — MIT-BIH DS2 (real beats) ===")
    beat_rows = eval_beat(model, args.root, device)
    for k, v in beat_rows.items():
        print(f"  {k:7s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")

    print("\n=== RHYTHM — real (afdb/nsrdb/vfdb/mitdb aux) ===")
    rhythm_rows = eval_rhythm(model, args.root, device)
    for k, v in rhythm_rows.items():
        print(f"  {k:6s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"beat_real": beat_rows, "rhythm_real": rhythm_rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
