"""Real multi-layer (frame/rhythm/beat) cache at 500 Hz sample-res, for the
real:synth ratio sweep (v60). Record-level train/val/test split so the sweep's
early-stop val and the final eval never see a training record.

No single real recording has all three expert layers, so each window
supervises only the layer(s) it actually has GT for; the rest are IGN=255
(data-driven masking — NOT the artificial "rhythm/beat = synth-only" choice).

Layer sources:
  * FRAME  — LUDB/QTDB/ISP (v57_500hz cache). These supervise frame only;
             rhythm/beat remain IGN unless explicitly enabled as weak labels.
  * BEAT   — MIT-BIH DS1 (train) at expert R-peaks (+-40 ms), AAMI->our scheme.
             MIT-BIH also has rhythm aux, so rhythm is supervised too; frame IGN.
  * RHYTHM — AFDB / VFDB / NSRDB aux episodes; beat + frame IGN.

TEST records (held out, used by eval_multihead_real_500): MIT-BIH DS2, plus the
afdb/vfdb/nsrdb TEST lists below. VAL records feed early-stop only.

    python scripts/build_real_multilayer_cache.py --split train --out data/cache/real_ml_train.npz
    python scripts/build_real_multilayer_cache.py --split val   --out data/cache/real_ml_val.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.signal as ss
import wfdb

FS, WS, IGN = 500, 5000, 255
S_QRS, S_PACED = 2, 4
R_SINUS, R_AVB, R_PACED, R_AFIB, R_BBB, R_VENT = range(6)
B_NONE, B_SINUS, B_VPC, B_PACED, B_FUSION, B_UNK = range(6)
RTOL = 20                                       # +-40 ms around an R-peak

BEAT_SYM = {"N": B_SINUS, "L": B_SINUS, "R": B_SINUS, "e": B_SINUS, "j": B_SINUS,
            "A": B_SINUS, "a": B_SINUS, "J": B_SINUS, "S": B_SINUS, "n": B_SINUS,
            "V": B_VPC, "E": B_VPC, "F": B_FUSION, "f": B_FUSION,
            "/": B_PACED, "Q": B_UNK, "?": B_UNK}
AUX = {"(AFIB": R_AFIB, "(AFL": R_AFIB, "(N": R_SINUS, "(NSR": R_SINUS,
       "(VT": R_VENT, "(VFL": R_VENT, "(VF": R_VENT, "(B": R_VENT}

# ---- record-level splits (deterministic) ----
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS1_VAL = [108, 116, 209, 223]
DS1_TRAIN = [r for r in DS1 if r not in DS1_VAL]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]   # held-out beat test (matches eval)
AFDB = ["04015","04043","04048","04126","04746","04908","04936","05091","05121",
        "05261","06426","06453","06995","07162","07859","07879","07910","08215",
        "08219","08378","08405","08434","08455"]
VFDB = ["418","419","420","421","422","423","424","425","426","427","428","429",
        "430","602","605","607","609","610","611","612","614","615"]
NSRDB = ["16265","16272","16273","16420","16483","16539","16773","16786","16795",
         "17052","17453","18177","18184","19088","19090","19093","19140","19830"]


def _split(lst, n_val, n_test):
    val = lst[:n_val]; test = lst[n_val:n_val + n_test]; train = lst[n_val + n_test:]
    return train, val, test


AFDB_TR, AFDB_VAL, AFDB_TE = _split(AFDB, 3, 5)
VFDB_TR, VFDB_VAL, VFDB_TE = _split(VFDB, 3, 5)
NSRDB_TR, NSRDB_VAL, NSRDB_TE = _split(NSRDB, 3, 3)


def _resample_500(sig, fs):
    if fs == FS:
        return sig.astype(np.float32)
    if FS % fs == 0:
        return ss.resample_poly(sig, FS // fs, 1).astype(np.float32)
    if fs % FS == 0:
        return ss.decimate(sig, fs // FS, zero_phase=True).astype(np.float32)
    return ss.resample(sig, int(round(len(sig) * FS / fs))).astype(np.float32)


def _znorm16(s):
    return ((s - s.mean()) / (s.std() + 1e-6)).astype(np.float16)


def _lead_ii(rec):
    names = [s.upper() for s in rec.sig_name]
    for c in ("II", "MLII"):
        if c in names:
            return names.index(c)
    return 0


def _rhythm_per_sample(ann, scale, n, default):
    """Per-sample rhythm from aux episodes; returns uint8 array or None.

    EVERY rhythm-change annotation (aux_note starting with '(') is a boundary.
    Episodes whose token is not in our 6-class AUX map are filled with IGN, so
    rhythms we don't model (NOISE, asystole, paced, idioventricular, SVTA,
    bi/trigeminy, sinus-brady, 2nd-deg block, ...) are MASKED OUT instead of
    silently inheriting the previous recognized episode's class. The region
    before the first episode is `default` if given, else IGN.
    """
    events = []  # (sample, class_id_or_IGN)
    for s, aux in zip(ann.sample, ann.aux_note):
        if not aux:
            continue
        tok = aux.strip().rstrip("\x00").split()[0]
        if not tok.startswith("("):
            continue                          # not a rhythm-change annotation
        events.append((int(s * scale), AUX.get(tok, IGN)))
    if not events and default is None:
        return None
    arr = np.full(n, default if default is not None else IGN, np.uint8)
    for i, (s, r) in enumerate(events):
        hi = events[i + 1][0] if i + 1 < len(events) else n
        arr[max(0, s):min(n, hi)] = r
    return arr


def _beat_records(root, rids, per_rec, *, default_rhythm=None):
    """MIT-BIH: beat (R-peak +-RTOL) + rhythm from aux when present; frame IGN."""
    out = []
    d = Path(root) / "mitdb"
    for rid in rids:
        rp = str(d / f"{rid}")
        try:
            rec = wfdb.rdrecord(rp); ann = wfdb.rdann(rp, "atr")
        except Exception:
            continue
        fs = int(rec.fs); scale = FS / fs
        sig = _resample_500(rec.p_signal[:, _lead_ii(rec)].astype(np.float64), fs)
        rps = _rhythm_per_sample(ann, scale, len(sig), default_rhythm)
        beats = [(int(s * scale), BEAT_SYM[sym]) for s, sym in zip(ann.sample, ann.symbol)
                 if sym in BEAT_SYM]
        nwin = min(per_rec, len(sig) // WS)
        for w in range(nwin):
            lo = w * WS
            seg = sig[lo:lo + WS]
            if seg.std() < 1e-4:
                continue
            beat = np.zeros(WS, np.uint8)            # NONE elsewhere
            for s, cls in beats:
                if lo <= s < lo + WS:
                    a = max(0, s - lo - RTOL); b = min(WS, s - lo + RTOL + 1)
                    beat[a:b] = cls
            rhythm = rps[lo:lo + WS].copy() if rps is not None else np.full(WS, IGN, np.uint8)
            frame = np.full(WS, IGN, np.uint8)
            out.append((_znorm16(seg), frame, rhythm, beat, 1))
    return out


def _rhythm_records(root, sub, rids, default, per_class):
    """afdb/vfdb/nsrdb: rhythm only (beat + frame IGN). Cap per class per record."""
    out = []
    d = Path(root) / sub
    for rid in rids:
        rp = str(d / rid)
        try:
            rec = wfdb.rdrecord(rp); ann = wfdb.rdann(rp, "atr")
        except Exception:
            continue
        fs = int(rec.fs); scale = FS / fs
        sig = _resample_500(rec.p_signal[:, 0].astype(np.float64), fs)
        rps = _rhythm_per_sample(ann, scale, len(sig), default)
        if rps is None:
            continue
        kept = {}
        for w in range(len(sig) // WS):
            lo = w * WS
            seg = sig[lo:lo + WS]
            if seg.std() < 1e-4:
                continue
            rseg = rps[lo:lo + WS]
            valid = rseg[rseg != IGN]
            if valid.size == 0:
                continue                     # no real rhythm GT in this window
            dom = int(np.bincount(valid).argmax())
            if kept.get(dom, 0) >= per_class:
                continue
            kept[dom] = kept.get(dom, 0) + 1
            out.append((_znorm16(seg),
                        np.full(WS, IGN, np.uint8), rseg.copy().astype(np.uint8),
                        np.full(WS, IGN, np.uint8), 1))
    return out


def _ludb_windows(npz_path, *, derive_rhythm_beat: bool = False):
    """v57_500hz frame cache -> frame real.

    Rhythm/beat are IGN by default because frame delineation is not expert
    rhythm/beat supervision. ``derive_rhythm_beat=True`` keeps the old weak
    label behaviour for explicit ablations only.
    """
    b = np.load(npz_path)
    sig, lbl, lead = b["signals"], b["labels"], b["lead_ids"]
    S_P, S_T = 1, 3
    out = []
    for i in range(len(sig)):
        fr = lbl[i].astype(np.uint8).copy()
        # Per-lead PARTIAL annotation: a window with P or T labelled but NO QRS is
        # physiologically impossible (every beat has a QRS) -> this lead's QRS was
        # simply not annotated. Don't supervise frame here at all (mask the whole
        # window to IGN); otherwise the model is penalised for correctly predicting
        # the un-annotated QRS. On LUDB val this is ~2% of windows (the worst-case
        # "broken label" diagnosis, 2026-06-16); ~0.16% of train windows.
        if ((fr == S_P).any() or (fr == S_T).any()) and not (fr == S_QRS).any():
            fr[:] = IGN
            out.append((_znorm16(sig[i].astype(np.float32)), fr,
                        np.full(WS, IGN, np.uint8), np.full(WS, IGN, np.uint8), int(lead[i])))
            continue
        # AFib / flutter / junctional: no P-wave is annotated, so the atrial /
        # baseline region (which carries fibrillatory f-waves, not true
        # isoelectric line) is filled as OTHER(0) by the delineation cache.
        # That is NOT real iso GT — mask it to IGN so the frame head is never
        # penalised for whatever it emits there (P is undefined in AFib). Detect
        # source-agnostically as "QRS present but ZERO P in the window"; QRS/T
        # spans stay supervised. On LUDB this hits exactly the afib/flutter
        # records (0 P-onsets) and never a sinus window (which has P labelled).
        if (fr == S_QRS).any() and not (fr == S_P).any():
            fr[fr == 0] = IGN
        if derive_rhythm_beat:
            beat = np.zeros(WS, np.uint8)
            beat[(fr == S_QRS) | (fr == S_PACED)] = B_SINUS
            rhythm = np.full(WS, R_SINUS, np.uint8)
        else:
            beat = np.full(WS, IGN, np.uint8)
            rhythm = np.full(WS, IGN, np.uint8)
        out.append((_znorm16(sig[i].astype(np.float32)), fr, rhythm, beat, int(lead[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--root", default="data/rare_arrhythmia")
    ap.add_argument("--ludb-train", default="data/cache/v57_500hz/train.npz")
    ap.add_argument("--ludb-val", default="data/cache/v57_500hz/ludb_val.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--derive-ludb-rhythm-beat",
        action="store_true",
        help="legacy weak-label mode: derive rhythm=sinus and beat from LUDB frame labels",
    )
    args = ap.parse_args()

    rows = []
    if args.split == "train":
        rows += _ludb_windows(
            args.ludb_train, derive_rhythm_beat=args.derive_ludb_rhythm_beat)
        rows += _beat_records(args.root, DS1_TRAIN, per_rec=80)
        rows += _rhythm_records(args.root, "afdb", AFDB_TR, None, per_class=80)
        rows += _rhythm_records(args.root, "vfdb", VFDB_TR, None, per_class=80)
        rows += _rhythm_records(args.root, "nsrdb", NSRDB_TR, R_SINUS, per_class=60)
    else:
        rows += _ludb_windows(
            args.ludb_val, derive_rhythm_beat=args.derive_ludb_rhythm_beat)
        rows += _beat_records(args.root, DS1_VAL, per_rec=60)
        rows += _rhythm_records(args.root, "afdb", AFDB_VAL, None, per_class=60)
        rows += _rhythm_records(args.root, "vfdb", VFDB_VAL, None, per_class=60)
        rows += _rhythm_records(args.root, "nsrdb", NSRDB_VAL, R_SINUS, per_class=40)

    N = len(rows)
    sigs = np.empty((N, WS), np.float16)
    frame = np.empty((N, WS), np.uint8); rhythm = np.empty((N, WS), np.uint8)
    beat = np.empty((N, WS), np.uint8); leads = np.empty(N, np.int64)
    for i, (s, fr, rh, bt, ld) in enumerate(rows):
        sigs[i], frame[i], rhythm[i], beat[i], leads[i] = s, fr, rh, bt, ld

    # supervision coverage report
    def cov(ch):
        return round(float((ch != IGN).any(axis=1).mean()), 3)
    print(f"[{args.split}] N={N}  frame-sup {cov(frame)}  rhythm-sup {cov(rhythm)}  beat-sup {cov(beat)}")
    for nm, ch, k in [("rhythm", rhythm, 6), ("beat", beat, 6)]:
        vis = ch[ch != IGN]
        d = {i: int((vis == i).sum()) for i in range(k)}
        print(f"  {nm} sample dist (ex-IGN): {d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, signals=sigs, frame=frame, rhythm=rhythm, beat=beat, lead_ids=leads)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.0f} MB, {N} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
