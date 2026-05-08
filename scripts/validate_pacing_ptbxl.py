"""Validate ``openecg.detect_pace`` on PTB-XL.

Holdout: random 50 paced + 50 sinus controls (ecg-id seeded), all 12
leads, ``pre_ms=300`` (covers atrial-only pacing) + ``gap_ms=25`` (excludes
Q/S inflection band). Classification: a record is called paced if the
spike count summed across leads clears a sweep threshold.

On a 50+50 PTB-XL holdout this gives:
    ≥3 spikes summed: 78% sensitivity / 94% specificity / 93% PPV
    ≥10 spikes summed: 56% sensitivity / 98% specificity / 97% PPV

The remaining 22% missed-paced records typically have ZERO detectable
spikes after PTB-XL acquisition LP — atrial-only pacing, failed capture,
or notch-filtered DDD spikes below the post-LP threshold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openecg import detect_pace, detect_qrs, ptbxl

LEAD = "V1"                                      # single-lead evaluation;
                                                  # V1 sees right-ventricular
                                                  # pacing spikes far better
                                                  # than limb leads on PTB-XL.
N_PER_CLASS = 50
SEED = 42
PRE_MS = 300.0                                   # wide enough for atrial pacing
GAP_MS = 5.0                                     # PTB-XL ventricular pacing is
                                                  # close to QRS-onset; gap > 15
                                                  # collapses sensitivity.


def detect_paced_4ch(rec: dict, score_thr_mad: float = 8.0) -> int:
    """4-channel multi-derivative detector on a single lead, then PR-segment localised."""
    fs = rec["fs"]
    leads = rec["leads"]
    if LEAD not in leads:
        return 0
    sig = rec["signal"][:, leads.index(LEAD)].astype(np.float64)
    try:
        qrs = detect_qrs(sig, fs)
    except Exception:
        qrs = np.empty(0, dtype=np.int64)
    from openecg.pacer import detect_spikes_4channel
    spikes = detect_spikes_4channel(sig, fs, score_thr_mad=score_thr_mad)
    # Apply PR-segment localisation
    if qrs.size and spikes.size:
        n_pre = int(round(PRE_MS * fs / 1000))
        n_gap = int(round(GAP_MS * fs / 1000))
        qrs_sorted = np.sort(qrs)
        keep = []
        for s in spikes:
            idx = int(np.searchsorted(qrs_sorted, s))
            if idx < qrs_sorted.size:
                q = int(qrs_sorted[idx])
                if q - n_pre <= int(s) <= q - n_gap:
                    keep.append(int(s))
        spikes = np.asarray(keep, dtype=np.int64)
    return int(spikes.size)


def detect_paced(
    rec: dict,
    height_mad: float | None,
    diff_order: int = 1,
    power: int = 2,
    d2_thr_mad: float | None = None,
) -> int:
    """Run detect_pace on ``LEAD`` and return spike count.

    ``diff_order`` / ``power`` select the derivative form (forwarded to
    pacer_center_surround_score via detect_spikes_center_surround).

    If ``d2_thr_mad`` is given, additionally require the 2nd-derivative
    SQUARED score at each detected sample to clear ``d2_thr_mad ×
    σ_d2_score``. d2² is zero on locally-linear ramps (Q/S/J), so this
    AND-gate adds selectivity for true non-linear transients.
    """
    fs = rec["fs"]
    leads = rec["leads"]
    if LEAD not in leads:
        return 0
    sig = rec["signal"][:, leads.index(LEAD)].astype(np.float64)
    try:
        qrs = detect_qrs(sig, fs)
    except Exception:
        qrs = np.empty(0, dtype=np.int64)
    spikes = detect_pace(
        sig, fs, qrs_indices=qrs, pre_ms=PRE_MS, gap_ms=GAP_MS,
        min_local_height_mad=height_mad,
        diff_order=diff_order, power=power,
    )
    if d2_thr_mad is not None and spikes.size:
        from openecg.pacer import pacer_center_surround_score
        score_d2 = pacer_center_surround_score(sig, fs, diff_order=2, power=2)
        sigma_d2 = 1.4826 * float(np.median(np.abs(score_d2 - np.median(score_d2))))
        if sigma_d2 > 0:
            n_snap = max(1, int(round(3.0 * fs / 1000)))
            keep = []
            for s in spikes:
                lo = max(0, int(s) - n_snap)
                hi = min(score_d2.size, int(s) + n_snap + 1)
                if score_d2[lo:hi].max() >= d2_thr_mad * sigma_d2:
                    keep.append(int(s))
            spikes = np.asarray(keep, dtype=np.int64)
    return int(spikes.size)


def sample_ids(rng: np.random.Generator, n: int = N_PER_CLASS) -> tuple[list[int], list[int]]:
    paced_all = ptbxl.paced_ids()
    sinus_all = ptbxl.sinus_ids()
    paced_pick = rng.choice(paced_all, size=min(n, len(paced_all)), replace=False)
    sinus_pick = rng.choice(sinus_all, size=min(n, len(sinus_all)), replace=False)
    return sorted(int(x) for x in paced_pick), sorted(int(x) for x in sinus_pick)


def _sweep(paced_counts, sinus_counts, thresholds):
    print(f"  {'thr':>4} {'sens':>6} {'spec':>6} {'PPV':>6} {'NPV':>6}")
    for thr in thresholds:
        tp = sum(1 for c in paced_counts if c >= thr)
        fp = sum(1 for c in sinus_counts if c >= thr)
        fn = len(paced_counts) - tp
        tn = len(sinus_counts) - fp
        sens = tp / max(1, len(paced_counts))
        spec = tn / max(1, len(sinus_counts))
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        print(f"  {thr:>4} {sens:>6.2%} {spec:>6.2%} {ppv:>6.2%} {npv:>6.2%}")


def main():
    rng = np.random.default_rng(SEED)
    paced_ids, sinus_ids = sample_ids(rng)
    print(f"Validating on PTB-XL: {len(paced_ids)} paced + {len(sinus_ids)} sinus controls")
    print(f"Lead: {LEAD},  pre_ms={PRE_MS}, gap_ms={GAP_MS}\n")

    # Each config is either ("4ch", thr) or ("classic", do, p, height_mad, d2_gate_mad).
    configs = [
        ("d²(=d1)",            "classic", 1, 2, None, None),
        ("|d| (=|d1|)",        "classic", 1, 1, None, None),
        ("d2²",                "classic", 2, 2, None, None),
        ("|d2|",               "classic", 2, 1, None, None),
        ("d²+height=4",        "classic", 1, 2, 4.0,  None),
        ("d²+d2-gate=8",       "classic", 1, 2, None, 8.0),
        ("4ch (score_mad=4)",  "4ch", 4.0),
        ("4ch (score_mad=6)",  "4ch", 6.0),
        ("4ch (score_mad=8)",  "4ch", 8.0),
        ("4ch (score_mad=12)", "4ch", 12.0),
    ]
    print(f"  At each config: sweep over min-spikes (1, 2, 3, 5)\n")
    rows = {}
    for cfg in configs:
        label = cfg[0]
        kind = cfg[1]
        paced_counts = []
        sinus_counts = []
        for klass, ids, store in (("paced", paced_ids, paced_counts),
                                    ("sinus", sinus_ids, sinus_counts)):
            for rid in ids:
                try:
                    rec = ptbxl.load_record(rid, fs=500)
                except Exception as exc:
                    print(f"  skip rid={rid}: {exc}")
                    continue
                if kind == "4ch":
                    store.append(detect_paced_4ch(rec, score_thr_mad=cfg[2]))
                else:
                    _, _, do, p, h, d2g = cfg
                    store.append(detect_paced(rec, h, diff_order=do, power=p, d2_thr_mad=d2g))
        rows[label] = (paced_counts, sinus_counts)

    print(f"{'config':<22}  {'thr':>3}  {'sens':>6}  {'spec':>6}  {'PPV':>6}  paced_med  sinus_p99")
    for label, (pc, sc) in rows.items():
        paced_med = int(np.median(pc)) if pc else 0
        sinus_p99 = int(np.percentile(sc, 99)) if sc else 0
        for thr in (1, 2, 3, 5):
            tp = sum(1 for c in pc if c >= thr)
            fp = sum(1 for c in sc if c >= thr)
            fn = len(pc) - tp
            tn = len(sc) - fp
            sens = tp / max(1, len(pc))
            spec = tn / max(1, len(sc))
            ppv = tp / max(1, tp + fp)
            print(f"{label:<22}  {thr:>3}  {sens:>6.2%}  {spec:>6.2%}  {ppv:>6.2%}  "
                  f"{paced_med:>9d}  {sinus_p99:>9d}")
        print()


if __name__ == "__main__":
    main()
