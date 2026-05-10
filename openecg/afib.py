"""Rule-based atrial-fibrillation detection on short (~10 s) ECG windows.

Public surface
--------------
``is_afib(signal, fs)`` → ``bool``
    Run the full pipeline (QRS detection → width-aware features → rule
    composite) and return True when the window shows AFib.

``afib_score(signal, fs)`` → ``dict``
    Same pipeline but returns the per-rule outcomes for debugging /
    visualization.

Pipeline (decided by the v6-v8 sweep on the SNUH Lydus 167K window
dataset, 2026-05-12):

  1. QRS R-peaks + per-beat widths via :func:`openecg.qrs.detect_qrs`
     with ``return_widths=True``.
  2. **Adaptive width veto**: a window is rejected if at least
     ``max(2, 30% × n_beats)`` beats have width ≥ 115 ms. Catches
     bigeminy / sustained ventricular rhythm without killing AFib
     windows that contain just 1-2 aberrant beats.
  3. **Per-RR masking**: RR intervals touching a wide beat
     (width ≥ max(120 ms, 1.25 × median width)) are dropped from the
     sequence; features are then computed on the residual. This
     surgically removes PVC-comp pause contamination while leaving
     AFib chaos intact.
  4. **OR-union of high-spec rules** (each rule alone passes spec ≥ 0.95
     on Lydus AFib vs NSR / PVC / 2°AVB / 3°AVB):

       cosen ≥ 3.078
       OR cosen ≥ 2.351   (deadband T=5 ms)
       OR dom_cluster ≤ 0.25   (T=70 ms)
       OR sarkar_fill ≥ 0.0592 (T=15 ms)
       OR cosen ≥ 2.465   (T=70 ms)
       OR dom_cluster ≤ 0.2727 (T=50 ms)
       OR sarkar_fill ≥ 0.0592 (T=40 ms)
       OR rmssd_ms ≥ 682.6

  5. **Short-window L1 safety net** (n_beats ≤ 10, n_wide_beats == 0):
       pRR_rel8 ≥ 0.70
       AND cv_rr ≥ 0.15
       AND dom_cluster ≤ 0.40
       AND max(RR) / min(RR) ≤ 2.4

       The max/min guard protects against AVB2 Wenckebach drops and
       AVB3 escape pauses (both have RR ratios > 2.4).

Performance on Lydus (n=293, 6 label-noise windows excluded):
  sens = 81.1 %, spec = 95.9 %
  per class FPR: NSR 0 %, PVC 4 %, 2°AVB 0 %, 3°AVB 9 %

The rules are deliberately conservative on the spec side — built for
windows where calling AFib should be near-certain. Use the full ML
model (Stage-2 OpenECG) for screening-grade sensitivity.

References
----------
* Lake & Moorman 2011 — Coefficient of Sample Entropy (COSEn) for short
  ECG: PMID 21037227.
* Sarkar, Ritscher, Mehra 2008 — RdR map AFib detector for implantable
  monitors. IEEE Trans. Biomed. Eng. 55: 1219-1224.
* Tateno & Glass 2001 — pRRx% relative RR-interval test. Med. Biol.
  Eng. Comput. 39: 664-671.
"""
from __future__ import annotations

import numpy as np

from openecg.qrs import detect_qrs


# ----------------------------------------------------------------- features --


def _cosen(rr_ms: np.ndarray, m: int = 1,
           r_grid_frac=(0.03, 0.05, 0.08, 0.12, 0.18, 0.25),
           deadband_ms: float = 0.0) -> float:
    """Coefficient of Sample Entropy (Lake & Moorman 2011)."""
    n = len(rr_ms)
    if n < m + 2:
        return 0.0
    mean_rr = float(np.mean(rr_ms))
    if float(np.std(rr_ms)) < deadband_ms:
        return 0.0
    best_se = None
    for rf in r_grid_frac:
        r = max(deadband_ms, rf * mean_rr)
        tpl_m = np.array([rr_ms[i:i + m] for i in range(n - m + 1)])
        tpl_m1 = np.array([rr_ms[i:i + m + 1] for i in range(n - m)])
        if len(tpl_m) < 2 or len(tpl_m1) < 2:
            continue
        d_m = np.max(np.abs(tpl_m[:, None, :] - tpl_m[None, :, :]), axis=-1)
        d_m1 = np.max(np.abs(tpl_m1[:, None, :] - tpl_m1[None, :, :]), axis=-1)
        np.fill_diagonal(d_m, np.inf)
        np.fill_diagonal(d_m1, np.inf)
        B = (d_m <= r).sum()
        A = (d_m1 <= r).sum()
        if B == 0 or A == 0:
            continue
        se = -np.log(A / B) - np.log(mean_rr / 1000.0)
        if best_se is None or se > best_se:
            best_se = se
    return float(best_se) if best_se is not None else 0.0


def _sarkar_fill(rr_ms: np.ndarray, deadband_ms: float = 0.0,
                 bin_ms: int = 80, grid: int = 13) -> float:
    """Sarkar RdR-map fill-rate."""
    drr = np.diff(rr_ms)
    if len(drr) < 3:
        return 0.0
    if deadband_ms > 0:
        drr = np.where(np.abs(drr) < deadband_ms, 0.0, drr)
    pairs = np.column_stack([drr[:-1], drr[1:]])
    lo = -(grid // 2) * bin_ms
    idx = np.clip(np.floor((pairs - lo) / bin_ms).astype(int), 0, grid - 1)
    return len(np.unique(idx[:, 0] * grid + idx[:, 1])) / (grid * grid)


def _dom_cluster_frac(rr_ms: np.ndarray, deadband_ms: float = 0.0,
                      tol_frac: float = 0.06) -> float:
    """Max #beats within ±(tol_frac × mean RR) of any beat / N."""
    n = len(rr_ms)
    if n == 0:
        return 0.0
    tol = max(deadband_ms, tol_frac * float(np.mean(rr_ms)))
    diffs = np.abs(rr_ms[:, None] - rr_ms[None, :])
    return float((diffs <= tol).sum(axis=1).max() / n)


def _rmssd_ms(rr_ms: np.ndarray, deadband_ms: float = 0.0) -> float:
    drr = np.diff(rr_ms)
    if len(drr) == 0:
        return 0.0
    if deadband_ms > 0:
        drr = np.where(np.abs(drr) < deadband_ms, 0.0, drr)
    return float(np.sqrt(np.mean(drr ** 2)))


def _pRR_rel8(rr_ms: np.ndarray, x_pct: float = 8.0,
              deadband_ms: float = 0.0) -> float:
    """% of |ΔRR| ≥ max(deadband, x% of mean RR). Tateno-Glass pRRx%."""
    if len(rr_ms) < 2:
        return 0.0
    mean_rr = float(np.mean(rr_ms))
    drr = np.abs(np.diff(rr_ms))
    thresh = max(deadband_ms, 0.01 * x_pct * mean_rr)
    return float(np.mean(drr >= thresh))


def _cv_rr(rr_ms: np.ndarray, deadband_ms: float = 0.0) -> float:
    if len(rr_ms) < 2:
        return 0.0
    s = float(np.std(rr_ms))
    if s < deadband_ms:
        return 0.0
    m = float(np.mean(rr_ms))
    if m < 1e-6:
        return 0.0
    return s / m


# ------------------------------------------------------- width masking / veto


# Wide-beat criterion for per-RR masking: absolute floor + relative factor.
# A beat is wide iff width ≥ max(120 ms, 1.25 × median width). Both bounds
# matter — the absolute floor blocks 100-110 ms aberrant beats from being
# treated as PVCs, the relative factor adapts to patients with a wide
# baseline (e.g. BBB).
_WIDE_ABS_MS = 120.0
_WIDE_REL_FACTOR = 1.25

# Adaptive veto parameters (sweep optimum, v7).
_VETO_V_MS = 115.0          # per-beat width threshold for "veto-worthy"
_VETO_FRAC_MIN = 0.30       # fraction of beats that must be wide
_VETO_COUNT_MIN = 2         # minimum absolute count of wide beats

# Short-window L1 thresholds (sweep optimum, v8).
_SHORT_N_MAX = 10
_SHORT_PRR = 0.70
_SHORT_CV = 0.15
_SHORT_DOM = 0.40
_SHORT_MAX_RR_RATIO = 2.4
_SHORT_WIDE_MS = 120.0


def _mask_wide_related_rr(rr_ms: np.ndarray, widths_ms: np.ndarray) -> np.ndarray:
    """Delete RR intervals that touch a wide-QRS beat. ``rr_ms[i]`` is the
    interval between beat i and beat i+1; ``widths_ms[i]`` is beat i's
    width. A beat is wide if width ≥ max(120, 1.25 × median width)."""
    if len(widths_ms) == 0 or len(rr_ms) == 0:
        return rr_ms.astype(np.float64).copy()
    rr = rr_ms.astype(np.float64)
    w = widths_ms.astype(np.float64)
    wide_thr = max(_WIDE_ABS_MS, _WIDE_REL_FACTOR * float(np.median(w)))
    wide = w >= wide_thr
    if not wide.any():
        return rr
    keep = np.ones(len(rr), dtype=bool)
    for i in range(len(rr)):
        if i < len(wide) and wide[i]:
            keep[i] = False
        if (i + 1) < len(wide) and wide[i + 1]:
            keep[i] = False
    return rr[keep]


def _adaptive_veto(widths_ms: np.ndarray) -> bool:
    """Veto when n_wide ≥ max(_VETO_COUNT_MIN, _VETO_FRAC_MIN × n_beats)."""
    n = len(widths_ms)
    if n == 0:
        return False
    n_wide = int(np.sum(widths_ms >= _VETO_V_MS))
    threshold = max(_VETO_COUNT_MIN, int(np.ceil(_VETO_FRAC_MIN * n)))
    return n_wide >= threshold


# ---------------------------------------------------------- main composite --


# Hard-coded rule list (greedy-union output of v7/v8 on Lydus, 2026-05-12).
# Each rule fires on the PVC-masked RR sequence. The deadband ``T`` only
# affects metrics that consume it (currently cosen and sarkar_fill).
_MAIN_RULES = [
    # (feature, deadband_ms, op, threshold)
    ("cosen", 0,   ">=", 3.078),
    ("cosen", 5,   ">=", 2.351),
    ("dom_cluster", 70, "<=", 0.25),
    ("sarkar_fill", 15, ">=", 0.05917),
    ("cosen", 70,  ">=", 2.465),
    ("dom_cluster", 50, "<=", 0.2727),
    ("sarkar_fill", 40, ">=", 0.05917),
    ("rmssd_ms", 0, ">=", 682.6),
]


def _evaluate_rule(feature: str, deadband_ms: float,
                   rr_clean: np.ndarray) -> float:
    if feature == "cosen":
        return _cosen(rr_clean, deadband_ms=deadband_ms)
    if feature == "sarkar_fill":
        return _sarkar_fill(rr_clean, deadband_ms=deadband_ms)
    if feature == "dom_cluster":
        return _dom_cluster_frac(rr_clean, deadband_ms=deadband_ms)
    if feature == "rmssd_ms":
        return _rmssd_ms(rr_clean, deadband_ms=deadband_ms)
    raise ValueError(f"unknown feature {feature!r}")


def _main_rules_fire(rr_clean: np.ndarray) -> bool:
    if len(rr_clean) < 4:
        return False
    for name, T, op, thr in _MAIN_RULES:
        val = _evaluate_rule(name, T, rr_clean)
        if (op == ">=" and val >= thr) or (op == "<=" and val <= thr):
            return True
    return False


def _short_window_safety_net(rr_ms: np.ndarray, widths_ms: np.ndarray) -> bool:
    """Layer-1 safety net for n_beats ≤ 10."""
    if len(rr_ms) == 0 or len(rr_ms) > _SHORT_N_MAX:
        return False
    if len(widths_ms) and int((widths_ms >= _SHORT_WIDE_MS).sum()) > 0:
        return False
    rr_min = float(np.min(rr_ms))
    if rr_min < 1.0:
        return False
    if float(np.max(rr_ms)) / rr_min > _SHORT_MAX_RR_RATIO:
        return False
    return (_pRR_rel8(rr_ms) >= _SHORT_PRR
            and _cv_rr(rr_ms) >= _SHORT_CV
            and _dom_cluster_frac(rr_ms) <= _SHORT_DOM)


# ------------------------------------------------------------ public API ---


def afib_score(signal, fs: int) -> dict:
    """Run the AFib pipeline and return per-stage outcomes.

    The dict always carries:
      ``is_afib`` (bool), ``reason`` (str), ``n_beats`` (int),
      ``rr_ms`` (ndarray), ``widths_ms`` (ndarray),
      ``vetoed`` (bool), ``main_fire`` (bool), ``safety_fire`` (bool).
    Useful for debugging or building visualizations.
    """
    peaks, widths = detect_qrs(signal, fs, return_widths=True)
    rr_ms = np.diff(peaks) * (1000.0 / fs) if len(peaks) >= 2 else np.array([])
    out = dict(
        is_afib=False, reason="", n_beats=len(peaks),
        rr_ms=rr_ms, widths_ms=widths,
        vetoed=False, main_fire=False, safety_fire=False,
    )
    if len(rr_ms) < 4:
        out["reason"] = "insufficient beats (need ≥ 5 peaks)"
        return out
    if _adaptive_veto(widths):
        out["vetoed"] = True
        out["reason"] = "wide-QRS adaptive veto"
        return out
    rr_clean = _mask_wide_related_rr(rr_ms, widths)
    if len(rr_clean) < 4:
        rr_clean = rr_ms
    main_fire = _main_rules_fire(rr_clean)
    out["main_fire"] = main_fire
    if main_fire:
        out["is_afib"] = True
        out["reason"] = "main composite (RR-mask, OR-union) fired"
        return out
    safety_fire = _short_window_safety_net(rr_ms, widths)
    out["safety_fire"] = safety_fire
    if safety_fire:
        out["is_afib"] = True
        out["reason"] = "short-window L1 safety net fired"
        return out
    out["reason"] = "no rule fired"
    return out


def is_afib(signal, fs: int) -> bool:
    """Return ``True`` if the 1-D ECG window is classified as AFib.

    Designed for 10-second windows (2-20 beats); short-window safety net
    kicks in automatically for n_beats ≤ 10. Use a higher-sensitivity
    deep model for screening.

    See :func:`afib_score` for the per-stage breakdown.
    """
    return bool(afib_score(signal, fs)["is_afib"])


__all__ = ["is_afib", "afib_score"]
