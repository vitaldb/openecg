# openecg/synth.py
"""Compositional synthetic ECG generator for AV-block scenarios.

Real public datasets (LUDB, ISP, QTDB) all provide P-wave labels only when
P couples with a following QRS within ~200 ms. They have no labels for the
"orphan" P-waves seen in 2nd-degree (Mobitz I/II) or 3rd-degree AV block,
which is exactly the population the model needs but never sees during
training.

This module fixes the gap by **compositional synthesis**: extract clean P
templates and QRS-T templates from real LUDB sinus records (so morphology
is real), then re-time them on independent atrial and ventricular schedules
to manufacture labeled examples of:

    * Mobitz I (Wenckebach): groups of N coupled beats with progressive
      PR prolongation, then one P with no following QRS.
    * Mobitz II: fixed PR, periodic dropped QRS (2:1, 3:1 conduction).
    * Complete (3rd-degree): atrial rate independent of ventricular escape.

Boundary GT (p_on/off, qrs_on/off, t_on/off) is preserved from the real
templates and shifted to the placed beat times, yielding pixel-accurate
labels for the on/off targets we actually train on.

Public API:
    TemplateBank.from_ludb(record_ids, leads, fs=250)
    generate_avb_window(bank, lead, scenario, rng, fs=250, duration_s=10.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import scipy.signal as scipy_signal

from openecg import ludb

LEADS_12 = ludb.LEADS_12
WINDOW_SAMPLES_DEFAULT = 2500   # 10s @ 250Hz

Scenario = Literal["mobitz1", "mobitz2", "complete", "paced"]


@dataclass
class _PTemplate:
    """Single P-wave template extracted from a real beat."""
    waveform: np.ndarray   # (length,) float32, baseline-subtracted
    peak_offset: int       # index of P peak within waveform
    on_offset: int         # index of p_on within waveform
    off_offset: int        # index of p_off within waveform


@dataclass
class _QRSTTemplate:
    """QRS+T pair template extracted from a real beat (no preceding P)."""
    waveform: np.ndarray   # (length,) float32, baseline-subtracted
    qrs_on_offset: int
    qrs_peak_offset: int
    qrs_off_offset: int
    t_on_offset: int
    t_off_offset: int


# Margins around extracted segments (samples @ template fs).
P_MARGIN = 16          # ~64 ms @ 250 Hz
QRST_MARGIN = 8


def _baseline_subtract(seg: np.ndarray) -> np.ndarray:
    """Linear-detrend a short segment so the endpoints sit at zero."""
    n = len(seg)
    if n < 2:
        return seg.astype(np.float32)
    a, b = float(seg[0]), float(seg[-1])
    ramp = np.linspace(a, b, n, dtype=np.float32)
    return (seg - ramp).astype(np.float32)


class TemplateBank:
    """Per-lead bank of clean P, QRS-T, and (optionally) paced QRS-T templates
    from real LUDB records.

    All templates are stored at `fs` (default 250 Hz). The native LUDB rate
    is 500 Hz; we decimate during extraction.

    Three template buckets:
      * `p[lead]`         : P-wave templates (sinus only)
      * `qrst[lead]`      : narrow QRS-T templates (sinus only)
      * `qrst_paced[lead]`: wide paced QRS-T templates (LUDB paced records,
                            excluding the 3°AVB-paced cohort that overlaps
                            with our test set)
    """

    def __init__(self, fs: int = 250):
        self.fs = fs
        self.p: dict[str, list[_PTemplate]] = {l: [] for l in LEADS_12}
        self.qrst: dict[str, list[_QRSTTemplate]] = {l: [] for l in LEADS_12}
        self.qrst_paced: dict[str, list[_QRSTTemplate]] = {l: [] for l in LEADS_12}
        self.iso_baselines: dict[str, list[np.ndarray]] = {l: [] for l in LEADS_12}

    @classmethod
    def from_ludb(
        cls,
        record_ids: Iterable[int] | None = None,
        leads: Iterable[str] | None = None,
        fs: int = 250,
        max_per_lead: int = 600,
        only_sinus: bool = True,
        paced_record_ids: Iterable[int] | None = None,
        exclude_avb_paced: bool = True,
    ) -> "TemplateBank":
        """Build a bank from LUDB records.

        Args:
            record_ids: source records for sinus P / QRS-T templates. Defaults
                to all sinus-rhythm records when `only_sinus` is True.
            paced_record_ids: source records for the wide paced QRS-T
                templates. If None, defaults to all LUDB paced records.
                When `exclude_avb_paced` is True, the 3°AVB-paced records
                that also appear in our held-out evaluation cohort
                (rid 34, 74, 90, 104, 111) are removed to avoid leak.
        """
        bank = cls(fs=fs)
        meta = ludb.load_metadata()
        if record_ids is None:
            if only_sinus:
                record_ids = [r["id_int"] for r in meta
                              if r["rhythm"].lower() == "sinus rhythm"]
            else:
                record_ids = [r["id_int"] for r in meta]
        if paced_record_ids is None:
            paced_record_ids = [r["id_int"] for r in meta if r.get("pacemaker")]
        if exclude_avb_paced:
            paced_record_ids = [rid for rid in paced_record_ids
                                if rid not in {34, 74, 90, 104, 111}]
        if leads is None:
            leads = LEADS_12
        leads = tuple(leads)

        for rid in record_ids:
            try:
                signal_500 = ludb.load_record(rid)
            except Exception:
                continue
            for lead in leads:
                sig_500 = signal_500.get(lead)
                if sig_500 is None:
                    continue
                sig = scipy_signal.decimate(sig_500, 500 // fs, zero_phase=True)
                ann500 = ludb.load_annotations(rid, lead)
                ann = {k: [int(round(v * fs / 500)) for v in vals]
                       for k, vals in ann500.items()}
                bank._extract_from_record(sig, ann, lead, max_per_lead)

        for rid in paced_record_ids:
            try:
                signal_500 = ludb.load_record(rid)
            except Exception:
                continue
            for lead in leads:
                sig_500 = signal_500.get(lead)
                if sig_500 is None:
                    continue
                sig = scipy_signal.decimate(sig_500, 500 // fs, zero_phase=True)
                ann500 = ludb.load_annotations(rid, lead)
                ann = {k: [int(round(v * fs / 500)) for v in vals]
                       for k, vals in ann500.items()}
                bank._extract_paced_qrst(sig, ann, lead, max_per_lead)
        return bank

    # ---- extraction internals --------------------------------------------

    def _extract_from_record(self, sig, ann, lead, max_per_lead):
        n = len(sig)
        # P templates: each (p_on, p_peak, p_off) triplet
        for on, peak, off in zip(ann.get("p_on", []),
                                 ann.get("p_peak", []),
                                 ann.get("p_off", [])):
            lo = max(0, on - P_MARGIN)
            hi = min(n, off + P_MARGIN + 1)
            if hi - lo < 4:
                continue
            seg = _baseline_subtract(sig[lo:hi])
            self.p[lead].append(_PTemplate(
                waveform=seg,
                peak_offset=peak - lo,
                on_offset=on - lo,
                off_offset=off - lo,
            ))
            if len(self.p[lead]) >= max_per_lead:
                break

        # QRS-T templates: pair each QRS triplet to the next T triplet whose
        # t_on falls within 600 ms after qrs_off. LUDB sometimes drops the last
        # beat's T or has off-by-one count mismatches, so a strict zip fails;
        # proximity-pairing is robust to those edge cases.
        max_qt_gap = int(0.6 * self.fs)   # 600 ms tolerance qrs_off -> t_on
        t_on_arr = np.asarray(ann.get("t_on", []), dtype=np.int64)
        t_peak_arr = np.asarray(ann.get("t_peak", []), dtype=np.int64)
        t_off_arr = np.asarray(ann.get("t_off", []), dtype=np.int64)
        qrs_triplets = list(zip(ann.get("qrs_on", []),
                                ann.get("qrs_peak", []),
                                ann.get("qrs_off", [])))
        if len(t_on_arr) > 0 and len(t_off_arr) == len(t_on_arr) == len(t_peak_arr):
            for q_on, q_peak, q_off in qrs_triplets:
                idx = np.searchsorted(t_on_arr, q_off)
                if idx >= len(t_on_arr):
                    continue
                if t_on_arr[idx] - q_off > max_qt_gap:
                    continue
                t_on = int(t_on_arr[idx])
                t_off = int(t_off_arr[idx])
                lo = max(0, q_on - QRST_MARGIN)
                hi = min(n, t_off + QRST_MARGIN + 1)
                if hi - lo < 8:
                    continue
                seg = _baseline_subtract(sig[lo:hi])
                self.qrst[lead].append(_QRSTTemplate(
                    waveform=seg,
                    qrs_on_offset=q_on - lo,
                    qrs_peak_offset=q_peak - lo,
                    qrs_off_offset=q_off - lo,
                    t_on_offset=t_on - lo,
                    t_off_offset=t_off - lo,
                ))
                if len(self.qrst[lead]) >= max_per_lead:
                    break

        # ISO baseline: the longest run of "no annotation" within the labeled
        # range. We use the labeled_range start..first p_on as a quiet patch.
        return self._iso_extract(sig, ann, lead)

    def _extract_paced_qrst(self, sig, ann, lead, max_per_lead):
        """Same proximity-pairing as _extract_from_record but stores into
        qrst_paced[lead]. P templates are NOT collected from paced records
        because the P-wave morphology of a 3°AVB-paced patient is still
        normal sinus (atrial focus is unaffected); reusing the sinus P bank
        gives a wider variety of P shapes."""
        n = len(sig)
        max_qt_gap = int(0.6 * self.fs)
        t_on_arr = np.asarray(ann.get("t_on", []), dtype=np.int64)
        t_peak_arr = np.asarray(ann.get("t_peak", []), dtype=np.int64)
        t_off_arr = np.asarray(ann.get("t_off", []), dtype=np.int64)
        qrs_triplets = list(zip(ann.get("qrs_on", []),
                                ann.get("qrs_peak", []),
                                ann.get("qrs_off", [])))
        if not (len(t_on_arr) > 0
                and len(t_off_arr) == len(t_on_arr) == len(t_peak_arr)):
            return
        for q_on, q_peak, q_off in qrs_triplets:
            idx = np.searchsorted(t_on_arr, q_off)
            if idx >= len(t_on_arr):
                continue
            if t_on_arr[idx] - q_off > max_qt_gap:
                continue
            t_on = int(t_on_arr[idx]); t_off = int(t_off_arr[idx])
            lo = max(0, q_on - QRST_MARGIN)
            hi = min(n, t_off + QRST_MARGIN + 1)
            if hi - lo < 8:
                continue
            seg = _baseline_subtract(sig[lo:hi])
            self.qrst_paced[lead].append(_QRSTTemplate(
                waveform=seg,
                qrs_on_offset=q_on - lo,
                qrs_peak_offset=q_peak - lo,
                qrs_off_offset=q_off - lo,
                t_on_offset=t_on - lo,
                t_off_offset=t_off - lo,
            ))
            if len(self.qrst_paced[lead]) >= max_per_lead:
                break

    def _iso_extract(self, sig, ann, lead):
        n = len(sig)
        rng_lab = (min(min(ann["p_on"], default=n), min(ann["qrs_on"], default=n)),
                   max(max(ann["t_off"], default=0), max(ann["qrs_off"], default=0)))
        if rng_lab[0] >= 32:
            iso = sig[: max(32, rng_lab[0] - 4)]
            self.iso_baselines[lead].append(_baseline_subtract(iso).astype(np.float32))


# ============================================================================
# Composition
# ============================================================================

def _atrial_period_samples(atrial_bpm: float, fs: int) -> float:
    return 60.0 * fs / atrial_bpm


def _gen_atrial_times(rate_bpm: float, fs: int, n_samples: int,
                      jitter_frac: float, rng: np.random.Generator) -> list[int]:
    """Place P-wave centers at ~rate_bpm with small per-beat jitter."""
    period = _atrial_period_samples(rate_bpm, fs)
    out = []
    t = rng.uniform(0.05, 0.4) * period
    while t < n_samples:
        out.append(int(t))
        t += period * (1.0 + rng.normal(0, jitter_frac))
    return out


def _gen_ventricular_for_complete(
    p_times: list[int], rate_bpm: float, fs: int, n_samples: int,
    jitter_frac: float, rng: np.random.Generator,
) -> list[int]:
    """Independent ventricular escape rhythm for 3rd-degree AV block."""
    period = _atrial_period_samples(rate_bpm, fs)
    out = []
    t = rng.uniform(0.0, 1.0) * period   # phase-independent of atrial
    while t < n_samples:
        out.append(int(t))
        t += period * (1.0 + rng.normal(0, jitter_frac))
    return out


def _gen_ventricular_for_mobitz1(
    p_times: list[int], fs: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Wenckebach: a group of N=3..5 P-waves; first conducts at PR0=160ms,
    subsequent conduct with PR + delta * k, last P drops (no QRS).
    Returns list of (p_index_in_list, qrs_time_samples) pairs.
    """
    pairs = []
    pr_step = int(rng.integers(8, 16))             # 32-64 ms per step
    pr_base = int(rng.integers(40, 70))            # 160-280 ms
    group_size = int(rng.integers(3, 6))           # 3..5
    i = 0
    while i < len(p_times):
        # First (group_size - 1) P -> conduct, last drops
        for k in range(group_size - 1):
            if i + k >= len(p_times):
                break
            pr = pr_base + pr_step * k
            qrs_t = p_times[i + k] + pr
            pairs.append((i + k, qrs_t))
        i += group_size
    return pairs


def _gen_ventricular_for_mobitz2(
    p_times: list[int], fs: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Mobitz II: fixed PR, every Nth (2 or 3) P drops."""
    pr = int(rng.integers(45, 80))                 # 180-320ms
    ratio = int(rng.choice([2, 3]))                # 2:1 or 3:1
    pairs = []
    for i, p_t in enumerate(p_times):
        if (i + 1) % ratio == 0:
            continue   # this P drops
        pairs.append((i, p_t + pr))
    return pairs


def _add_template(out: np.ndarray, t_center: int, template: np.ndarray, peak_offset: int):
    """Add `template` to `out`, aligning template[peak_offset] to out[t_center]."""
    n = len(out)
    L = len(template)
    lo_out = t_center - peak_offset
    hi_out = lo_out + L
    lo_t = 0
    hi_t = L
    if lo_out < 0:
        lo_t -= lo_out
        lo_out = 0
    if hi_out > n:
        hi_t -= (hi_out - n)
        hi_out = n
    if hi_out > lo_out and hi_t > lo_t:
        out[lo_out:hi_out] += template[lo_t:hi_t]


def _pink_noise(n: int, rng: np.random.Generator, sigma: float = 0.02) -> np.ndarray:
    """Cheap 1/f-ish noise via cumulative gaussian increments + AR(1) low-pass."""
    eps = rng.normal(0, sigma, size=n).astype(np.float32)
    a = 0.95
    out = np.zeros(n, dtype=np.float32)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = a * out[i - 1] + eps[i]
    return out


def _baseline_wander(n: int, fs: int, rng: np.random.Generator,
                      sigma_range: tuple[float, float] = (0.05, 0.20)) -> np.ndarray:
    """Sum of 2-4 low-frequency sinusoids (0.1-0.5 Hz) emulating respiration
    and motion baseline drift."""
    sigma = float(rng.uniform(*sigma_range))
    n_components = int(rng.integers(2, 5))
    t = np.arange(n, dtype=np.float32) / fs
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_components):
        f = float(rng.uniform(0.05, 0.6))
        phi = float(rng.uniform(0.0, 2 * np.pi))
        out += sigma * np.sin(2 * np.pi * f * t + phi).astype(np.float32)
    return out


def _powerline_hum(n: int, fs: int, rng: np.random.Generator,
                    sigma_range: tuple[float, float] = (0.005, 0.04)) -> np.ndarray:
    """50 or 60 Hz powerline interference."""
    sigma = float(rng.uniform(*sigma_range))
    f = float(rng.choice([50.0, 60.0]))
    phi = float(rng.uniform(0.0, 2 * np.pi))
    t = np.arange(n, dtype=np.float32) / fs
    return (sigma * np.sin(2 * np.pi * f * t + phi)).astype(np.float32)


def _motion_artifact(n: int, fs: int, rng: np.random.Generator,
                      sigma_range: tuple[float, float] = (0.3, 1.5)) -> np.ndarray:
    """Sparse triangular spikes simulating electrode motion. ~1-3 spikes."""
    out = np.zeros(n, dtype=np.float32)
    n_spikes = int(rng.integers(1, 4))
    for _ in range(n_spikes):
        center = int(rng.integers(0, n))
        width = int(rng.integers(int(0.04 * fs), int(0.30 * fs) + 1))
        amp = float(rng.uniform(*sigma_range)) * (1.0 if rng.random() < 0.5 else -1.0)
        half = max(1, width // 2)
        lo = max(0, center - half)
        hi = min(n, center + half)
        for i in range(lo, hi):
            d = abs(i - center)
            out[i] += amp * (1.0 - d / half)
    return out


def _place_template(out: np.ndarray, t_center: int, template: np.ndarray,
                    peak_offset: int, amp: float, shift: int):
    """Add `amp * template` to `out` aligned at `t_center + shift`."""
    _add_template(out, t_center + shift, (template * amp).astype(np.float32),
                  peak_offset)


def _stretch_qrst(tmpl: _QRSTTemplate, scale: float) -> _QRSTTemplate:
    """Time-stretch a QRS-T template by `scale` (>1.0 widens QRS).

    Used to synthesize the wide-QRS ventricular escape rhythm seen in
    complete AVB. A scale of 1.5 turns a 90 ms QRS into a 135 ms QRS,
    which is the clinically diagnostic wide-QRS pattern.
    """
    if abs(scale - 1.0) < 1e-3:
        return tmpl
    new_L = max(8, int(round(len(tmpl.waveform) * scale)))
    new_wave = scipy_signal.resample(tmpl.waveform, new_L).astype(np.float32)
    return _QRSTTemplate(
        waveform=new_wave,
        qrs_on_offset=int(round(tmpl.qrs_on_offset * scale)),
        qrs_peak_offset=int(round(tmpl.qrs_peak_offset * scale)),
        qrs_off_offset=int(round(tmpl.qrs_off_offset * scale)),
        t_on_offset=int(round(tmpl.t_on_offset * scale)),
        t_off_offset=int(round(tmpl.t_off_offset * scale)),
    )


def generate_avb_window(
    bank: TemplateBank,
    lead: str,
    scenario: Scenario,
    rng: np.random.Generator,
    fs: int = 250,
    duration_s: float = 10.0,
    atrial_bpm_range: tuple[float, float] = (60.0, 100.0),
    junctional_bpm_range: tuple[float, float] = (40.0, 60.0),
    ventricular_bpm_range: tuple[float, float] = (25.0, 45.0),
    paced_bpm_range: tuple[float, float] = (50.0, 80.0),
    atrial_jitter: float = 0.03,
    vent_jitter: float = 0.03,
    # Per-beat morphology jitter — small enough to preserve the regular,
    # consistent-morphology look that distinguishes AVB clinically from afib.
    beat_amp_range: tuple[float, float] = (0.85, 1.15),
    beat_shift_max: int = 1,            # samples (= 4 ms @ 250 Hz)
    # Probability that complete AVB uses a wide-QRS ventricular escape
    # (template time-stretched 1.3-1.7x). Otherwise junctional escape (narrow).
    ventricular_escape_prob: float = 0.5,
    qrs_stretch_range: tuple[float, float] = (1.3, 1.7),
    # Background noise / artifacts (acquisition-side, not beat-related).
    baseline_wander_prob: float = 0.85,
    powerline_hum_prob: float = 0.30,
    motion_artifact_prob: float = 0.20,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Synthesize one (signal, label) example for a chosen AV-block scenario.

    Clinically the diagnostic feature of complete AVB is **two regular but
    independent rhythms** — the same atrial focus fires P-waves at a constant
    rate and the same ventricular escape focus fires QRS at a (different,
    slower) constant rate. Beat-to-beat morphology should be CONSISTENT
    within one window for both P and QRS; varying it makes the synthesis
    look like atrial fibrillation instead of AVB.

    To match this, one P template and one QRS-T template are drawn at the
    start of the window and reused for every beat. For complete AVB we
    optionally time-stretch the QRS-T (the "ventricular escape" wide-QRS
    pattern); junctional escape uses the narrow template as-is. Per-beat
    amplitude/position jitter is kept small so the rhythm still looks
    regular.

    Returns
        signal : float32 np.ndarray, length = fs * duration_s, mean ~0, std ~1
        labels : dict with keys p_on, p_off, qrs_on, qrs_off, t_on, t_off
    """
    if not bank.p[lead] or not bank.qrst[lead]:
        raise ValueError(f"TemplateBank has no templates for lead {lead}")
    n_samples = int(duration_s * fs)
    sig = _pink_noise(n_samples, rng, sigma=0.02)
    if rng.random() < baseline_wander_prob:
        sig += _baseline_wander(n_samples, fs, rng)
    if rng.random() < powerline_hum_prob:
        sig += _powerline_hum(n_samples, fs, rng)
    if rng.random() < motion_artifact_prob:
        sig += _motion_artifact(n_samples, fs, rng)

    # Single P / QRS-T template per window (consistent morphology across beats).
    p_template = bank.p[lead][int(rng.integers(0, len(bank.p[lead])))]
    if scenario == "paced":
        paced_bank = bank.qrst_paced.get(lead) or []
        if not paced_bank:
            # Fallback: stretch a sinus template if no paced templates available.
            tmpl = bank.qrst[lead][int(rng.integers(0, len(bank.qrst[lead])))]
            scale = float(rng.uniform(*qrs_stretch_range))
            qrst_template = _stretch_qrst(tmpl, scale)
        else:
            qrst_template = paced_bank[int(rng.integers(0, len(paced_bank)))]
    else:
        qrst_template = bank.qrst[lead][int(rng.integers(0, len(bank.qrst[lead])))]

    # Pick escape mechanism for complete AVB (junctional vs ventricular).
    is_ventricular_escape = (
        scenario == "complete" and rng.random() < ventricular_escape_prob
    )
    if is_ventricular_escape:
        scale = float(rng.uniform(*qrs_stretch_range))
        qrst_template = _stretch_qrst(qrst_template, scale)

    labels: dict[str, list[int]] = {k: [] for k in
        ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off")}

    atrial_bpm = float(rng.uniform(*atrial_bpm_range))
    p_times = _gen_atrial_times(atrial_bpm, fs, n_samples, atrial_jitter, rng)

    def _place_p(tp: int):
        amp = float(rng.uniform(*beat_amp_range))
        shift = int(rng.integers(-beat_shift_max, beat_shift_max + 1))
        _place_template(sig, tp, p_template.waveform, p_template.peak_offset,
                        amp, shift)
        peak_at = tp + shift
        on_s = peak_at - (p_template.peak_offset - p_template.on_offset)
        off_s = peak_at + (p_template.off_offset - p_template.peak_offset)
        if 0 <= on_s < n_samples and 0 <= off_s < n_samples:
            labels["p_on"].append(on_s)
            labels["p_off"].append(off_s)

    def _place_qrst(tv: int):
        amp = float(rng.uniform(*beat_amp_range))
        shift = int(rng.integers(-beat_shift_max, beat_shift_max + 1))
        _place_template(sig, tv, qrst_template.waveform,
                        qrst_template.qrs_peak_offset, amp, shift)
        peak_at = tv + shift
        for k_on, k_off, attr_on, attr_off in (
            ("qrs_on", "qrs_off",
             qrst_template.qrs_on_offset, qrst_template.qrs_off_offset),
            ("t_on", "t_off",
             qrst_template.t_on_offset, qrst_template.t_off_offset),
        ):
            on_s = peak_at - (qrst_template.qrs_peak_offset - attr_on)
            off_s = peak_at + (attr_off - qrst_template.qrs_peak_offset)
            if 0 <= on_s < n_samples and 0 <= off_s < n_samples:
                labels[k_on].append(on_s)
                labels[k_off].append(off_s)

    if scenario == "complete":
        bpm_range = (ventricular_bpm_range if is_ventricular_escape
                     else junctional_bpm_range)
        v_bpm = float(rng.uniform(*bpm_range))
        v_times = _gen_ventricular_for_complete(p_times, v_bpm, fs, n_samples,
                                                 vent_jitter, rng)
        for tp in p_times:
            _place_p(tp)
        for tv in v_times:
            _place_qrst(tv)
    elif scenario == "paced":
        # Atrium fires at sinus rate independently of the pacemaker, which
        # drives the ventricle at a fixed paced rate. Same independence
        # structure as complete AVB but with a real paced QRS-T morphology
        # (wide, no preceding P-wave coupling).
        v_bpm = float(rng.uniform(*paced_bpm_range))
        v_times = _gen_ventricular_for_complete(p_times, v_bpm, fs, n_samples,
                                                 vent_jitter, rng)
        for tp in p_times:
            _place_p(tp)
        for tv in v_times:
            _place_qrst(tv)
    else:
        if scenario == "mobitz1":
            pairs = _gen_ventricular_for_mobitz1(p_times, fs, rng)
        elif scenario == "mobitz2":
            pairs = _gen_ventricular_for_mobitz2(p_times, fs, rng)
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        for tp in p_times:
            _place_p(tp)
        for _i, tv in pairs:
            if 0 <= tv < n_samples:
                _place_qrst(tv)

    # z-norm to roughly match the model's input distribution.
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    return sig.astype(np.float32), labels
