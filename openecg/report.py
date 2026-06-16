"""Agent-facing ECG reader — one call, one structured clinical report.

:func:`report` turns a raw ECG strip into an :class:`EcgReport`: a compact,
JSON-serialisable summary an AI monitoring agent (or a human) can act on
without touching the per-sample codec channels.  It fuses three sources:

  * the **layered codec** (:func:`openecg.encode`) — rhythm, beat type, and
    wave delineation (P / QRS / T intervals) at sample resolution;
  * the **pure-numpy QRS detector** (:func:`openecg.detect_qrs`) — the robust,
    validated heart-rate / R-peak source;
  * the **rule-based AFib check** (:func:`openecg.afib_score`) — an independent
    second opinion that is cross-checked against the codec's rhythm head;
  * the **rule-based pacemaker-spike check** (:func:`openecg.detect_pacings`) —
    the *authoritative* pacing signal, since the codec's neural ``paced`` rhythm
    class never learned the spike and is unreliable.

    >>> import openecg
    >>> rep = openecg.report(signal_500hz, fs=500)
    >>> rep.summary
    'Sinus rhythm, HR 84 bpm, regular. No ectopy. (AFib rule: negative.)'
    >>> rep.to_json()                           # ready for an LLM / API payload
    '{"fs": 500, "duration_s": 10.0, "rhythm": {...}, ...}'

By default the codec runs through the **int8 ONNX** graph (~3.6 MB,
``pip install openecg[deploy]`` → onnxruntime only, no PyTorch); it falls back
to the torch checkpoint (``openecg[stage2]``) when onnxruntime is missing.  If
neither backend is installed, ``report`` degrades to a rules-only report (heart
rate + AFib + R-peaks) and flags ``"codec_unavailable"`` instead of raising.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from openecg.afib import afib_score
from openecg.layered import (
    BEAT_NAMES, BEAT_NONE, BEAT_VPC, DEFAULT_WINDOW_S, RHYTHM_AFIB,
    RHYTHM_NAMES, RHYTHM_PACED, encode, encode_stream,
)
from openecg.pacer import detect_pacings
from openecg.qrs import detect_qrs

# Frame-layer wave classes (kept local to avoid an eval import in callers).
from openecg.eval import SUPER_P, SUPER_QRS, SUPER_T, SUPER_PACED_QRS

_QRS_FRAME = (SUPER_QRS, SUPER_PACED_QRS)
# Tachy/brady cut-offs and the RR coefficient-of-variation above which the
# rhythm reads as irregular (clinically ~ atrial fibrillation territory).
_BRADY_BPM = 60.0
_TACHY_BPM = 100.0
_IRREGULAR_CV = 0.12


@dataclass
class EcgReport:
    """Structured ECG report — the return value of :func:`report`.

    Every field is JSON-native; :meth:`to_dict` / :meth:`to_json` emit a
    payload suitable for an LLM prompt, a monitoring dashboard, or an API.
    """
    fs: int
    duration_s: float
    rhythm: dict
    heart_rate: dict
    beats: dict
    intervals_ms: dict
    afib_check: dict
    pacing_check: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---- helpers -------------------------------------------------------------

def _median_segment_ms(events, classes, fs) -> Optional[float]:
    """Median duration (ms) of run-length segments whose class is in ``classes``."""
    lens = [(e - s) for s, e, c in events if c in classes]
    if not lens:
        return None
    return round(float(np.median(lens)) * 1000.0 / fs, 1)


def _pair_interval_ms(events, fs, *, src_classes, dst_classes,
                      max_ms, edge="onset"):
    """Median interval between each ``src`` segment and the nearest following
    ``dst`` segment within ``max_ms``.  ``edge='onset'`` measures from the
    src start to the dst start (e.g. PR: P-onset -> QRS-onset); ``edge='offset'``
    measures from the src start to the dst end (e.g. QT: QRS-onset -> T-offset).
    """
    src = [(s, e) for s, e, c in events if c in src_classes]
    dst = [(s, e) for s, e, c in events if c in dst_classes]
    if not src or not dst:
        return None
    max_n = max_ms * fs / 1000.0
    vals = []
    for s_on, _s_off in src:
        cand = [d for d in dst if d[0] >= s_on and (d[0] - s_on) <= max_n]
        if not cand:
            continue
        d_on, d_off = cand[0]
        end = d_off if edge == "offset" else d_on
        vals.append(end - s_on)
    if not vals:
        return None
    return round(float(np.median(vals)) * 1000.0 / fs, 1)


def _beat_type_at(beat_channel, peaks, fs):
    """Beat-type name for each R-peak: the dominant codec beat label inside a
    +/-60 ms window around the peak. A real R-peak the codec left ungated
    (typically in the 2-s eval guards) reads as ``"untyped"``, not ``"none"``
    — it *is* a beat (detect_qrs found it), just one the codec did not label."""
    half = max(1, int(round(0.06 * fs)))
    n = beat_channel.size
    out = []
    for p in peaks:
        lo, hi = max(0, p - half), min(n, p + half + 1)
        seg = beat_channel[lo:hi]
        seg = seg[seg != BEAT_NONE]
        if seg.size:
            out.append(BEAT_NAMES.get(int(np.bincount(seg).argmax()), "unknown"))
        else:
            out.append("untyped")
    return out


def _resolve_codec_model(model):
    """Pick a codec backend for :func:`report`.

    ``None`` prefers the **torch-free int8 ONNX** codec (~3.6 MB, only
    onnxruntime), falling back to the torch checkpoint when onnxruntime is
    unavailable.  ``"onnx"`` / ``"torch"`` force a backend; any other value
    (model object, checkpoint path) is passed through to :func:`encode`.
    Returns ``None`` to mean "let encode load its default torch codec".
    """
    if isinstance(model, str) and model.lower() == "onnx":
        from openecg.deploy import OnnxCodec
        return OnnxCodec()
    if isinstance(model, str) and model.lower() in ("torch", "pt", "default"):
        return "default"
    if model is None:
        try:
            from openecg.deploy import OnnxCodec
            return OnnxCodec()
        except Exception:
            return "default"      # onnxruntime missing -> torch path
    return model


def _rhythm_distribution(rhythm_channel, mask):
    """Fraction of (eval-band) samples per rhythm name, dominant first."""
    arr = rhythm_channel[mask] if mask is not None else rhythm_channel
    if arr.size == 0:
        return "sinus", 1.0, {}
    vals, cnts = np.unique(arr, return_counts=True)
    total = float(cnts.sum())
    dist = {RHYTHM_NAMES.get(int(v), str(int(v))): round(c / total, 3)
            for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1])}
    dom_id = int(vals[int(np.argmax(cnts))])
    return RHYTHM_NAMES.get(dom_id, str(dom_id)), round(cnts.max() / total, 3), dist


def _pacing_check(sig, fs, peaks, n_beats) -> dict:
    """Rule-based pacemaker-spike check — the AUTHORITATIVE pacing signal.

    The codec's neural ``paced`` rhythm class is unreliable (it never learned the
    high-frequency spike: ~0.02 recall on confirmed pacing). :func:`detect_pacings`
    (4-channel, fs-agnostic, ~100% specificity on modern ECG) is the right tool.
    ``is_paced`` requires spikes on a substantial fraction of beats (a paced rhythm
    has one spike per paced beat). **Caveat:** modern *bipolar* pacing often emits
    no visible spike — a negative here does NOT rule out pacing.
    """
    try:
        # whole-signal 4-channel detector (the validated, fs-agnostic config —
        # ~100% specificity; do NOT pass qrs_indices, which localises to detect_qrs
        # peaks and silently misses spikes when the HR estimate is off).
        spikes = detect_pacings(sig, fs)
    except Exception:
        return {}
    n_spk = int(np.asarray(spikes).size)
    spb = round(n_spk / n_beats, 2) if n_beats else 0.0
    # >=3 spikes is the high-precision point (~1% FP on sinus); when beats are
    # known, also require spikes on a fair fraction of them (a paced rhythm spikes
    # ~once per beat) to reject the odd spurious cluster.
    is_paced = n_spk >= 3 and (n_beats < 3 or spb >= 0.3)
    return {"is_paced": bool(is_paced), "n_spikes": n_spk, "spikes_per_beat": spb}


# ---- public API ----------------------------------------------------------

def report(signal, fs: int = 500, *, model=None, lead_id: int = 0,
           stream: Optional[bool] = None, max_beat_events: int = 64,
           window_s: float = DEFAULT_WINDOW_S) -> EcgReport:
    """Read one ECG strip into a structured :class:`EcgReport`.

    Parameters
    ----------
    signal : 1-D ECG at ``fs`` Hz (mV).  Any length; strips longer than
        ``window_s`` are slid through the codec via :func:`encode_stream`.
    fs : sample rate (Hz).  The bundled codec runs natively at 500 Hz.
    model : codec backend.  ``None`` (default) prefers the **torch-free int8
        ONNX** codec (~3.6 MB, onnxruntime only), falling back to the torch
        ``codec_v4`` if onnxruntime is missing.  ``"onnx"`` / ``"torch"`` force
        a backend; ``"rules"`` skips the codec entirely (heart-rate + AFib only,
        pure numpy); a model object / checkpoint path is passed to :func:`encode`.
    lead_id : lead index hint for the model (the codec is lead-agnostic; this
        only seeds an unused embedding slot and can be left at 0).
    stream : force sliding-window encoding on/off.  ``None`` auto-selects
        (stream when ``len(signal) > window_s * fs``).
    max_beat_events : cap on the number of per-beat events embedded in the
        reading (HR / counts always reflect *all* beats).

    Returns
    -------
    EcgReport — call :meth:`EcgReport.to_json` for an agent/API payload.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    fs = int(fs)
    n = sig.size
    duration_s = round(n / fs, 3) if fs else 0.0
    flags: list[str] = []

    # ---- robust, codec-independent QRS / heart rate ---------------------
    peaks, widths = detect_qrs(sig, fs, return_widths=True)
    n_beats = int(peaks.size)
    rr_ms = np.diff(peaks) * (1000.0 / fs) if n_beats >= 2 else np.array([])
    if rr_ms.size:
        inst_bpm = 60000.0 / rr_ms
        rr_cv = float(np.std(rr_ms) / np.mean(rr_ms)) if np.mean(rr_ms) else 0.0
        bpm = round(float(np.median(inst_bpm)), 0)
        hr = {
            "bpm": bpm,
            "min_bpm": round(float(inst_bpm.min()), 0),
            "max_bpm": round(float(inst_bpm.max()), 0),
            "rr_mean_ms": round(float(rr_ms.mean()), 0),
            "rr_cv": round(rr_cv, 3),
            "regularity": "irregular" if rr_cv > _IRREGULAR_CV else "regular",
        }
    else:
        bpm, rr_cv = 0.0, 0.0
        hr = {"bpm": 0.0, "min_bpm": 0.0, "max_bpm": 0.0,
              "rr_mean_ms": 0.0, "rr_cv": 0.0, "regularity": "unknown"}

    # ---- independent rule-based AFib second opinion ---------------------
    sc = afib_score(sig, fs)
    afib_check = {"is_afib": bool(sc["is_afib"]), "reason": sc["reason"]}

    # ---- rule-based pacemaker-spike check (authoritative over neural paced) --
    pacing_check = _pacing_check(sig, fs, peaks, n_beats)

    # ---- codec: rhythm / beat type / wave intervals ---------------------
    rhythm = {"label": "unknown", "confidence": 0.0, "distribution": {}}
    beats = {"count": n_beats, "by_type": {}, "vpc_count": 0, "events": []}
    intervals_ms: dict = {}
    use_codec = str(model).lower() != "rules"

    if use_codec:
        try:
            backend = _resolve_codec_model(model)
            if stream is None:
                stream = n > int(round(window_s * fs))
            if stream:
                codec = encode_stream(sig.astype(np.float32), fs=fs,
                                      window_s=window_s,
                                      model=backend or "default", lead_id=lead_id)
                mask = None  # stitched stream has no internal guard band
            else:
                codec = encode(sig.astype(np.float32), fs=fs,
                               model=backend or "default", lead_id=lead_id)
                mask = codec.eval_mask if codec.n_samples > 2 * codec.margin_samples else None

            label, conf, dist = _rhythm_distribution(codec.rhythm, mask)
            rhythm = {"label": label, "confidence": conf, "distribution": dist}

            # Beat typing: codec beat channel sampled at each detected R-peak.
            types = _beat_type_at(codec.beat, peaks, fs)
            by_type: dict[str, int] = {}
            for t in types:
                by_type[t] = by_type.get(t, 0) + 1
            ev = [{"t_s": round(float(p) / fs, 3), "sample": int(p), "type": t}
                  for p, t in zip(peaks.tolist(), types)]
            if len(ev) > max_beat_events:
                ev = ev[:max_beat_events]
                flags.append("beat_events_truncated")
            beats = {"count": n_beats, "by_type": by_type,
                     "vpc_count": int(by_type.get(BEAT_NAMES[BEAT_VPC], 0)),
                     "events": ev}

            # Wave intervals from the frame delineator.
            fev = codec.events("frame")
            intervals_ms = {
                "p_duration": _median_segment_ms(fev, (SUPER_P,), fs),
                "qrs_duration": _median_segment_ms(fev, _QRS_FRAME, fs),
                "t_duration": _median_segment_ms(fev, (SUPER_T,), fs),
                "pr": _pair_interval_ms(fev, fs, src_classes=(SUPER_P,),
                                        dst_classes=_QRS_FRAME, max_ms=400,
                                        edge="onset"),
                "qt": _pair_interval_ms(fev, fs, src_classes=_QRS_FRAME,
                                        dst_classes=(SUPER_T,), max_ms=600,
                                        edge="offset"),
            }
        except ImportError:
            flags.append("codec_unavailable")
            rhythm = {"label": "unknown", "confidence": 0.0, "distribution": {}}
        except Exception as e:  # never let a model hiccup hide the vitals
            flags.append(f"codec_error:{type(e).__name__}")

    # ---- cross-checks & flags -------------------------------------------
    codec_afib = rhythm["label"] == RHYTHM_NAMES[RHYTHM_AFIB]
    afib_check["agrees_with_codec"] = (afib_check["is_afib"] == codec_afib) \
        if rhythm["label"] != "unknown" else None
    if rhythm["label"] != "unknown" and afib_check["is_afib"] != codec_afib:
        flags.append("afib_rule_codec_disagreement")
    if pacing_check:
        codec_paced = rhythm["label"] == RHYTHM_NAMES[RHYTHM_PACED]
        pacing_check["agrees_with_codec"] = codec_paced \
            if rhythm["label"] != "unknown" else None
        if pacing_check.get("is_paced"):
            flags.append("paced_rhythm")
            # the rule found spikes the unreliable codec rhythm head did not call paced
            if rhythm["label"] != "unknown" and not codec_paced:
                flags.append("pacing_spikes_codec_missed")
    if n_beats < 3:
        flags.append("too_few_beats")
    if bpm and bpm < _BRADY_BPM:
        flags.append("bradycardia")
    if bpm and bpm > _TACHY_BPM:
        flags.append("tachycardia")
    if beats["vpc_count"] > 0:
        flags.append("ventricular_ectopy")
    amp = float(np.percentile(np.abs(sig - np.median(sig)), 98)) if n else 0.0
    if amp < 0.1:  # < 0.1 mV peak deflection — likely lead-off / low voltage
        flags.append("low_amplitude")

    summary = _summarize(rhythm, hr, beats, afib_check, pacing_check, flags)
    return EcgReport(fs=fs, duration_s=duration_s, rhythm=rhythm,
                     heart_rate=hr, beats=beats, intervals_ms=intervals_ms,
                     afib_check=afib_check, pacing_check=pacing_check,
                     flags=flags, summary=summary)


def _summarize(rhythm, hr, beats, afib_check, pacing_check, flags) -> str:
    """One human/agent-readable sentence."""
    parts = []
    rl = rhythm["label"]
    parts.append(f"{rl.capitalize()} rhythm" if rl != "unknown" else "Rhythm undetermined")
    if hr["bpm"]:
        reg = hr["regularity"]
        parts.append(f"HR {hr['bpm']:.0f} bpm" + (f", {reg}" if reg != "unknown" else ""))
    else:
        parts.append("no beats detected")
    vpc = beats.get("vpc_count", 0)
    parts.append(f"{vpc} VPC{'s' if vpc != 1 else ''}" if vpc else "no ectopy")
    afib_tag = "positive" if afib_check["is_afib"] else "negative"
    tail = f"(AFib rule: {afib_tag}.)"
    s = ", ".join(parts) + ". " + tail
    if pacing_check.get("is_paced"):
        s += (f" PACED: {pacing_check['n_spikes']} pacemaker spikes detected "
              f"({pacing_check['spikes_per_beat']}/beat).")
    if "low_amplitude" in flags:
        s += " WARNING: low signal amplitude — check lead contact."
    if "afib_rule_codec_disagreement" in flags:
        s += " NOTE: rule/codec AFib disagreement — review."
    return s


__all__ = ["report", "EcgReport"]
