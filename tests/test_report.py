"""Unit tests for openecg.report — the agent-facing structured ECG report."""
from __future__ import annotations

import json

import numpy as np
import pytest

import openecg
from openecg.report import EcgReport, report

FS = 500


def _synth_sinus(rr_ms: int = 750, n_beats: int = 12, fs: int = FS):
    """Clean sinus-like ECG: P, QRS, T bumps at fixed RR. Returns (sig, n)."""
    rng = np.random.default_rng(0)
    n = int(rr_ms * (n_beats + 1) * fs / 1000)
    t = np.arange(n) / fs
    sig = 0.03 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.01, n)
    qrs = np.hanning(int(0.08 * fs))
    pw = 0.2 * np.hanning(int(0.10 * fs))
    tw = 0.3 * np.hanning(int(0.16 * fs))
    for i in range(n_beats):
        c = int((i + 1) * rr_ms * fs / 1000)
        lo = c - len(qrs) // 2
        sig[lo:lo + len(qrs)] += qrs
        p0 = c - int(0.18 * fs)
        sig[p0:p0 + len(pw)] += pw
        t0 = c + int(0.18 * fs)
        sig[t0:t0 + len(tw)] += tw
    return sig.astype(np.float64), n


def _synth_paced(rr_ms: int = 750, n_beats: int = 12, amp: float = 2.0, fs: int = FS):
    """Sinus-like ECG with a narrow tall pacemaker spike ~40 ms before each QRS."""
    sig, n = _synth_sinus(rr_ms=rr_ms, n_beats=n_beats, fs=fs)
    for i in range(n_beats):
        c = int((i + 1) * rr_ms * fs / 1000)
        s = c - int(0.04 * fs)
        sig[s] += amp
        sig[s + 1] += amp * 0.5
    return sig, n


def test_rules_only_report_structure():
    """model='rules' returns a fully-formed report with no torch/onnx."""
    sig, _ = _synth_sinus()
    rep = report(sig, fs=FS, model="rules")
    assert isinstance(rep, EcgReport)
    # Top-level keys present and typed.
    d = rep.to_dict()
    for k in ("fs", "duration_s", "rhythm", "heart_rate", "beats",
              "intervals_ms", "afib_check", "pacing_check", "flags", "summary"):
        assert k in d
    # Heart rate recovered near the synthesized 80 bpm (750 ms RR).
    assert 70 <= rep.heart_rate["bpm"] <= 90
    assert rep.heart_rate["regularity"] == "regular"
    assert rep.afib_check["is_afib"] is False
    # Codec was skipped -> rhythm undetermined, codec agreement is None.
    assert rep.rhythm["label"] == "unknown"
    assert rep.afib_check["agrees_with_codec"] is None


def test_report_is_json_serializable():
    sig, _ = _synth_sinus()
    rep = report(sig, fs=FS, model="rules")
    s = rep.to_json()
    parsed = json.loads(s)               # round-trips without error
    assert parsed["heart_rate"]["bpm"] == rep.heart_rate["bpm"]
    # No numpy scalars leaked into the payload.
    assert isinstance(parsed["beats"]["count"], int)


def test_flags_bradycardia():
    sig, _ = _synth_sinus(rr_ms=1200)    # 50 bpm
    rep = report(sig, fs=FS, model="rules")
    assert "bradycardia" in rep.flags


def test_flags_too_few_beats():
    sig = np.random.default_rng(0).normal(0, 1e-4, FS)   # flat noise, ~no beats
    rep = report(sig, fs=FS, model="rules")
    assert "too_few_beats" in rep.flags
    assert rep.heart_rate["regularity"] in ("unknown", "regular")


def test_summary_is_human_readable():
    sig, _ = _synth_sinus()
    rep = report(sig, fs=FS, model="rules")
    assert "HR" in rep.summary and "bpm" in rep.summary
    assert rep.summary.endswith(")")


def test_pacing_detected_flags_paced():
    """A signal with pacemaker spikes -> rule-based pacing_check fires + flag.
    This is independent of the codec (the neural paced class is unreliable)."""
    sig, _ = _synth_paced(amp=2.0)
    rep = report(sig, fs=FS, model="rules")
    assert rep.pacing_check["is_paced"] is True
    assert rep.pacing_check["n_spikes"] >= 3
    assert "paced_rhythm" in rep.flags
    assert "PACED" in rep.summary


def test_pacing_absent_on_clean_sinus():
    """Clean sinus (no spikes) -> not flagged paced, no false positive."""
    sig, _ = _synth_sinus()
    rep = report(sig, fs=FS, model="rules")
    assert rep.pacing_check["is_paced"] is False
    assert rep.pacing_check["n_spikes"] == 0
    assert "paced_rhythm" not in rep.flags


def _has_onnx():
    try:
        import onnxruntime  # noqa: F401
        from openecg.deploy import bundled_codec_onnx_path
        bundled_codec_onnx_path()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_onnx(), reason="onnxruntime / bundled ONNX codec not available")
def test_onnx_backend_fills_rhythm_and_intervals():
    """The torch-free ONNX backend populates rhythm + wave intervals."""
    sig, _ = _synth_sinus()
    rep = report(sig, fs=FS, model="onnx")
    assert rep.rhythm["label"] != "unknown"
    assert 0.0 <= rep.rhythm["confidence"] <= 1.0
    # Beat typing ran against detected R-peaks.
    assert rep.beats["count"] >= 5
    assert sum(rep.beats["by_type"].values()) == rep.beats["count"]
    # agrees_with_codec is now a real bool (codec produced a rhythm).
    assert isinstance(rep.afib_check["agrees_with_codec"], bool)


@pytest.mark.skipif(not _has_onnx(), reason="onnxruntime / bundled ONNX codec not available")
def test_onnx_codec_matches_layered_encode():
    """openecg.encode(model=OnnxCodec()) yields a valid LayeredCodec."""
    from openecg.deploy import OnnxCodec
    sig, n = _synth_sinus(n_beats=12)
    codec = openecg.encode(sig.astype(np.float32)[:5000], fs=FS, model=OnnxCodec())
    assert codec.channels.shape == (3, min(n, 5000))
    assert codec.frame.max() <= 3            # 4-class frame head (other/P/QRS/T)
