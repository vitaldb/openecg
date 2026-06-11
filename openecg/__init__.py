"""openecg — public top-level entry points.

Layered codec — one call, multi-channel label stream at sample
resolution::

    >>> import openecg
    >>> codec = openecg.encode(signal_250hz)     # 10-s window @ 250 Hz
    >>> codec.channels.shape                       # (n_layers, n_samples)
    (3, 2500)
    >>> codec.frame, codec.beat, codec.rhythm      # per-layer uint8 views
    >>> codec.events("beat", drop_class=0)         # [(start, end, class_id), ...]

See :mod:`openecg.layered` for layer definitions, class ids, and
predictor injection points.

Two single-call detectors are surfaced here:

    >>> from openecg import detect_pacings, detect_qrs
    >>> qrs = detect_qrs(sig, fs)              # R-peak indices (Pan-Tompkins)
    >>> spikes = detect_pacings(sig, fs)         # pacer-spike indices

``detect_pacings`` accepts an optional ``qrs_indices=...`` to localise
spikes to the PR-segment, which sharpens specificity on records where
the device emits ventricular pacing only. Atrial-only pacing puts the
spike 100-300 ms before R, so leave ``qrs_indices=None`` (the default)
to detect those too.

Both detectors are **pure numpy** — no scipy, wfdb, neurokit2, or torch
needed for ``detect_qrs`` / ``detect_pacings``. The DSP primitives
(Butterworth IIR design, filtfilt, find_peaks) live in ``openecg.dsp``.

Heavier features live behind optional extras:
  * ``pip install openecg[loaders]``   — wfdb + scipy (PhysioNet readers)
  * ``pip install openecg[stage2]``    — torch + transformers + hf-hub
  * ``pip install openecg[delineate]`` — neurokit2 (full P/Q/R/S/T waves)
  * ``pip install openecg[deploy]``    — onnxruntime (ONNX boundary detector)
  * ``pip install openecg[all]``       — everything

ONNX boundary detector (v54i) — runs without PyTorch::

    >>> from openecg.deploy import Inference
    >>> det = Inference("v54i.onnx")
    >>> windows = det.predict(signal_250hz)     # list of [Boundary] per window

Lower-level pieces (``pacer_center_surround_score``, the multichannel
features, the BUT PDB / LUDB / PTB-XL / MIT-BIH loaders, etc.) live in
their respective modules and are not re-exported here.
"""
__version__ = "0.8.0"

from openecg.afib import afib_score, is_afib
from openecg.dsp import rank_normalize, remove_baseline_wander
from openecg.layered import (
    LayeredCodec, encode, encode_stream, load_codec, load_codec_onnx,
)
from openecg.pacer import detect_pacings, is_paced_record
from openecg.qrs import detect_qrs, measure_qrs_widths
from openecg.report import EcgReport, report

__all__ = [
    "afib_score", "detect_pacings", "detect_qrs",
    "encode", "encode_stream", "load_codec", "load_codec_onnx", "is_afib",
    "is_paced_record", "LayeredCodec", "measure_qrs_widths",
    "rank_normalize", "remove_baseline_wander",
    "report", "EcgReport",
    "__version__",
]
