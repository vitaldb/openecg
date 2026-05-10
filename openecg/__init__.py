"""openecg — public top-level entry points.

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
  * ``pip install openecg[all]``       — everything

Lower-level pieces (``pacer_center_surround_score``, the multichannel
features, the BUT PDB / LUDB / PTB-XL / MIT-BIH loaders, etc.) live in
their respective modules and are not re-exported here.
"""
__version__ = "0.4.0"

from openecg.afib import afib_score, is_afib
from openecg.pacer import detect_pacings, is_paced_record
from openecg.qrs import detect_qrs, measure_qrs_widths

__all__ = [
    "afib_score", "detect_pacings", "detect_qrs", "is_afib",
    "is_paced_record", "measure_qrs_widths", "__version__",
]
