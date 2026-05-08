"""openecg — public top-level entry points.

Two single-call detectors are surfaced here:

    >>> from openecg import detect_pace, detect_qrs
    >>> qrs = detect_qrs(sig, fs)              # R-peak indices (Pan-Tompkins)
    >>> spikes = detect_pace(sig, fs)          # pacer-spike indices

``detect_pace`` accepts an optional ``qrs_indices=...`` to localise
spikes to the PR-segment, which sharpens specificity on records where
the device emits ventricular pacing only. Atrial-only pacing puts the
spike 100-300 ms before R, so leave ``qrs_indices=None`` (the default)
to detect those too.

Both detectors are **pure numpy** — no scipy, wfdb, neurokit2, or torch
needed for ``detect_qrs`` / ``detect_pace``. The DSP primitives
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
__version__ = "0.3.2"

from openecg.pacer import detect_pace, is_paced_record
from openecg.qrs import detect_qrs

__all__ = ["detect_pace", "detect_qrs", "is_paced_record", "__version__"]
