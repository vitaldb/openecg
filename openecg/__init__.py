"""openecg — public top-level entry points.

Two single-call detectors are surfaced here:

    >>> from openecg import detect_pace, detect_qrs
    >>> qrs = detect_qrs(sig, fs)              # R-peak indices (neurokit2)
    >>> spikes = detect_pace(sig, fs)        # pacer-spike indices

``detect_pace`` accepts an optional ``qrs_indices=...`` to localise
spikes to the PR-segment, which sharpens specificity on records where
the device emits ventricular pacing only. Atrial-only pacing puts the
spike 100-300 ms before R, so leave ``qrs_indices=None`` (the default)
to detect those too.

Lower-level pieces (``pacer_center_surround_score``, the multichannel
features, the BUT PDB / LUDB / PTB-XL loaders, etc.) live in their
respective modules and are not re-exported here.
"""
__version__ = "0.1.0"

import numpy as np

from openecg.pacer import detect_pace, is_paced_record


def detect_qrs(signal, fs: int) -> np.ndarray:
    """R-peak sample indices via neurokit2's ``neurokit`` method.

    Validated on this project's reference cohort at 100% sensitivity and
    80-100% PPV (see ``scripts/probe_qrs_detector.py``).
    """
    import neurokit2 as nk
    out = nk.ecg_findpeaks(
        np.asarray(signal, dtype=np.float64),
        sampling_rate=int(fs), method="neurokit",
    )
    return np.asarray(out["ECG_R_Peaks"], dtype=np.int64)


__all__ = ["detect_pace", "detect_qrs", "is_paced_record", "__version__"]
