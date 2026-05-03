# ecgcode/delineate.py
"""NeuroKit2 ecg_delineate wrapper.

NK provides per-beat onset/peak/offset for P, QRS, T plus separate Q/S peaks.
Missing waves are marked with NaN inside NK output; we keep that and let
labeler handle (via np.isnan checks).

Spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md §6
"""

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np


@dataclass
class DelineateResult:
    """Per-lead wave delineation output. All arrays length = num beats.
    Missing wave indices stored as NaN."""
    p_onsets: np.ndarray
    p_peaks: np.ndarray
    p_offsets: np.ndarray
    q_peaks: np.ndarray
    r_onsets: np.ndarray
    r_peaks: np.ndarray
    r_offsets: np.ndarray
    s_peaks: np.ndarray
    t_onsets: np.ndarray
    t_peaks: np.ndarray
    t_offsets: np.ndarray

    @classmethod
    def empty(cls) -> "DelineateResult":
        e = np.array([], dtype=float)
        return cls(*([e] * 11))

    @property
    def n_beats(self) -> int:
        return len(self.r_peaks)


def run(signal: np.ndarray, fs: int = 500, method: str = "dwt") -> DelineateResult:
    """Run NK ecg_peaks + ecg_delineate. Returns DelineateResult.

    On any NK exception or 0 R peaks detected, returns DelineateResult.empty().
    """
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
        if len(rpeaks) == 0:
            return DelineateResult.empty()
        _, waves = nk.ecg_delineate(
            signal, rpeaks=rpeaks.astype(int), sampling_rate=fs, method=method
        )
    except Exception:
        return DelineateResult.empty()

    return DelineateResult(
        p_onsets=np.asarray(waves["ECG_P_Onsets"], dtype=float),
        p_peaks=np.asarray(waves["ECG_P_Peaks"], dtype=float),
        p_offsets=np.asarray(waves["ECG_P_Offsets"], dtype=float),
        q_peaks=np.asarray(waves["ECG_Q_Peaks"], dtype=float),
        r_onsets=np.asarray(waves["ECG_R_Onsets"], dtype=float),
        r_peaks=rpeaks,
        r_offsets=np.asarray(waves["ECG_R_Offsets"], dtype=float),
        s_peaks=np.asarray(waves["ECG_S_Peaks"], dtype=float),
        t_onsets=np.asarray(waves["ECG_T_Onsets"], dtype=float),
        t_peaks=np.asarray(waves["ECG_T_Peaks"], dtype=float),
        t_offsets=np.asarray(waves["ECG_T_Offsets"], dtype=float),
    )
