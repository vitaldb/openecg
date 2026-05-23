"""18-class arrhythmia scheme for the class-conditional diffusion model.

Each window in the multi-source training pool is assigned exactly one
class_id from this scheme so the conditional UNet can be sampled with
explicit class control at inference time.

Class assignment is rule-based per source:

  * Lydus → primary diagnosis priority order (VT > VF > AVB_high > AVB1
    > AFib > Aflutter > paced > BBB > VPC > brady > tachy > NSR)
  * synarrdb → direct scenario → class_id map
  * MITDB → beat-symbol majority over a window
  * PTB-XL → primary scp_code priority
  * CUDB/VFDB/LQTDB → annotation-driven (Phase B)
"""
from __future__ import annotations

from typing import Optional


# 0-17 enumeration. Stable across sources — never reorder once the
# generator was trained on it.
CLASS_NSR              = 0
CLASS_AFIB             = 1
CLASS_AFLUTTER         = 2
CLASS_SINUS_BRADY      = 3
CLASS_SINUS_TACHY      = 4
CLASS_AVB1             = 5
CLASS_AVB_HIGH         = 6   # 2°, 3°, variable, CAVB
CLASS_RBBB             = 7
CLASS_LBBB             = 8
CLASS_PACED            = 9
CLASS_VPC              = 10
CLASS_VT_MONO          = 11
CLASS_VT_POLY          = 12
CLASS_VF               = 13
CLASS_TDP              = 14
CLASS_SVT              = 15
CLASS_WPW              = 16
CLASS_OTHER            = 17

N_CLASSES = 18

CLASS_NAMES = [
    "NSR", "AFib", "Aflutter", "Sinus_brady", "Sinus_tachy",
    "AVB1", "AVB_high", "RBBB", "LBBB", "Paced",
    "VPC", "VT_mono", "VT_poly", "VF", "TdP",
    "SVT", "WPW", "Other",
]


# Target distribution in the final synth dataset (sums to 1.0).
# rare classes (VF, TdP, WPW, VT_*) up-weighted relative to availability.
TARGET_DISTRIBUTION = {
    CLASS_NSR:           0.28,
    CLASS_AFIB:          0.12,
    CLASS_AFLUTTER:      0.04,
    CLASS_SINUS_BRADY:   0.06,
    CLASS_SINUS_TACHY:   0.05,
    CLASS_AVB1:          0.06,
    CLASS_AVB_HIGH:      0.03,
    CLASS_RBBB:          0.05,
    CLASS_LBBB:          0.03,
    CLASS_PACED:         0.05,
    CLASS_VPC:           0.03,
    CLASS_VT_MONO:       0.04,
    CLASS_VT_POLY:       0.03,
    CLASS_VF:            0.04,
    CLASS_TDP:           0.02,
    CLASS_SVT:           0.02,
    CLASS_WPW:           0.02,
    CLASS_OTHER:         0.03,
}
assert abs(sum(TARGET_DISTRIBUTION.values()) - 1.0) < 1e-6


def lydus_record_to_class(row: dict) -> Optional[int]:
    """Map a single Lydus duckdb row → class_id, or None to skip.

    Priority order (highest first): VT > AVB_high > AVB1 > VPC >
    AFib > Aflutter > LBBB > RBBB > paced > brady > tachy > NSR.
    "Other"-like rows (Nonspecific, Others, Undetermined, etc) are
    returned as None — they don't go into the labeled pool.

    Row dict keys (matching the ``records`` table columns): ``rhythm``,
    ``avb``, ``bbb``, ``premature_beat``, ``pacing``, ``vrate``,
    ``qrsd``, ``qtc``, ``dx``, ``conclusion``.
    """
    rhythm = (row.get("rhythm") or "").strip()
    avb = (row.get("avb") or "").strip()
    bbb = (row.get("bbb") or "").strip()
    prem = (row.get("premature_beat") or "").strip()
    pacing = (row.get("pacing") or "").strip()
    dx = (row.get("dx") or "")
    conclusion = (row.get("conclusion") or "")
    vrate = row.get("vrate") or 0
    qrsd = row.get("qrsd") or 0

    # ---- Critical rhythms first (rare, must catch) ----
    # Use word-boundary patterns so "supraventricular tachycardia"
    # doesn't get mis-classified as VT.
    import re as _re
    dx_lower = (dx + " " + conclusion).lower()
    if (_re.search(r"\b(monomorphic |sustained |non[\- ]?sustained )?ventricular tachycardia\b", dx_lower)
            and "supraventricular" not in dx_lower):
        return CLASS_VT_MONO
    if _re.search(r"\bvt\b", dx_lower):
        return CLASS_VT_MONO

    # ---- AVB (highest precedence among remaining) ----
    if avb in {"3'AVB", "AV dissociation"}:
        return CLASS_AVB_HIGH
    if avb in {"2'AVB", "Variable AVB", "IVB"}:
        return CLASS_AVB_HIGH
    if avb == "1'AVB":
        return CLASS_AVB1

    # ---- Paced ----
    if rhythm == "Possible pacing" or pacing == "Possible pacing":
        return CLASS_PACED

    # ---- Atrial rhythm disorders ----
    if rhythm == "A.fib":
        return CLASS_AFIB
    if rhythm == "A.flutter":
        return CLASS_AFLUTTER
    if rhythm == "SVT":
        return CLASS_SVT
    if rhythm == "WPW":
        return CLASS_WPW

    # ---- Conduction (BBB) — only when not overridden by rhythm above ----
    if bbb == "LBBB":
        return CLASS_LBBB
    if bbb in {"RBBB", "BFB(RBBB+LAFB)", "BRB(RBBB+LPFB)", "BFB", "BRB(RBBB+)"}:
        return CLASS_RBBB

    # ---- Premature beats (when overall rhythm sinus) ----
    if prem and "VPC" in prem:
        return CLASS_VPC

    # ---- Sinus rate / NSR ----
    if rhythm == "S.brady" or (rhythm == "NSR" and 0 < vrate < 60):
        return CLASS_SINUS_BRADY
    if rhythm == "S.tachy" or (rhythm == "NSR" and vrate >= 100):
        return CLASS_SINUS_TACHY
    if rhythm == "NSR":
        return CLASS_NSR
    if rhythm == "Junctional rhythm":
        return CLASS_SINUS_BRADY  # treat as a slow non-sinus regular rhythm
    if rhythm == "S.arrhythmia":
        return CLASS_NSR  # sinus arrhythmia — treat as NSR variant

    # Anything else (Nonspecific, Others, Undetermined, Other sinus
    # rhythm, etc) is too vague to label — skip.
    return None


# Synarrdb scenario → class_id direct map. Scenarios that don't fit
# the openecg class scheme cleanly (e.g. mobitz_2to1 is technically
# AVB_high but synarrdb's parametric version is regular 2:1 conduction)
# are mapped to the clinically-closest class.
SYNARRDB_SCENARIO_TO_CLASS = {
    "nsr":              CLASS_NSR,
    "mobitz1":          CLASS_AVB_HIGH,
    "mobitz2_2to1":     CLASS_AVB_HIGH,
    "mobitz2_3to1":     CLASS_AVB_HIGH,
    "mobitz2_4to1":     CLASS_AVB_HIGH,
    "cavb_junctional":  CLASS_AVB_HIGH,
    "cavb_ventricular": CLASS_AVB_HIGH,
    "paced_aai":        CLASS_PACED,
    "paced_vvi":        CLASS_PACED,
    "paced_ddd":        CLASS_PACED,
    "afib":             CLASS_AFIB,
    "aflutter_2to1":    CLASS_AFLUTTER,
    "vt_mono":          CLASS_VT_MONO,
    "vt_poly":          CLASS_VT_POLY,
}


__all__ = [
    "N_CLASSES", "CLASS_NAMES", "TARGET_DISTRIBUTION",
    "lydus_record_to_class", "SYNARRDB_SCENARIO_TO_CLASS",
] + [name for name in dir() if name.startswith("CLASS_")]
