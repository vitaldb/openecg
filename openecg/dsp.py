"""openecg.dsp — backwards-compatible re-export of ``opendsp``.

This module used to host a 870-line numpy-only DSP layer (Butterworth,
filtfilt, find_peaks, wavelets). That code was extracted to the
``opendsp`` PyPI package in v0.4.x so it could be shared across the
open biosignal family (openvital, openecg, openeeg) without making
openeeg awkwardly import from openecg just to get a bandpass filter.

The public API is unchanged — existing code that does::

    from openecg.dsp import butter, filtfilt, find_peaks

continues to work. New code should import directly from ``opendsp``::

    from opendsp import butter, filtfilt, find_peaks

This shim will be removed in a future release (no earlier than openecg
v0.6); migrate at your convenience.

Set ``OPENECG_LFILTER_BACKEND`` (or the new ``OPENDSP_LFILTER_BACKEND``)
to force the lfilter backend. Both names are honoured.
"""
from __future__ import annotations

from opendsp import (  # noqa: F401
    butter, lfilter, lfilter_zi, filtfilt, lfilter_backend,
    find_peaks,
    wavedec, waverec, cwt,
    rank_normalize, remove_baseline_wander,
)

# Backwards-compat aliases for internal names some callers reached into.
from opendsp.peaks import argrelmax as _argrelmax  # noqa: F401
from opendsp.util import _uniform_filter1d  # noqa: F401

__all__ = [
    "butter", "lfilter", "lfilter_zi", "filtfilt", "lfilter_backend",
    "find_peaks",
    "wavedec", "waverec", "cwt",
    "rank_normalize", "remove_baseline_wander",
]
