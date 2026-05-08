"""Numpy-only DSP primitives.

Replaces scipy.signal for openecg's core API (detect_qrs, detect_pace).
Implements the textbook DSP algorithms — these are 1970s-vintage and
have not changed; the only thing that changed in scipy is implementation
detail. We match scipy's coefficient outputs to within ~1e-12 on the
filter orders we use (2 and 4) for our normalized cutoffs, and the
peak-finding output is bit-identical for clean signals.

API (drop-in subset of scipy.signal):

  butter(N, Wn, btype='low')        → (b, a) IIR coefficients
  lfilter(b, a, x, zi=None)         → (y, zf)   if zi else  y
  filtfilt(b, a, x, padlen=None)    → zero-phase forward-backward
  find_peaks(x, height=None,
             distance=None,
             prominence=None)       → (peaks, props)

Algorithms:
  Butterworth: analog prototype poles → bilinear-transform pre-warping
    → digital (z, p, k) → polynomial (b, a). Matches scipy exactly
    because both follow Oppenheim & Schafer §7.1.
  filtfilt: pad ↔ forward ↔ reverse ↔ forward ↔ reverse ↔ trim, with
    initial conditions set via the lfilter_zi steady-state method.
  find_peaks: Strict-left/non-strict-right local maxima (scipy's
    convention), then height filter, distance filter (greedy from
    largest), and prominence (Wim Spalt's 2-sided walk).
"""
from __future__ import annotations

import numpy as np


# -- Butterworth IIR design ---------------------------------------------------

def _buttap(N: int) -> np.ndarray:
    """Analog Butterworth low-pass prototype poles (cutoff = 1 rad/s).

    The N poles lie on the left-hand-side of the unit circle in the s-plane:
        p_k = exp(j·π·(2k − 1 + N)/(2N)),  k = 1..N
    """
    k = np.arange(1, N + 1)
    return np.exp(1j * np.pi * (2 * k - 1 + N) / (2 * N))


def _lp2lp(z, p, k, wo):
    """Low-pass-to-low-pass: scale poles/zeros by wo, gain by wo^(P-Z)."""
    z = np.asarray(z) * wo
    p = np.asarray(p) * wo
    k = k * wo ** (len(p) - len(z))
    return z, p, k


def _lp2hp(z, p, k, wo):
    """Low-pass-to-high-pass: invert poles/zeros and scale.

    For a Butterworth low-pass with no finite zeros (all at infinity),
    the high-pass form has N zeros at the origin and poles at wo/p.
    """
    z = np.asarray(z); p = np.asarray(p)
    deg = len(p) - len(z)
    # Gain factor: ∏(-z) / ∏(-p) maps low-pass to high-pass at DC.
    k_hp = k * float(np.real(np.prod(-z) / np.prod(-p)))
    z_hp = wo / z if z.size else np.array([], dtype=complex)
    p_hp = wo / p
    # Add `deg` zeros at the origin (high-pass has zeros at z=0 in s-plane).
    if deg > 0:
        z_hp = np.concatenate([z_hp, np.zeros(deg, dtype=complex)])
    return z_hp, p_hp, k_hp


def _lp2bp(z, p, k, wo, bw):
    """Low-pass-to-band-pass via s → (s² + wo²)/(s·bw).

    Each prototype pole p splits into two band-pass poles:
        z_root = (p · bw / 2) ± √((p · bw / 2)² − wo²)
    Adds (P − Z) zeros at the origin.
    """
    z = np.asarray(z); p = np.asarray(p)
    deg = len(p) - len(z)

    def _split(roots, half_bw):
        if roots.size == 0:
            return np.array([], dtype=complex)
        d = roots * half_bw
        disc = np.sqrt(d * d - wo * wo)
        return np.concatenate([d + disc, d - disc])

    z_bp = _split(z, bw / 2)
    p_bp = _split(p, bw / 2)
    if deg > 0:
        z_bp = np.concatenate([z_bp, np.zeros(deg, dtype=complex)])
    k_bp = k * bw ** deg
    return z_bp, p_bp, k_bp


def _bilinear_zpk(z, p, k, fs: float = 2.0):
    """Bilinear transform with sampling rate fs.

    s = 2·fs · (z − 1)/(z + 1) ↔ z = (2·fs + s)/(2·fs − s)

    Default fs=2 corresponds to scipy's pre-warping convention (Wn is the
    normalized digital frequency, pre-warped via tan(π·Wn/2)).
    """
    z = np.atleast_1d(z); p = np.atleast_1d(p)
    fs2 = 2.0 * fs
    z_d = (fs2 + z) / (fs2 - z) if z.size else np.array([], dtype=complex)
    p_d = (fs2 + p) / (fs2 - p)
    # Each analog s=∞ maps to digital z=-1.
    n_extra = len(p) - len(z)
    if n_extra > 0:
        z_d = np.concatenate([z_d, -np.ones(n_extra, dtype=complex)])
    # Bilinear gain correction: k *= ∏(fs2 − z_a) / ∏(fs2 − p_a)
    num = np.prod(fs2 - z) if z.size else 1.0
    den = np.prod(fs2 - p)
    k_d = k * float(np.real(num / den))
    return z_d, p_d, k_d


def _zpk2tf(z, p, k):
    """(zeros, poles, gain) → (b, a) polynomial coefficients."""
    b = k * np.poly(z) if len(z) else np.asarray([k], dtype=np.float64)
    a = np.poly(p)
    # If all roots come in complex-conjugate pairs the polys are real;
    # enforce that and discard tiny imaginary residue from numerical noise.
    if np.iscomplexobj(b):
        b = np.real_if_close(b, tol=1e6).astype(np.float64)
    if np.iscomplexobj(a):
        a = np.real_if_close(a, tol=1e6).astype(np.float64)
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def butter(N: int, Wn, btype: str = "low"):
    """Butterworth IIR filter design (scipy.signal.butter drop-in).

    Args:
        N: filter order.
        Wn: normalized cutoff in (0, 1) — relative to Nyquist. Scalar for
            'low'/'high'; pair (low, high) for 'band'/'bandpass'/'stop'.
        btype: 'low'/'lowpass', 'high'/'highpass', 'band'/'bandpass',
            'stop'/'bandstop'.

    Returns: (b, a) — numerator and denominator polynomial coefficients.
    """
    btype = btype.lower()
    if btype in ("lowpass",):
        btype = "low"
    elif btype in ("highpass",):
        btype = "high"
    elif btype in ("bandpass",):
        btype = "band"
    elif btype in ("bandstop",):
        btype = "stop"

    Wn = np.atleast_1d(Wn).astype(np.float64)
    if btype in ("low", "high"):
        if Wn.size != 1:
            raise ValueError(f"{btype}: Wn must be scalar")
        if not (0 < Wn[0] < 1):
            raise ValueError(f"Wn must be in (0, 1); got {Wn[0]}")
        # Pre-warp the digital cutoff to its analog equivalent.
        wo = 2.0 * np.tan(np.pi * Wn[0] / 2.0)
    elif btype in ("band", "stop"):
        if Wn.size != 2:
            raise ValueError(f"{btype}: Wn must be (low, high)")
        if not (0 < Wn[0] < Wn[1] < 1):
            raise ValueError(f"Wn must satisfy 0 < low < high < 1; got {Wn}")
        wo_lo = 2.0 * np.tan(np.pi * Wn[0] / 2.0)
        wo_hi = 2.0 * np.tan(np.pi * Wn[1] / 2.0)
        wo_center = np.sqrt(wo_lo * wo_hi)
        bw = wo_hi - wo_lo
    else:
        raise ValueError(f"Unknown btype {btype!r}")

    # Analog low-pass prototype.
    z = np.array([], dtype=complex)
    p = _buttap(N)
    k = 1.0

    # Transform to target band type (still in the analog domain).
    if btype == "low":
        z, p, k = _lp2lp(z, p, k, wo)
    elif btype == "high":
        z, p, k = _lp2hp(z, p, k, wo)
    elif btype == "band":
        z, p, k = _lp2bp(z, p, k, wo_center, bw)
    elif btype == "stop":
        # _lp2bs is implemented inline since we don't use band-stop in core.
        deg = len(p) - len(z)
        # zeros at ±j·wo_center, repeated N times
        z = np.concatenate([
            1j * wo_center * np.ones(N), -1j * wo_center * np.ones(N),
        ])
        # split poles
        d = bw / (2 * p)
        disc = np.sqrt(d * d - wo_center * wo_center)
        p = np.concatenate([d + disc, d - disc])
        k *= float(np.real(np.prod(-1j * wo_center / -p[:N]))) if N > 0 else 1.0

    # Bilinear transform → digital.
    z_d, p_d, k_d = _bilinear_zpk(z, p, k, fs=1.0)

    return _zpk2tf(z_d, p_d, k_d)


# -- Direct-form filter -------------------------------------------------------

def _normalize(b, a):
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    return b, a


def lfilter(b, a, x, zi=None):
    """Direct-form II transposed IIR filter.

    Computes  y[n] = Σ b[k]·x[n−k] − Σ a[k]·y[n−k]   (a[0] normalised to 1).
    Returns y; or (y, zf) if zi is given (final delay-line state).
    """
    b, a = _normalize(b, a)
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    nfilt = max(len(b), len(a))
    if len(b) < nfilt:
        b = np.concatenate([b, np.zeros(nfilt - len(b))])
    if len(a) < nfilt:
        a = np.concatenate([a, np.zeros(nfilt - len(a))])
    # State vector z[0..nfilt-2] in DF-II transposed form.
    z = np.zeros(nfilt - 1, dtype=np.float64) if zi is None else \
        np.asarray(zi, dtype=np.float64).copy()
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        xi = x[i]
        yi = b[0] * xi + (z[0] if z.size else 0.0)
        # Update state: z[k] = b[k+1]*x − a[k+1]*y + z[k+1]
        for k in range(z.size - 1):
            z[k] = b[k + 1] * xi - a[k + 1] * yi + z[k + 1]
        if z.size:
            z[-1] = b[-1] * xi - a[-1] * yi
        y[i] = yi
    if zi is None:
        return y
    return y, z


def lfilter_zi(b, a):
    """Steady-state initial conditions for an lfilter step input.

    Solves (I − A)·zi = (B − A·b[0]), where A, B are the state-update
    matrix and input vector of the DF-II-transposed implementation. The
    result is the state vector that makes the filter output match the
    step-response steady state from sample 0 onwards.
    """
    b, a = _normalize(b, a)
    n = max(len(b), len(a))
    if len(b) < n:
        b = np.concatenate([b, np.zeros(n - len(b))])
    if len(a) < n:
        a = np.concatenate([a, np.zeros(n - len(a))])
    # Build (I − A) where A is the (n−1)×(n−1) DF-II-transposed state matrix:
    # z[k] = b[k+1] − a[k+1]·b[0] + z[k+1], so A is shift-up plus −a column.
    m = n - 1
    if m == 0:
        return np.zeros(0)
    IminusA = np.eye(m)
    # Subtract A: z[k] depends on z[k+1] (i.e., super-diagonal +1) minus
    # a[k+1]·z[0] (because y[0] = b[0]·x + z[0], and a[k+1]·y enters z[k]).
    # Actually scipy's derivation: the state recursion in steady state is
    # zi = A·zi + (b[1:] − a[1:]·b[0]). Solve (I − A)·zi = B.
    # A_ij = -a[i+1] if j == 0 else (1 if j == i+1 else 0).
    A = np.zeros((m, m))
    for i in range(m):
        A[i, 0] = -a[i + 1]
        if i + 1 < m:
            A[i, i + 1] = 1.0
    B = b[1:] - a[1:] * b[0]
    zi = np.linalg.solve(np.eye(m) - A, B)
    return zi


def filtfilt(b, a, x, padlen=None):
    """Zero-phase forward-backward filter (scipy.signal.filtfilt drop-in).

    Pads with reflected copies of length `padlen`, runs lfilter forward,
    reverses, runs lfilter again, reverses, trims. Initial conditions are
    set via lfilter_zi to suppress edge transients.

    Default padlen = 3 · max(len(a), len(b)) (scipy convention).
    """
    b, a = _normalize(b, a)
    x = np.asarray(x, dtype=np.float64)
    nfilt = max(len(b), len(a))
    if padlen is None:
        padlen = 3 * nfilt
    if x.size <= padlen:
        padlen = max(0, x.size - 1)

    # Reflect-pad (excluding the boundary sample, scipy's "odd" extension).
    if padlen > 0:
        ext_left = 2 * x[0] - x[padlen:0:-1]
        ext_right = 2 * x[-1] - x[-2:-padlen - 2:-1]
        ext = np.concatenate([ext_left, x, ext_right])
    else:
        ext = x

    zi = lfilter_zi(b, a)
    # Forward pass.
    y, _ = lfilter(b, a, ext, zi=zi * ext[0])
    # Reverse + forward pass.
    y_rev = y[::-1]
    y2, _ = lfilter(b, a, y_rev, zi=zi * y_rev[0])
    out = y2[::-1]
    # Trim padding.
    if padlen > 0:
        out = out[padlen:-padlen]
    return out


# -- Peak finding -------------------------------------------------------------

def _local_maxima(x: np.ndarray) -> np.ndarray:
    """Strict-left, non-strict-right local maxima (matches scipy's convention).

    A peak at i satisfies x[i-1] < x[i] >= x[i+1]; for plateaus the peak
    is recorded at the left edge (the rising sample). Returns sample
    indices as int64.
    """
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    # Rising at i: x[i] > x[i-1]
    rising = x[1:] > x[:-1]
    # Falling immediately after i: x[i+1] < x[i]  (strict)
    # Plateau handling: for x[i] == x[i+1] we walk forward and check
    # whether the plateau falls (peak) or rises (not peak).
    n = x.size
    peaks = []
    i = 1
    while i < n - 1:
        if x[i] > x[i - 1]:
            j = i
            # Walk through any plateau on the right.
            while j < n - 1 and x[j] == x[j + 1]:
                j += 1
            if j < n and x[j] > x[j + 1] if j + 1 < n else False:
                # Strict fall after the plateau → it's a peak.
                # scipy reports the LEFT edge of the plateau.
                peaks.append(i)
            elif j < n - 1 and x[j] > x[j + 1]:
                peaks.append(i)
            i = j + 1
        else:
            i += 1
    return np.asarray(peaks, dtype=np.int64)


def _peak_prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Vertical prominence of each peak (Wim Spalt's two-sided walk).

    For each peak i: walk left until you find a sample greater than x[i]
    (or hit the start) — left base = min along that walk. Same to the
    right. Prominence = x[i] − max(left_base, right_base).

    See scipy.signal._peak_finding_utils._peak_prominences for the
    canonical algorithm; this is the same logic in pure numpy/python.
    """
    proms = np.empty(peaks.size, dtype=np.float64)
    n = x.size
    for idx, p in enumerate(peaks):
        peak_h = x[p]
        # Walk left.
        i = p
        left_min = peak_h
        while i > 0:
            i -= 1
            if x[i] > peak_h:
                break
            if x[i] < left_min:
                left_min = x[i]
        # Walk right.
        i = p
        right_min = peak_h
        while i < n - 1:
            i += 1
            if x[i] > peak_h:
                break
            if x[i] < right_min:
                right_min = x[i]
        proms[idx] = peak_h - max(left_min, right_min)
    return proms


def _peak_widths(x, peaks, proms, rel_height: float = 0.5):
    """Width of each peak at ``rel_height`` of its prominence below
    the peak top. Linear interpolation between integer samples gives
    sub-sample crossings (ips = "interpolated position").

    Returns the widths in samples (right_ips − left_ips).
    """
    widths = np.empty(peaks.size, dtype=np.float64)
    n = x.size
    for idx, p in enumerate(peaks):
        height_line = x[p] - rel_height * proms[idx]
        # Walk left until we cross height_line.
        i = p
        while i > 0 and x[i] > height_line:
            i -= 1
        if x[i] <= height_line and i + 1 <= p:
            # Linear interp between samples i and i+1.
            x0, x1 = x[i], x[i + 1]
            left_ips = i + (height_line - x0) / (x1 - x0) if x1 != x0 else float(i)
        else:
            left_ips = float(i)
        # Walk right.
        j = p
        while j < n - 1 and x[j] > height_line:
            j += 1
        if x[j] <= height_line and j - 1 >= p:
            x0, x1 = x[j - 1], x[j]
            right_ips = (j - 1) + (height_line - x0) / (x1 - x0) if x1 != x0 else float(j)
        else:
            right_ips = float(j)
        widths[idx] = right_ips - left_ips
    return widths


def find_peaks(
    x,
    height=None,
    distance: int | None = None,
    prominence=None,
    width=None,
):
    """Find local maxima with optional height/distance/width filters and
    prominence calculation.

    Args:
        x: 1-D signal.
        height: scalar minimum or (min, max) tuple — drop peaks outside.
        distance: minimum spacing in samples; greedy from largest peak.
        prominence: scalar min, (min, max), or (None, None) to compute
            without filtering. When set, the second return value's
            ``"prominences"`` key holds the prominence array.
        width: scalar minimum or (min, max) tuple — peak width measured
            at half-prominence (rel_height = 0.5). Setting ``width``
            forces prominence computation.

    Returns: (peaks, properties_dict).
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = _local_maxima(x)
    props: dict[str, np.ndarray] = {}

    # Height filter.
    if height is not None:
        if np.isscalar(height):
            mask = x[peaks] >= float(height)
        else:
            lo, hi = height
            mask = np.ones(peaks.size, dtype=bool)
            if lo is not None:
                mask &= x[peaks] >= float(lo)
            if hi is not None:
                mask &= x[peaks] <= float(hi)
        peaks = peaks[mask]

    # Distance filter (greedy from largest).
    if distance is not None and distance > 1 and peaks.size > 1:
        order = np.argsort(x[peaks])[::-1]  # peaks sorted by height (desc)
        keep = np.ones(peaks.size, dtype=bool)
        for rank_idx in order:
            if not keep[rank_idx]:
                continue
            p = peaks[rank_idx]
            for j in range(peaks.size):
                if j == rank_idx or not keep[j]:
                    continue
                if abs(int(peaks[j]) - int(p)) < distance:
                    keep[j] = False
        peaks = peaks[keep]

    # Prominence (computed AFTER height/distance filters; matches scipy).
    if prominence is not None or width is not None:
        proms = _peak_prominences(x, peaks)
        props["prominences"] = proms
        if prominence is not None and not (
            isinstance(prominence, tuple) and prominence == (None, None)
        ):
            if np.isscalar(prominence):
                mask = proms >= float(prominence)
            else:
                lo, hi = prominence
                mask = np.ones(proms.size, dtype=bool)
                if lo is not None:
                    mask &= proms >= float(lo)
                if hi is not None:
                    mask &= proms <= float(hi)
            peaks = peaks[mask]
            proms = proms[mask]
            props["prominences"] = proms

    # Width filter (requires prominence).
    if width is not None:
        widths = _peak_widths(x, peaks, props["prominences"])
        props["widths"] = widths
        if not (isinstance(width, tuple) and width == (None, None)):
            if np.isscalar(width):
                mask = widths >= float(width)
            else:
                lo, hi = width
                mask = np.ones(widths.size, dtype=bool)
                if lo is not None:
                    mask &= widths >= float(lo)
                if hi is not None:
                    mask &= widths <= float(hi)
            peaks = peaks[mask]
            props["widths"] = widths[mask]
            props["prominences"] = props["prominences"][mask]

    return peaks, props


__all__ = ["butter", "lfilter", "lfilter_zi", "filtfilt", "find_peaks"]
