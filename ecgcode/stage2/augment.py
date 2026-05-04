# ecgcode/stage2/augment.py
"""ECG-specific augmentations per SemiSegECG (arXiv 2507.18323).

Apply during training only. Skip horizontal flip and baseline shift (paper warns against).
"""

import numpy as np


def powerline_noise(sig, fs=250, freq=50.0, amplitude=0.05, rng=None):
    """Add 50Hz (or 60Hz) sinusoidal interference."""
    rng = rng or np.random.default_rng()
    t = np.arange(len(sig)) / fs
    phase = float(rng.uniform(0, 2 * np.pi))
    return sig + amplitude * np.sin(2 * np.pi * freq * t + phase)


def sine_noise(sig, fs=250, freq_range=(2.0, 30.0), amplitude=0.03, rng=None):
    """Random low-frequency sine component."""
    rng = rng or np.random.default_rng()
    freq = float(rng.uniform(*freq_range))
    t = np.arange(len(sig)) / fs
    phase = float(rng.uniform(0, 2 * np.pi))
    return sig + amplitude * np.sin(2 * np.pi * freq * t + phase)


def white_noise(sig, sigma=0.05, rng=None):
    rng = rng or np.random.default_rng()
    return sig + rng.normal(0, sigma, size=sig.shape).astype(sig.dtype)


def amplitude_scaling(sig, scale_range=0.2, rng=None):
    rng = rng or np.random.default_rng()
    scale = 1.0 + float(rng.uniform(-scale_range, scale_range))
    return sig * scale


def randaugment_ecg(sig, fs=250, n_ops=2, rng=None):
    """Apply n_ops randomly chosen ECG augmentations from the safe set."""
    rng = rng or np.random.default_rng()
    ops = [
        lambda s: powerline_noise(s, fs=fs, freq=float(rng.choice([50.0, 60.0])),
                                  amplitude=float(rng.uniform(0.02, 0.08)), rng=rng),
        lambda s: sine_noise(s, fs=fs, amplitude=float(rng.uniform(0.01, 0.05)), rng=rng),
        lambda s: white_noise(s, sigma=float(rng.uniform(0.02, 0.08)), rng=rng),
        lambda s: amplitude_scaling(s, scale_range=float(rng.uniform(0.1, 0.3)), rng=rng),
    ]
    rng.shuffle(ops)
    out = sig.astype(np.float32)
    for op in ops[:n_ops]:
        out = op(out).astype(np.float32)
    return out


# --- Time-axis augmentations (transform both signal AND labels) ---
#
# These create genuinely new training samples by simulating different heart
# rates / phases, NOT just adding noise. Signal-only aug was tested and found
# to hurt performance on clean LUDB/ISP data. Time-axis aug effectively
# enlarges the data in a way that addresses the LUDB scale bottleneck.
#
# Both functions return (sig, labels) with sample-frame alignment preserved:
#  - shift uses integer frames (1 frame = 5 samples @ 250Hz, 20ms) so no
#    quantization error
#  - stretch uses Fourier resample on signal + nearest-neighbor on labels,
#    then center-crop/pad to original length (signal stays 2500 samples,
#    labels stay 500 frames).
#
# Edge fills use replication (sig[0], sig[-1], labels[0], labels[-1]) to
# avoid baseline jumps at the boundary. The replicated frames are typically
# OTHER (isoelectric) so they don't introduce false events.


def time_shift_aligned(sig, labels, fs_sig=250, frame_ms=20,
                       max_shift_ms=200, rng=None):
    """Shift signal and labels together by an integer number of frames.

    Shift is in units of one frame (5 samples @ 250Hz = 20ms) so the label
    array stays exactly aligned with the signal. Vacated edge is filled with
    the boundary value (replication pad).
    """
    rng = rng or np.random.default_rng()
    samples_per_frame = int(round(frame_ms * fs_sig / 1000.0))
    max_frames = max(1, int(max_shift_ms / frame_ms))
    n_frames_shift = int(rng.integers(-max_frames, max_frames + 1))
    if n_frames_shift == 0:
        return sig, labels
    n_samples_shift = n_frames_shift * samples_per_frame
    sig_out = np.empty_like(sig)
    label_out = np.empty_like(labels)
    if n_samples_shift > 0:
        sig_out[:n_samples_shift] = sig[0]
        sig_out[n_samples_shift:] = sig[:-n_samples_shift]
        label_out[:n_frames_shift] = labels[0]
        label_out[n_frames_shift:] = labels[:-n_frames_shift]
    else:
        sig_out[n_samples_shift:] = sig[-1]
        sig_out[:n_samples_shift] = sig[-n_samples_shift:]
        label_out[n_frames_shift:] = labels[-1]
        label_out[:n_frames_shift] = labels[-n_frames_shift:]
    return sig_out, label_out


def time_stretch_aligned(sig, labels, scale_range=(1.0, 1.2), rng=None):
    """Resample signal and labels by factor f, then random-crop to original
    length. f > 1 stretches (slower beats per window); f == 1 is identity.

    Constraints:
      - scale_range MUST be subset of [1.0, 1.2]. Stretching beyond +20% widens
        QRS to look like bundle branch block / wide-complex (clinically
        different); the model would then label wide QRS as normal QRS.
      - f < 1 (compression) is NOT supported because LUDB/ISP windows are
        isolated 10s recordings with no neighboring signal to fill the resulting
        edge gap. Replication/reflection padding produces unrealistic flat-line
        artifacts at the boundaries. To simulate fast HR, more source data is
        needed.

    Signal: Fourier resample (preserves morphology). Labels: nearest-neighbor.
    A random window of the original size is cropped from the resampled signal,
    giving phase diversity beyond fixed center cropping.
    """
    import scipy.signal as scipy_signal
    rng = rng or np.random.default_rng()
    if scale_range[0] < 1.0:
        raise ValueError(
            f"scale_range must be >= 1.0 (got {scale_range}); compression with "
            "padding is unsafe. See time_stretch_aligned docstring."
        )
    if scale_range[1] > 1.2:
        raise ValueError(
            f"scale_range upper bound > 1.2 ({scale_range}) widens QRS beyond "
            "normal range (≥120ms = wide complex). Stay within +20%."
        )
    f = float(rng.uniform(*scale_range))
    if abs(f - 1.0) < 1e-3:
        return sig, labels
    n_orig = len(sig)
    n_new = int(round(n_orig * f))
    sig_resampled = scipy_signal.resample(sig, n_new).astype(sig.dtype)
    n_frames_orig = len(labels)
    n_frames_new = int(round(n_frames_orig * f))
    new_to_old = np.linspace(0, n_frames_orig - 1, n_frames_new).astype(int)
    label_resampled = labels[new_to_old]

    # Random crop. Use the same proportional position for sig and labels so they
    # stay aligned (both crop at the matching fraction of resampled length).
    if n_new <= n_orig or n_frames_new <= n_frames_orig:
        return sig, labels  # nothing to crop, identity
    crop_max = n_new - n_orig
    crop_max_f = n_frames_new - n_frames_orig
    sample_start = int(rng.integers(0, crop_max + 1))
    # Map sample_start fraction to frame_start so they crop the same time region
    frame_start = int(round(sample_start * crop_max_f / max(1, crop_max)))
    frame_start = min(frame_start, crop_max_f)
    sig_out = sig_resampled[sample_start:sample_start + n_orig]
    label_out = label_resampled[frame_start:frame_start + n_frames_orig]
    return sig_out, label_out


def time_axis_augment(sig, labels, p_shift=0.5, p_stretch=0.5,
                      max_shift_ms=200, scale_range=(1.0, 1.2), rng=None):
    """Apply time-axis augmentations (shift, stretch) with given probabilities.
    Signal and labels are transformed together so alignment is preserved.

    Default scale_range=(1.0, 1.2) is stretch-only within +20% (avoids
    compression padding artifacts and stays within normal QRS-width range).
    See `time_stretch_aligned` for constraints.
    """
    rng = rng or np.random.default_rng()
    if rng.random() < p_shift:
        sig, labels = time_shift_aligned(sig, labels, max_shift_ms=max_shift_ms, rng=rng)
    if rng.random() < p_stretch:
        sig, labels = time_stretch_aligned(sig, labels, scale_range=scale_range, rng=rng)
    return sig, labels
