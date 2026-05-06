# openecg/stage2/multi_dataset.py
"""Combined LUDB + QTDB + ISP dataset for v3 training.

Each sequence: 10s window @ 250Hz (2500 samples), z-normalized, with lead_id and frame_labels (50Hz, 500 frames).
"""

import numpy as np
import scipy.signal as scipy_signal
import torch
from torch.utils.data import Dataset

from openecg import eval as ee
from openecg import isp, ludb, qtdb

FRAME_MS = 20
WINDOW_SAMPLES = 2500       # 10s @ 250Hz
WINDOW_FRAMES = 500
N_CLASSES = 4

# QTDB lead names -> LUDB lead-id (LEADS_12 = i, ii, iii, avr, avl, avf, v1..v6).
# Anything not in this map is skipped to avoid polluting the lead embedding.
QTDB_LEAD_TO_LUDB_ID = {
    "I": 0, "i": 0,
    "MLII": 1, "ML II": 1, "II": 1, "ii": 1,
    "III": 2, "iii": 2,
    "aVR": 3, "AVR": 3, "avr": 3,
    "aVL": 4, "AVL": 4, "avl": 4,
    "aVF": 5, "AVF": 5, "avf": 5,
    "V1": 6, "v1": 6,
    "V2": 7, "v2": 7,
    "V3": 8, "v3": 8,
    "V4": 9, "v4": 9,
    "V5": 10, "v5": 10,
    "V6": 11, "v6": 11,
}


def _normalize(sig):
    mean = float(sig.mean())
    std = float(sig.std()) + 1e-6
    return ((sig - mean) / std).astype(np.float32)


def _decimate_to_250(sig, fs_native):
    factor = fs_native // 250
    if factor == 1:
        return sig.astype(np.float64)
    return scipy_signal.decimate(sig, factor, zero_phase=True)


def _find_annotation_clusters(samples, gap):
    """Group annotation sample indices into clusters of contiguous beats.

    Two consecutive sorted samples in different clusters iff their gap > `gap`.
    Returns a list of (lo, hi) inclusive sample-index ranges, one per cluster.
    Empty input returns []. Singletons return [(s, s)].
    """
    if not samples:
        return []
    s_sorted = sorted(samples)
    clusters = []
    cur_lo = cur_hi = s_sorted[0]
    for s in s_sorted[1:]:
        if s - cur_hi > gap:
            clusters.append((cur_lo, cur_hi))
            cur_lo = s
        cur_hi = s
    clusters.append((cur_lo, cur_hi))
    return clusters


class CombinedFrameDataset(Dataset):
    """Eager-load combined train/val from multiple datasets.

    `sources` is a list of strings: 'ludb_train', 'ludb_val', 'qtdb', 'isp_train', 'isp_test'.
    """

    LEAD_TO_ID = {lead: i for i, lead in enumerate(ludb.LEADS_12)}

    def __init__(self, sources: list[str], qtdb_q1c_pu_merge: bool = True,
                 qtdb_min_anns_per_window: int = 4):
        self._qtdb_q1c_pu_merge = qtdb_q1c_pu_merge
        self._qtdb_min_anns_per_window = qtdb_min_anns_per_window
        self.items = []           # list of (source, key) for debugging
        self.cache = []           # list of (sig_250, lead_idx, labels)

        for src in sources:
            if src == "ludb_train":
                self._load_ludb(ludb.load_split()["train"])
            elif src == "ludb_val":
                self._load_ludb(ludb.load_split()["val"])
            elif src == "qtdb":
                self._load_qtdb()
            elif src == "isp_train":
                self._load_isp("train")
            elif src == "isp_test":
                self._load_isp("test")
            else:
                raise ValueError(f"unknown source: {src}")

    def _add(self, sig_250, lead_idx, labels, src_key):
        self.cache.append((sig_250, lead_idx, labels))
        self.items.append(src_key)

    def _load_ludb(self, record_ids):
        for rid in record_ids:
            try:
                record = ludb.load_record(rid)
            except Exception:
                continue
            for lead_idx, lead in enumerate(ludb.LEADS_12):
                sig_500 = record[lead]
                sig_250 = _decimate_to_250(sig_500, 500)
                sig_n = _normalize(sig_250)
                try:
                    gt_ann = ludb.load_annotations(rid, lead)
                except Exception:
                    continue
                # Truncate or pad to WINDOW_SAMPLES
                if len(sig_n) >= WINDOW_SAMPLES:
                    sig_n = sig_n[:WINDOW_SAMPLES]
                else:
                    continue
                labels = ee.gt_to_super_frames(
                    gt_ann, n_samples=len(sig_500), fs=500, frame_ms=FRAME_MS
                ).astype(np.int64)
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    continue
                self._add(sig_n, lead_idx, labels, ("ludb", rid, lead))

    def _load_qtdb(self):
        n_loaded = 0
        n_skipped = 0
        n_sparse = 0
        for rid in qtdb.records_with_q1c():
            try:
                record = qtdb.load_record(rid)
                if self._qtdb_q1c_pu_merge:
                    ann = qtdb.load_q1c_pu_merged(rid, pu_lead=0)
                else:
                    ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            win = qtdb.annotated_window(ann, window_samples=WINDOW_SAMPLES, fs=250)
            if win is None:
                continue
            start, end = win
            if end > 225000:
                end = 225000
                start = end - WINDOW_SAMPLES
            # Sparsity guard: skip the record if even the densest window has
            # too few annotations (model would otherwise learn all-OTHER bias).
            n_in_win = sum(1 for k in ("p_on","p_off","qrs_on","qrs_off","t_on","t_off")
                           for s in ann.get(k, []) if start <= s < end)
            if n_in_win < self._qtdb_min_anns_per_window:
                n_sparse += 1
                continue

            # Build per-frame labels for the window (independent of lead).
            win_ann = {k: [s - start for s in v if start <= s < end] for k, v in ann.items()}
            n_samples = WINDOW_SAMPLES
            sample_labels = np.full(n_samples, ee.SUPER_OTHER, dtype=np.uint8)
            for on, off in zip(win_ann["p_on"], win_ann["p_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_P
            for on, off in zip(win_ann["qrs_on"], win_ann["qrs_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_QRS
            for on, off in zip(win_ann["t_on"], win_ann["t_off"]):
                sample_labels[max(0, on):min(n_samples, off + 1)] = ee.SUPER_T
            samples_per_frame = WINDOW_SAMPLES // WINDOW_FRAMES   # =5
            labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
            for f in range(WINDOW_FRAMES):
                seg = sample_labels[f * samples_per_frame:(f + 1) * samples_per_frame]
                vals, counts = np.unique(seg, return_counts=True)
                labels[f] = int(vals[np.argmax(counts)])

            # Use ALL leads in the record that map cleanly to a LUDB lead-id.
            # QTDB records typically have 2 leads (e.g., MLII + V5).
            for lead_name in record.keys():
                if lead_name not in QTDB_LEAD_TO_LUDB_ID:
                    n_skipped += 1
                    continue
                lead_idx = QTDB_LEAD_TO_LUDB_ID[lead_name]
                sig = record[lead_name][start:end]
                if len(sig) < WINDOW_SAMPLES:
                    continue
                sig_n = _normalize(sig)
                self._add(sig_n, lead_idx, labels.copy(), ("qtdb", rid, lead_name))
                n_loaded += 1
        print(f"QTDB: loaded {n_loaded} sequences (skipped {n_skipped} unmappable, "
              f"{n_sparse} sparse-window records)")

    def _load_isp(self, split: str):
        rec_ids = isp.load_split()[split]
        for rid in rec_ids:
            try:
                record = isp.load_record(rid, split=split)
                ann_super = isp.load_annotations_as_super(rid, split=split)
            except Exception:
                continue
            for lead_idx, lead in enumerate(isp.LEADS_12):
                sig_1000 = record[lead]
                sig_250 = _decimate_to_250(sig_1000, 1000)
                sig_n = _normalize(sig_250)
                # Pad signal to WINDOW_SAMPLES so the model always receives a
                # 10s tensor. ISP records are typically 9998-9999 samples at
                # 1000Hz, decimating to 2499-2500 at 250Hz.
                if len(sig_n) >= WINDOW_SAMPLES:
                    sig_n = sig_n[:WINDOW_SAMPLES]
                else:
                    pad = np.zeros(WINDOW_SAMPLES - len(sig_n), dtype=sig_n.dtype)
                    sig_n = np.concatenate([sig_n, pad])
                labels = ee.gt_to_super_frames(
                    ann_super, n_samples=len(sig_1000), fs=1000, frame_ms=FRAME_MS
                ).astype(np.int64)
                # Same for labels: pad to WINDOW_FRAMES with SUPER_OTHER.
                if len(labels) >= WINDOW_FRAMES:
                    labels = labels[:WINDOW_FRAMES]
                else:
                    pad = np.full(WINDOW_FRAMES - len(labels), ee.SUPER_OTHER, dtype=labels.dtype)
                    labels = np.concatenate([labels, pad])
                self._add(sig_n, lead_idx, labels, ("isp", rid, lead))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        sig, lead_idx, labels = self.cache[idx]
        return (
            torch.from_numpy(sig),
            torch.tensor(lead_idx, dtype=torch.long),
            torch.from_numpy(labels),
        )

    def label_counts(self):
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for (_, _, labels) in self.cache:
            for c in range(N_CLASSES):
                counts[c] += int((labels == c).sum())
        return counts

    def source_counts(self):
        from collections import Counter
        return Counter(item[0] for item in self.items)


class CombinedFrameDatasetAugmented(CombinedFrameDataset):
    """Same as CombinedFrameDataset but applies ECG-specific signal-domain augmentation.

    Augmentations are pure signal-domain (powerline / sine / white noise / amplitude scale)
    so labels are unchanged. Per SemiSegECG (arXiv 2507.18323), we avoid horizontal flip
    and baseline shift since they harm delineation models.
    """

    def __init__(self, sources, n_ops=2, seed=42):
        super().__init__(sources)
        self.n_ops = n_ops
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        from openecg.stage2.augment import randaugment_ecg
        sig, lead_idx, labels = super().__getitem__(idx)
        sig_np = sig.numpy()
        sig_np = randaugment_ecg(sig_np, fs=250, n_ops=self.n_ops, rng=self.rng)
        return (
            torch.from_numpy(sig_np.astype(np.float32)),
            lead_idx,
            labels,
        )


class QTDBSlidingDataset(Dataset):
    """QTDB-only dataset with sliding window + pre-cached time stretch.

    Loads each q1c-annotated record's full 15-min signal once. Pre-computes
    resampled signals at a fixed set of stretch factors (default
    {0.9, 1.0, 1.1, 1.2}) so __getitem__ is fast: pick a random factor + slide
    a 10s window. Labels are computed on-the-fly from cached resampled
    annotation positions (cheap).

    Why pre-cache: scipy.signal.resample on 225000 samples takes ~150ms per
    call; doing it on every __getitem__ stalls the data loader. With caching,
    __getitem__ is a slice + label-build operation (~5ms).

    Memory: ~3 leads/record × 4 factors × 225000 samples × 4 bytes per record
    = ~10 MB per record × 33 records = ~330 MB total.

    Compression (f<1.0) is safe because we slide within the longer original
    signal — no padding artifacts. Only window positions that overlap the
    annotated span (with `margin_samples` minimum overlap) are sampled;
    positions outside annotations get IGNORE_INDEX in labels (excluded from
    loss / metrics).
    """

    def __init__(self, fs=250, scale_factors=(0.9, 1.0, 1.1, 1.2),
                 windows_per_record=20, margin_samples=1000, seed=42,
                 ignore_missing_waves=False, q1c_pu_merge=False,
                 cluster_gap_samples=12500, min_anns_per_window=4):
        """
        ignore_missing_waves: if True, when a wave type has unpaired on/off
            (e.g., q1c with t_off but no t_on), mark a region around the
            unpaired offset as IGNORE_INDEX (estimate [off-250ms, off+50ms]).
        q1c_pu_merge: if True, fill missing wave-onsets in q1c from pu0
            (matched by QRS_on proximity). Mutually exclusive with the IGNORE
            approach.
        cluster_gap_samples: q1c on some records (sel114, sel116, ...) has two
            disjoint annotation clusters minutes apart. Using (ann_min, ann_max)
            as the sampling span produced training windows in the unannotated
            gap labelled all-OTHER, which the model dutifully learned. Annotation
            samples separated by more than this threshold start a new cluster;
            window centers are sampled only from inside a cluster.
        min_anns_per_window: minimum number of annotation samples required
            inside a chosen window. Sparser windows are resampled.
        """
        self._ignore_missing_waves = ignore_missing_waves
        self._q1c_pu_merge = q1c_pu_merge
        self._cluster_gap_samples = cluster_gap_samples
        self._min_anns_per_window = min_anns_per_window
        import scipy.signal as scipy_signal
        self.fs = fs
        self.scale_factors = tuple(scale_factors)
        self.windows_per_record = windows_per_record
        self.margin_samples = margin_samples
        self.rng = np.random.default_rng(seed)
        self.records = []
        n_skipped = 0
        for rid in qtdb.records_with_q1c():
            try:
                rec = qtdb.load_record(rid)
                if self._q1c_pu_merge:
                    ann = qtdb.load_q1c_pu_merged(rid, pu_lead=0)
                else:
                    ann = qtdb.load_q1c(rid)
            except Exception:
                continue
            all_pos = []
            for k in ("p_on", "p_off", "qrs_on", "qrs_off", "t_on", "t_off"):
                all_pos.extend(ann[k])
            if not all_pos:
                continue
            ann_min = min(all_pos)
            ann_max = max(all_pos)
            clusters_orig = _find_annotation_clusters(
                all_pos, gap=self._cluster_gap_samples)
            mappable = []
            for lead_name, sig in rec.items():
                if lead_name in QTDB_LEAD_TO_LUDB_ID:
                    mappable.append((lead_name, QTDB_LEAD_TO_LUDB_ID[lead_name], sig))
                else:
                    n_skipped += 1
            if not mappable:
                continue
            n_orig = len(mappable[0][2])
            # Pre-resample each lead at each scale factor + cache adjusted ann positions
            cache = {}
            for f in self.scale_factors:
                if abs(f - 1.0) < 1e-6:
                    n_new = n_orig
                else:
                    n_new = int(round(n_orig * f))
                lead_sigs = {}
                for lead_name, lead_idx, sig in mappable:
                    if abs(f - 1.0) < 1e-6:
                        lead_sigs[lead_name] = sig.astype(np.float32)
                    else:
                        lead_sigs[lead_name] = scipy_signal.resample(sig, n_new).astype(np.float32)
                ann_r = {k: [int(round(v * n_new / n_orig)) for v in vals]
                         for k, vals in ann.items()}
                ann_min_r = int(round(ann_min * n_new / n_orig))
                ann_max_r = int(round(ann_max * n_new / n_orig))
                clusters_r = [(int(round(lo * n_new / n_orig)),
                               int(round(hi * n_new / n_orig)))
                              for lo, hi in clusters_orig]
                ann_pos_r = sorted(int(round(v * n_new / n_orig))
                                    for v in all_pos)
                cache[f] = {"n": n_new, "leads": lead_sigs,
                            "ann": ann_r, "ann_min": ann_min_r, "ann_max": ann_max_r,
                            "clusters": clusters_r, "ann_pos": ann_pos_r}
            self.records.append({"rid": rid, "leads": mappable, "cache": cache})
        print(f"QTDBSliding: loaded {len(self.records)} records "
              f"({sum(len(r['leads']) for r in self.records)} record-lead pairs) "
              f"with {len(self.scale_factors)} pre-resampled stretch factors, "
              f"skipped {n_skipped} unmappable lead instances")

    def __len__(self):
        return sum(len(r["leads"]) for r in self.records) * self.windows_per_record

    def label_counts(self):
        """Approximate per-class frame counts based on annotation density.
        Use the f=1.0 cache (original sampling) for the estimate."""
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for r in self.records:
            cache = r["cache"].get(1.0) or next(iter(r["cache"].values()))
            ann = cache["ann"]
            n_p = len(ann.get("p_on", []))
            n_q = len(ann.get("qrs_on", []))
            n_t = len(ann.get("t_on", []))
            counts[ee.SUPER_P] += n_p * 30  # approx wave width in samples
            counts[ee.SUPER_QRS] += n_q * 30
            counts[ee.SUPER_T] += n_t * 30
            ann_span = max(0, cache["ann_max"] - cache["ann_min"])
            counts[ee.SUPER_OTHER] += max(0, ann_span - (n_p + n_q + n_t) * 30)
        return np.maximum(counts, 1)

    def _build_window(self, record_entry, lead_entry):
        import bisect
        lead_name, lead_idx, _ = lead_entry
        # Pick a pre-cached scale factor uniformly at random
        f = float(self.rng.choice(self.scale_factors))
        cache_f = record_entry["cache"][f]
        n_new = cache_f["n"]
        sig_resampled = cache_f["leads"][lead_name]
        ann_r = cache_f["ann"]
        clusters = cache_f.get("clusters") or [(cache_f["ann_min"], cache_f["ann_max"])]
        ann_pos = cache_f.get("ann_pos", [])
        margin = self.margin_samples

        # Sample a window that overlaps a real annotation cluster (avoids the
        # min/max-midpoint trap on records with disjoint q1c clusters). Retry
        # up to a few times if the chosen window is too sparse.
        win_start = None
        for _ in range(8):
            cluster_lo, cluster_hi = clusters[int(self.rng.integers(0, len(clusters)))]
            win_start_min = max(0, cluster_lo - WINDOW_SAMPLES + margin)
            win_start_max = min(n_new - WINDOW_SAMPLES, cluster_hi - margin)
            if win_start_max < win_start_min:
                # Cluster shorter than window-2*margin: center on it.
                cand = max(0, min(n_new - WINDOW_SAMPLES,
                                  (cluster_lo + cluster_hi) // 2 - WINDOW_SAMPLES // 2))
            else:
                cand = int(self.rng.integers(win_start_min, win_start_max + 1))
            if ann_pos:
                lo = bisect.bisect_left(ann_pos, cand)
                hi = bisect.bisect_right(ann_pos, cand + WINDOW_SAMPLES)
                if hi - lo >= self._min_anns_per_window:
                    win_start = cand
                    break
            else:
                win_start = cand
                break
        if win_start is None:
            # Fall back to whichever cluster center we last picked, even if sparse.
            win_start = cand
        win_end = win_start + WINDOW_SAMPLES

        # Recompute the local "annotated span" for IGNORE/OTHER labelling:
        # only annotations falling inside [win_start, win_end] count.
        local_anns = [s for s in ann_pos if win_start <= s < win_end]
        if local_anns:
            ann_min_r = min(local_anns)
            ann_max_r = max(local_anns)
        else:
            # No annotations in window — should be rare after the loop above.
            ann_min_r = win_start
            ann_max_r = win_start - 1  # makes local span empty

        sig_win = sig_resampled[win_start:win_end].astype(np.float32)
        sig_win = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)

        # Build sample-level labels: IGNORE outside annotated span, OTHER inside, then per-wave
        sample_labels = np.full(WINDOW_SAMPLES, ee.IGNORE_INDEX, dtype=np.int64)
        local_ann_lo = max(0, ann_min_r - win_start)
        local_ann_hi = min(WINDOW_SAMPLES, ann_max_r - win_start + 1)
        if local_ann_hi > local_ann_lo:
            sample_labels[local_ann_lo:local_ann_hi] = ee.SUPER_OTHER
        for cls_id, on_key, off_key in [(ee.SUPER_P, "p_on", "p_off"),
                                         (ee.SUPER_QRS, "qrs_on", "qrs_off"),
                                         (ee.SUPER_T, "t_on", "t_off")]:
            for on, off in zip(ann_r[on_key], ann_r[off_key]):
                if on >= win_end or off < win_start:
                    continue
                lo = max(0, on - win_start)
                hi = min(WINDOW_SAMPLES, off + 1 - win_start)
                if hi > lo:
                    sample_labels[lo:hi] = cls_id

        # If ignore_missing_waves: mark estimated regions around unpaired
        # offsets as IGNORE so model isn't trained with wrong OTHER labels.
        if getattr(self, "_ignore_missing_waves", False):
            spf = self.fs // 1000
            margin_before = 62  # ~250ms @ 250Hz: typical wave width upper bound
            margin_after = 12   # ~50ms tolerance after the marked offset
            for cls_id, on_key, off_key in [(ee.SUPER_P, "p_on", "p_off"),
                                             (ee.SUPER_QRS, "qrs_on", "qrs_off"),
                                             (ee.SUPER_T, "t_on", "t_off")]:
                if len(ann_r[on_key]) > 0 and len(ann_r[off_key]) > 0:
                    continue  # both present, normal pairing already handled
                # Apply IGNORE around any standalone offsets
                for off in ann_r[off_key]:
                    if off < win_start - margin_before or off >= win_end + margin_after:
                        continue
                    lo = max(0, off - margin_before - win_start)
                    hi = min(WINDOW_SAMPLES, off + margin_after - win_start + 1)
                    if hi > lo:
                        sample_labels[lo:hi] = ee.IGNORE_INDEX
                # Apply IGNORE around any standalone onsets
                for on in ann_r[on_key]:
                    if on < win_start - margin_after or on >= win_end + margin_before:
                        continue
                    lo = max(0, on - margin_after - win_start)
                    hi = min(WINDOW_SAMPLES, on + margin_before - win_start + 1)
                    if hi > lo:
                        sample_labels[lo:hi] = ee.IGNORE_INDEX

        # Frame-level labels (50Hz, 500 frames)
        spf = WINDOW_SAMPLES // WINDOW_FRAMES  # 5
        frame_labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
        for fi in range(WINDOW_FRAMES):
            seg = sample_labels[fi * spf:(fi + 1) * spf]
            valid = seg[seg != ee.IGNORE_INDEX]
            if len(valid) == 0:
                frame_labels[fi] = ee.IGNORE_INDEX
            else:
                vals, counts = np.unique(valid, return_counts=True)
                frame_labels[fi] = int(vals[np.argmax(counts)])
        return sig_win, lead_idx, frame_labels

    def __getitem__(self, idx):
        # Map flat idx -> (record_idx, lead_idx_within_record, window_idx)
        # We don't really need the window_idx since each call samples randomly,
        # but we use it to ensure each idx maps to one record-lead pair.
        n_lead_pairs = sum(len(r["leads"]) for r in self.records)
        flat = idx % (n_lead_pairs * self.windows_per_record)
        pair_idx = flat // self.windows_per_record
        # Find which record this pair_idx belongs to
        cur = 0
        chosen_rec = None
        chosen_lead = None
        for r in self.records:
            if pair_idx < cur + len(r["leads"]):
                chosen_rec = r
                chosen_lead = r["leads"][pair_idx - cur]
                break
            cur += len(r["leads"])
        sig_win, lead_idx, frame_labels = self._build_window(chosen_rec, chosen_lead)
        return (torch.from_numpy(sig_win),
                torch.tensor(lead_idx, dtype=torch.long),
                torch.from_numpy(frame_labels))


class QTDBPuFullDataset(Dataset):
    """QTDB sliding-window dataset using pu0/pu1 (dense automatic labels) over
    the FULL 15-min record. Yields ~37x more labeled beats than q1c training.

    Each record-lead pair contributes `windows_per_record` random 10s windows.
    Labels come from pu0 (for signal 0) or pu1 (for signal 1), per lead. No
    IGNORE_INDEX needed because pu covers the entire signal.
    """

    # Best-effort lead mapping for non-standard QTDB lead names. Holter-style
    # modifications and generic channels are mapped to nearest 12-lead approx.
    _EXTENDED_LEAD_MAP = {
        # Holter / modified bipolar limb leads ≈ lead II
        "MLII": 1, "ML II": 1, "ML5": 10, "MLIII": 2,
        # Modified chest leads (CM5/CC5 ≈ V5; CM2 ≈ V2; CM4 ≈ V4)
        "CM5": 10, "CC5": 10, "CM2": 7, "CM4": 9,
        # Lead III variants
        "D3": 2, "D4": 2,
        # Composite / variant precordial leads — fall back to nearest base
        "mod.V1": 6, "V1-V2": 6, "V2-V3": 7, "V4-V5": 9,
        # Generic unlabeled channels — default to lead II
        "ECG1": 1, "ECG2": 1,
    }

    def _resolve_lead_id(self, lead_name):
        if lead_name in QTDB_LEAD_TO_LUDB_ID:
            return QTDB_LEAD_TO_LUDB_ID[lead_name]
        if lead_name in self._EXTENDED_LEAD_MAP:
            return self._EXTENDED_LEAD_MAP[lead_name]
        return 1  # default: lead II

    def __init__(self, fs=250, windows_per_record=20, seed=42):
        self.fs = fs
        self.windows_per_record = windows_per_record
        self.rng = np.random.default_rng(seed)
        self.records = []  # list of {rid, leads: [(name, lead_idx, sig, ann_per_pu)]}
        n_no_ann = 0
        for rid in qtdb.records_with_q1c():
            try:
                rec = qtdb.load_record(rid)
            except Exception:
                continue
            mappable = []
            for sig_idx, (lead_name, sig) in enumerate(rec.items()):
                try:
                    ann = qtdb.load_pu(rid, lead=sig_idx)
                except Exception:
                    n_no_ann += 1
                    continue
                lead_id = self._resolve_lead_id(lead_name)
                mappable.append((lead_name, lead_id,
                                 sig.astype(np.float32), ann))
            if mappable:
                self.records.append({"rid": rid, "leads": mappable})
        n_pairs = sum(len(r["leads"]) for r in self.records)
        print(f"QTDBPuFull: {len(self.records)} records, {n_pairs} record-lead pairs, "
              f"{n_pairs * windows_per_record} total windows; "
              f"skipped {n_no_ann} no-ann")

    def __len__(self):
        return sum(len(r["leads"]) for r in self.records) * self.windows_per_record

    def label_counts(self):
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for r in self.records:
            for _, _, sig, ann in r["leads"]:
                n = len(sig)
                n_p = sum(off - on + 1 for on, off in
                          zip(ann.get("p_on", []), ann.get("p_off", [])))
                n_q = sum(off - on + 1 for on, off in
                          zip(ann.get("qrs_on", []), ann.get("qrs_off", [])))
                n_t = sum(off - on + 1 for on, off in
                          zip(ann.get("t_on", []), ann.get("t_off", [])))
                counts[ee.SUPER_P] += n_p
                counts[ee.SUPER_QRS] += n_q
                counts[ee.SUPER_T] += n_t
                counts[ee.SUPER_OTHER] += max(0, n - n_p - n_q - n_t)
        return np.maximum(counts, 1)

    def _build_window(self, sig, ann, lead_idx):
        n = len(sig)
        if n <= WINDOW_SAMPLES:
            win_start = 0
        else:
            win_start = int(self.rng.integers(0, n - WINDOW_SAMPLES + 1))
        win_end = win_start + WINDOW_SAMPLES
        sig_win = sig[win_start:win_end]
        if len(sig_win) < WINDOW_SAMPLES:
            pad = np.zeros(WINDOW_SAMPLES - len(sig_win), dtype=sig_win.dtype)
            sig_win = np.concatenate([sig_win, pad])
        sig_n = ((sig_win - sig_win.mean()) / (sig_win.std() + 1e-6)).astype(np.float32)

        sample_labels = np.full(WINDOW_SAMPLES, ee.SUPER_OTHER, dtype=np.int64)
        for cls_id, on_key, off_key in [(ee.SUPER_P, "p_on", "p_off"),
                                         (ee.SUPER_QRS, "qrs_on", "qrs_off"),
                                         (ee.SUPER_T, "t_on", "t_off")]:
            ons = ann.get(on_key, [])
            offs = ann.get(off_key, [])
            for on, off in zip(ons, offs):
                if on >= win_end or off < win_start:
                    continue
                lo = max(0, on - win_start)
                hi = min(WINDOW_SAMPLES, off + 1 - win_start)
                if hi > lo:
                    sample_labels[lo:hi] = cls_id

        spf = WINDOW_SAMPLES // WINDOW_FRAMES
        frame_labels = np.zeros(WINDOW_FRAMES, dtype=np.int64)
        for fi in range(WINDOW_FRAMES):
            seg = sample_labels[fi * spf:(fi + 1) * spf]
            vals, counts = np.unique(seg, return_counts=True)
            frame_labels[fi] = int(vals[np.argmax(counts)])
        return sig_n, lead_idx, frame_labels

    def __getitem__(self, idx):
        n_pairs = sum(len(r["leads"]) for r in self.records)
        flat = idx % (n_pairs * self.windows_per_record)
        pair_idx = flat // self.windows_per_record
        cur = 0
        chosen = None
        for r in self.records:
            if pair_idx < cur + len(r["leads"]):
                chosen = r["leads"][pair_idx - cur]
                break
            cur += len(r["leads"])
        lead_name, lead_idx, sig, ann = chosen
        sig_n, lead_idx, frame_labels = self._build_window(sig, ann, lead_idx)
        return (torch.from_numpy(sig_n),
                torch.tensor(lead_idx, dtype=torch.long),
                torch.from_numpy(frame_labels))


class CombinedFrameDatasetTimeAugmented(CombinedFrameDataset):
    """Combined dataset with TIME-AXIS augmentation (shift + stretch) plus
    optional signal-domain noise. Signal AND labels are transformed together
    so alignment is preserved. Time shift is in integer-frame units (no
    quantization error); time stretch uses Fourier resample for the signal
    and nearest-neighbor for labels, with center-crop/pad to keep window size.

    This addresses the LUDB data scale bottleneck by simulating different heart
    rates and beat phases — fundamentally more informative than signal-only
    noise (which was tested in CombinedFrameDatasetAugmented and found to hurt).
    """

    def __init__(self, sources, max_shift_ms=200, scale_range=(1.0, 1.2),
                 p_shift=0.5, p_stretch=0.5, n_ops_signal=0, seed=42):
        super().__init__(sources)
        self.max_shift_ms = max_shift_ms
        self.scale_range = scale_range
        self.p_shift = p_shift
        self.p_stretch = p_stretch
        self.n_ops_signal = n_ops_signal
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        from openecg.stage2.augment import randaugment_ecg, time_axis_augment
        sig, lead_idx, labels = super().__getitem__(idx)
        sig_np = sig.numpy()
        labels_np = labels.numpy()
        sig_np, labels_np = time_axis_augment(
            sig_np, labels_np,
            p_shift=self.p_shift, p_stretch=self.p_stretch,
            max_shift_ms=self.max_shift_ms, scale_range=self.scale_range,
            rng=self.rng,
        )
        if self.n_ops_signal > 0:
            sig_np = randaugment_ecg(sig_np, fs=250, n_ops=self.n_ops_signal, rng=self.rng)
        return (
            torch.from_numpy(sig_np.astype(np.float32)),
            lead_idx,
            torch.from_numpy(labels_np.astype(np.int64)),
        )
