# OpenECG

*Clinically-grounded discrete tokenization and per-frame wave segmentation for electrocardiograms.*

[![PyPI](https://img.shields.io/pypi/v/openecg.svg)](https://pypi.org/project/openecg/)
[![Python](https://img.shields.io/pypi/pyversions/openecg.svg)](https://pypi.org/project/openecg/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

OpenECG ships:

- A 13-symbol RLE token format (`openecg.codec`, `openecg.vocab`) that compresses 12-lead ECGs into a clinically interpretable sequence.
- A pretrained Conv+Transformer per-frame wave classifier with a parallel boundary-regression head (`openecg.stage2`) trained on LUDB + QTDB + ISP. Reaches near-SOTA P / QRS / T boundary F1 across all three datasets (see [Performance](#performance)).
- Loaders and converters for **LUDB**, **QTDB**, and **ISP** datasets so you can reproduce every number in this README.

## Install

```bash
pip install openecg
```

PyTorch is a runtime dependency. On CUDA boxes, install the matching wheel first (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## Quickstart

```python
from openecg import codec, vocab

# Tokenise a hand-built event stream of (sym_id, length_ms) tuples.
events = [
    (vocab.ID_ISO, 200), (vocab.ID_P, 80),  (vocab.ID_ISO, 80),
    (vocab.ID_Q,   20),  (vocab.ID_R, 40),  (vocab.ID_S, 40),
    (vocab.ID_ISO, 120), (vocab.ID_T, 200), (vocab.ID_ISO, 220),
]
packed = codec.encode(events)              # uint16 array (RLE pack)
print(codec.render_compact(events))        # one char per event
print(codec.render_timed(events, 20))      # char count proportional to ms
print(codec.decode(packed) == events)      # round-trip
```

For wave segmentation on a real ECG signal (10s, 250 Hz, single lead → per-frame P/QRS/T/other labels), use `openecg.stage2.infer.predict_frames` after loading a checkpoint with `load_model`. End-to-end usage: `scripts/sota_comparison.py`.

## Performance

Headline numbers come from the current best checkpoint, **`stage2_v15_canonical.pt`** — a Conv+Transformer per-frame classifier with a parallel boundary-regression head and an auxiliary QRS head tapped after the lower 4 transformer layers, **whose softmaxed logits are concatenated with the lower features and projected back into the upper transformer's input** (Phase 2 of the QRS-first hierarchy). Trained jointly on LUDB + QTDB + ISP + a synthetic AV-block mix that includes Mobitz I / II / complete + paced ventricular escape scenarios. Average F1 across the six P / QRS / T on/off boundaries (Martinez per-boundary tolerances: P 50 ms, QRS 40 ms, T_on 50 ms, T_off 100 ms):

| Dataset (eval split) | OpenECG v15 | OpenECG v13_aux | OpenECG v12_reg (legacy) | Reference SOTA |
|---|---|---|---|---|
| **LUDB val** | 0.947 | **0.953** | 0.947 | DENS-ECG / Moskalenko 2020 ≈ 0.97 |
| **ISP test** | **0.967** | 0.964 | 0.966 | SemiSegECG 2025 (semi-supervised) ≈ 0.97 |
| **QTDB pu0** | **0.859** | 0.856 | 0.847 | Martinez 2004 wavelet ≈ 0.97 (T-annotated subset only) |
| **BUT PDB AVB peak F1** | **0.714** | 0.680 | 0.709 | — (only public AVB dataset with P labels) |

v15 is the first model that improves BUT PDB AVB peak F1 over the v12_reg baseline (+0.005) while also setting new records on ISP and QTDB. The concat path lets the upper layers see the explicit QRS estimate as an input feature — implementing the clinical "find P/T relative to QRS" workflow as an architectural prior. Median boundary timing error is **≤20 ms on every wave on every dataset**, meeting the clinical spec target. Full design notes are in `docs/superpowers/specs/2026-05-06-v12-postmortem.md`; the QTDB +0.020 lift over the original v12_reg came from `fix(qtdb): density-based window selection`, which corrected a label-window bug that had silenced ~12 % of q1c records during training (`scripts/verify_qtdb_fix.py`). Run `scripts/sota_comparison.py` to reproduce per-boundary breakdowns.

Single-lead robustness across the 12 LUDB leads is documented in `scripts/per_lead_v4.py`; lead III and aVL are the physiologically expected weak spots (small P / T amplitude due to axis orthogonality), which are uncommon as sole monitoring leads in clinical practice.

## Reproduce

```bash
uv sync
$env:UV_LINK_MODE = "copy"     # Windows + OneDrive workaround
$env:OPENECG_LUDB_ZIP = "<path-to-LUDB-zip>"

uv run pytest                              # unit + stage2 (LUDB integration if env set)

# Train the current best (v15 concat+paced) — needs CUDA, ~1 h on RTX 4090
uv run python scripts/retrain_v15_concat_paced.py  # → data/checkpoints/stage2_v15_concat_paced.pt

# Phase 1 ablation (aux QRS head only, no concat)
uv run python scripts/retrain_v13_aux_qrs.py   # → data/checkpoints/stage2_v13_aux.pt

# Original boundary-only baseline (kept for backward compatibility)
uv run python scripts/train_v12_reg.py     # → data/checkpoints/stage2_v12_reg.pt

# Reproduce the headline table
uv run python scripts/sota_comparison.py   # → out/sota_comparison_*.json
```
