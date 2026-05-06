# OpenECG

*Clinically-grounded discrete tokenization and per-frame wave segmentation for electrocardiograms.*

[![PyPI](https://img.shields.io/pypi/v/openecg.svg)](https://pypi.org/project/openecg/)
[![Python](https://img.shields.io/pypi/pyversions/openecg.svg)](https://pypi.org/project/openecg/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

OpenECG ships:

- A 13-symbol RLE token format (`openecg.codec`, `openecg.vocab`) that compresses 12-lead ECGs into a clinically interpretable sequence.
- A pretrained Conv+Transformer per-frame wave classifier (`openecg.stage2`) trained on LUDB + QTDB + ISP that reaches near-SOTA P / QRS / T boundary F1 on ISP test (qrs_on F1 = 0.99).
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

For wave segmentation on a real ECG signal (10s, 250 Hz, single lead → per-frame P/QRS/T/other labels), use `openecg.stage2.infer.predict_frames` after loading a checkpoint with `load_model`. End-to-end examples: `scripts/validate_v4_lit_metrics.py`, `scripts/sota_comparison.py`.

## Status

Stage 1 v1.0 complete: tokenization pipeline + LUDB validation baseline.

- Spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md`
- Plan: `docs/superpowers/plans/2026-05-03-ecgcode-stage1.md`
- Latest validation: `out/validation_v1_*.json`, `out/ablation_*.json`

### v1.0 baseline metrics (NK dwt vs LUDB cardiologist on 41 val records)

| Metric | Result | Target |
|---|---|---|
| QRS_on boundary median | 8 ms | ≤ 20 ms ✓ |
| Q-loss rate | 5.1% | ≤ 20% ✓ |
| Pacer TPR | 10/10 records | ≥ 8/10 ✓ |
| P frame F1 | 0.49 | ≥ 0.80 ✗ |
| QRS frame F1 | 0.67 | ≥ 0.90 ✗ |
| T frame F1 | 0.51 | ≥ 0.75 ✗ |
| Pacer FPR | 8.07 / 10s | < 2 ✗ |

NK dwt over-detects waves vs cardiologist (boundary precision is high but recall too eager). Stage 2 (frame classifier) is the planned mitigation. Pacer FPR requires detector tuning (v1.1).

### Stage 2 v1.0 metrics (Conv+Transformer trained supervised on LUDB cardiologist labels, 41 val records)

| Metric | Model | NK direct (S1) | Δ | Target |
|---|---|---|---|---|
| P frame F1 | **0.604** | 0.492 | +0.112 | ≥ 0.80 |
| QRS frame F1 | **0.806** | 0.666 | +0.140 | ≥ 0.90 |
| T frame F1 | **0.695** | 0.512 | +0.183 | ≥ 0.75 |
| P_on boundary sens / median | 0.89 / 8 ms | 0.79 / 12 ms | — | — |
| QRS_on boundary sens / median | 0.97 / 8 ms | 0.76 / 6 ms | — | ≤ 20 ms ✓ |
| T_on boundary sens / median | 0.89 / 12 ms | 0.41 / 32 ms | — | — |

**Critical pass criterion (model > NK on all wave classes): ✓** Model improves over NK direct by +0.11~0.18 F1 across P/QRS/T. Boundary sensitivity jumps from NK's 0.41-0.79 to model's 0.89-0.98. Absolute F1 targets (P≥0.80, QRS≥0.90, T≥0.75) not yet hit — v2 candidates: bigger model, augmentation, longer training, multi-lead joint.

Stage 2 spec: `docs/superpowers/specs/2026-05-03-ecgcode-stage2-design.md`. Plan: `docs/superpowers/plans/2026-05-03-ecgcode-stage2.md`. Train: `scripts/train_stage2.py` (~35s on RTX 4090). Validate: `scripts/validate_stage2.py`. Checkpoint: `data/checkpoints/stage2_v1.pt` (gitignored).

**v1.1 ablation (uniformly worse, kept for reference)**: scaling model to d=128/L=8 + augmentation + longer training produced ~-0.013 F1 across all classes (P=0.591, QRS=0.791, T=0.683). Conclusion: 211K params is right-sized for 1908 LUDB sequences; bigger model without more data slightly regresses. Augmentation (time-shift especially) may have introduced label/signal misalignment. Reverted defaults to v1.0; LUDBFrameDatasetAugmented kept for future v3 experiments. Ablation script: `scripts/train_stage2_v11.py`.

### Stage 2 v3 investigation, ISP alignment bug, and v4 baseline

The v3 setup (combined LUDB+QTDB+ISP, d=128/L=8, focal+aug) initially appeared to regress on LUDB val by ~0.15-0.20 F1 vs v1.0. Initial 5-setting ablation seemed to isolate `lead_emb` as a "dataset proxy" — removing it appeared to recover most of the regression.

**Both findings were artifacts of a label alignment bug** (`scripts/check_isp_alignment.py`). `gt_to_super_frames` previously computed `samples_per_frame = n_samples // n_frames`, which gave 19 instead of 20 for ISP records of 9999 samples at 1000Hz with frame_ms=20. ISP labels then drifted by up to 500ms by frame 499. LUDB (exactly 5000 samples) and QTDB (separate code path) were unaffected, so all "LUDB-only" results stayed valid.

Fix: `samples_per_frame = round(fs * frame_ms / 1000)`, with the trailing partial frame dropped and labels padded to `WINDOW_FRAMES` in the ISP loader. After re-running every combined-data setting (`scripts/redo_v3_after_isp_fix.py`, results in `out/redo_v3_after_isp_fix_*.json`):

| Setting | LUDB val P/QRS/T | QTDB ext | ISP test |
|---|---|---|---|
| F (LUDB only, no lead_emb) | 0.633 / 0.806 / 0.710 | 0.484 / 0.554 / 0.163 | 0.687 / 0.891 / 0.756 |
| **C (combined, big, lead_emb on, CE)** | **0.659 / 0.798 / 0.704** | **0.751 / 0.756 / 0.537** | **0.833 / 0.935 / 0.848** |
| D (= v3: combined + big + focal+aug + lead_emb) | 0.649 / 0.789 / 0.705 | 0.642 / 0.641 / 0.249 | 0.821 / 0.935 / 0.841 |
| E (combined, small, **no lead_emb**) | 0.640 / 0.789 / 0.702 | 0.536 / 0.582 / 0.209 | 0.813 / 0.931 / 0.837 |

**v4 = C** (combined LUDB+QTDB+ISP, d=128/L=8, lead_emb on, CE loss): best on every domain. Cross-domain wins are large (+0.10-0.15 on ISP, +0.20-0.30 on QTDB) at the cost of only -0.008 on LUDB val QRS vs F.

**Revised conclusions (post-fix):**
- v3's original design (combined data + bigger model) was correct — the apparent regression was the alignment bug, not bad architecture choices.
- Lead embedding is roughly neutral with clean labels (lead_emb on vs off differs by ~0.01-0.03 F1).
- Combined training does help cross-domain generalization (the original v3 intent).
- Focal loss + augmentation (D vs C) is fine on LUDB/ISP but causes QTDB T-wave F1 to collapse from 0.54 → 0.25; recommend plain CE (= C) until that's understood.

**Reference checkpoints**: `data/checkpoints/stage2_v4_C.pt` (C, primary v4), `data/checkpoints/stage2_v4_ludb_only.pt` (F, LUDB-only reference), `data/checkpoints/stage2_v4_combined_fixed.pt` (G, lead-agnostic combined).

### Stage 2 v4 — literature-style boundary metrics (150ms tolerance, post-proc)

`scripts/validate_v4_lit_metrics.py` reports boundary F1 / Se / PPV / median timing error in the format used by Martinez 2004, LUDB / Kalyakulina 2020, SemiSegECG 2025.

| Boundary | C LUDB | F LUDB | **C ISP** | F ISP | C QTDB | F QTDB | Literature |
|---|---|---|---|---|---|---|---|
| p_on | 0.758 | 0.701 | **0.919** | 0.795 | 0.801 | 0.769 | LUDB 0.93–0.96 / SemiSegECG ISP 0.97 |
| qrs_on | 0.870 | 0.857 | **0.970** | 0.958 | 0.844 | 0.829 | LUDB 0.98–0.99 / Martinez QTDB 0.99 / ISP 0.99 |
| t_on | 0.778 | 0.752 | **0.935** | 0.885 | 0.484 | 0.467 | LUDB 0.92–0.95 / ISP 0.95 |
| p_off | 0.774 | 0.730 | **0.931** | 0.833 | 0.801 | 0.786 | LUDB 0.93–0.96 |
| qrs_off | 0.878 | 0.865 | **0.970** | 0.957 | 0.845 | 0.833 | LUDB 0.98–0.99 / Martinez QTDB 0.99 |
| t_off | 0.771 | 0.747 | **0.928** | 0.877 | 0.828 | 0.792 | LUDB 0.92–0.95 / Martinez QTDB 0.93 |

Median timing error in ms (spec target ≤20ms): C achieves 8–20ms on every boundary across LUDB/ISP/QTDB except p_off LUDB (20ms, tied at target). F achieves 8–24ms (t_off ISP=24ms is the only miss).

**ISP test ≈ literature SOTA**: C reaches QRS F1=0.970 vs SOTA ~0.99 (gap 0.02), T F1=0.93–0.94 vs ~0.95–0.96 (gap 0.02), P F1=0.92–0.93 vs ~0.97 (gap 0.04) — supervised CE only, no semi-supervised tricks.

**LUDB val ≈ 0.10–0.15 below SOTA**: 1908 train sequences from 159 records is small for a 1M-param Transformer; FP rate is high (n_pred > n_true by ~15–20%) which suppresses PPV. Larger models and/or LUDB-style augmentation are the obvious next step. Median timing error is already at spec.

### Stage 2 v4 — SOTA paper comparison (Martinez per-boundary tolerances)

`scripts/sota_comparison.py` re-evaluates with per-boundary tolerances used in the literature (P 50ms / QRS 40ms / T_on 50ms / T_off 100ms — stricter than the 150ms loose standard) and reports F1 / Se / PPV / signed mean ± SD timing error in the format used by Martinez 2004, DENS-ECG, SemiSegECG.

**ISP test (vs SemiSegECG 2025 semi-supervised SOTA):**

| Boundary | C F1 | C Se% | mean±SD ms | SOTA F1 | Gap |
|---|---|---|---|---|---|
| **qrs_on** | **0.988** | 98.9% | +2.2 ± 9.6 | 0.99 | **≈ SOTA** |
| qrs_off | 0.953 | 95.3% | -2.7 ± 12.5 | 0.99 | -0.04 |
| t_off | 0.926 | 93.0% | -4.5 ± 25.0 | 0.96 | -0.03 |
| p_on | 0.900 | 87.9% | +1.4 ± 15.2 | 0.97 | -0.07 |
| t_on | 0.853 | 85.6% | +0.4 ± 21.1 | 0.95 | -0.10 |
| p_off | 0.843 | 82.3% | +19.7 ± 17.4 | 0.97 | -0.13 |

**LUDB val (vs DENS-ECG / Moskalenko 2020):**

| Boundary | C Se% | C mean±SD ms | DENS Se% | DENS mean±SD | Gap |
|---|---|---|---|---|---|
| qrs_on | 95.6% | -1.9 ± 12.1 | 99.6% | -1.5 ± 4.6 | -4 pp |
| qrs_off | 92.7% | +1.8 ± 13.2 | 99.6% | +1.0 ± 6.0 | -7 pp |
| p_on | 82.9% | -2.2 ± 15.7 | 96.4% | -0.6 ± 9.9 | -14 pp |
| **p_off** | **68.7%** | **+21.7 ± 20.0** | 96.4% | -0.6 ± 9.4 | **-28 pp** |
| t_on | 81.7% | +1.5 ± 18.8 | 95.0% | -2.7 ± 13.7 | -13 pp |
| t_off | 88.2% | +3.4 ± 24.6 | 95.7% | +1.3 ± 18.1 | -7 pp |

**QTDB T-subset (vs Martinez 2004 wavelet):**

| Boundary | C F1 | C mean±SD ms | Martinez Se% | Martinez mean±SD |
|---|---|---|---|---|
| qrs_on | 0.913 | -1.4 ± 14.7 | 100.0% | +4.5 ± 7.7 |
| qrs_off | 0.897 | -1.5 ± 16.6 | 100.0% | +0.8 ± 10.9 |
| t_off | 0.828 | -13.1 ± 40.1 | 99.8% | -1.6 ± 18.1 |

**Key finding — p_off systematic bias and fix**: LUDB val p_off had mean error +21.7ms — the model predicted P-wave offset 22ms LATE relative to cardiologist annotation, dragging Se to 68.7%. Same pattern on ISP (+19.7ms). Tested two fixes (`scripts/fix_p_off_bias.py`):

| Strategy | C LUDB p_off F1 | C ISP p_off F1 |
|---|---|---|
| Baseline | 0.608 (+22ms) | 0.843 (+20ms) |
| Signal-aware trim k=2.0 | 0.628 (+17ms) | 0.830 (+10ms) |
| **Fixed -22ms shift** | **0.737 (+6ms)** | **0.911 (+1ms)** |

Per-checkpoint shifts now in `openecg/stage2/infer.py`: `BOUNDARY_SHIFT_C = {"p_off": -22}` and `BOUNDARY_SHIFT_F = {"p_off": -15}` (F has smaller +14ms bias). Use with the new `extract_boundaries(frames, boundary_shift_ms=BOUNDARY_SHIFT_C)` helper. With C+shift, LUDB val avg Martinez F1 rises 0.757 → 0.779 (+0.022); ISP test 0.911 → 0.922 (+0.011). p_off is no longer the LUDB outlier; remaining gap to DENS-ECG is uniform ~7-14pp Se across P/QRS/T (likely capacity / data scale).

Signal-aware trim was less effective because P-wave's gradual return to baseline isn't crisply distinguishable from baseline noise at the std level. The bias is the model learning to extend P inclusively — fixing it via a learned p_off head (Stage 3 boundary refinement) is the principled path; the shift is a deployment workaround.

### Stage 2 v4 — model capacity scan (`scripts/train_v4_bigger.py`)

| Model | params | LUDB val avg Martinez F1 | ISP test avg F1 |
|---|---|---|---|
| **C (d=128/L=8, current v4)** | **1.08M** | **0.779** | **0.922** |
| Cbig d=192/L=10 | 2.51M | 0.780 (+0.001) | 0.919 (-0.003) |
| Cbig d=256/L=8 | 3.21M | 0.783 (+0.004) | 0.918 (-0.004) |

3× more parameters yields essentially zero improvement on LUDB val (+0.004 within noise) and a slight decrease on ISP test. **LUDB train scale (1908 sequences) is the bottleneck, not model capacity.** Closing the remaining ~7-14pp Se gap to DENS-ECG SOTA needs more data, not more parameters: Stage 4 SSL pretraining (Icentia 11k / MIMIC-IV / SNUH per PLAN.md), augmentation re-test now that ISP labels are clean, or a Stage 3 learned boundary refinement head. Keep C at d=128/L=8 as the operating point.

### Stage 2 v3 investigation — QTDB label sparsity

**QTDB T-wave F1 ≈0.49 was a label-sparsity artifact** (`scripts/eval_qtdb_t_annotated.py`): q1c annotates QRS+P on essentially every examined beat but T on only ~half. Per-record windowed T:QRS ratio: median **0.00**, mean 0.45 (only 39 of 105 records have ratio ≥ 0.8). The model correctly predicts T at every beat but unannotated beats become FP. Re-evaluating only on the 39-record T-annotated subset:

| Boundary | C full QTDB | C T-subset | Δ |
|---|---|---|---|
| qrs_on / qrs_off | 0.858 / 0.861 | **0.957 / 0.958** | +0.099 / +0.098 |
| t_on / t_off | 0.492 / 0.828 | **0.858 / 0.866** | **+0.366 / +0.038** |
| p_on / p_off | 0.799 / 0.799 | 0.803 / 0.801 | +0.004 / +0.002 |

C on the T-annotated subset reaches QRS F1 ~0.96 vs Martinez QTDB ~0.99 — within 0.03 of literature SOTA. Use the T-subset numbers when comparing to QTDB-based papers.

**Post-processing defaults tuned**: `scripts/tune_postproc_v4.py` swept `min_duration_ms × merge_gap_ms` on LUDB val and found `(60, 200)` beats the previous `(40, 300)` by avg +0.010 boundary F1 on C and +0.022 on F. Defaults updated in `openecg/stage2/infer.py`. The remaining LUDB gap to literature (~0.10) is from model capacity / data scale, not post-proc tuning.

### Stage 2 v4 — per-lead robustness (12 leads on LUDB val)

`scripts/per_lead_v4.py` breaks down boundary F1 per LUDB lead. The deployment goal is a single-lead model that works on any of the 12 leads (Holter, wearable, ICU monitor).

| Boundary F1 std across 12 leads | C (combined) | F (LUDB only) |
|---|---|---|
| qrs_on | **0.011** | 0.007 |
| qrs_off | **0.008** | 0.007 |
| p_on | **0.029** | 0.036 |
| p_off | **0.030** | 0.040 |
| t_on | **0.020** | 0.049 |
| t_off | **0.030** | 0.051 |

C is more uniformly robust on P/T (std 0.020–0.030 vs F's 0.040–0.051). QRS robustness is similar (both std ≤0.011). The weakest single leads are physiologically expected: lead **III** (small P/T amplitude due to electrical axis orthogonality) and **aVL** (small P), which are uncommon as sole monitoring leads in clinical practice.

Per-lead median timing error meets the ≤20ms spec target on QRS (8–12ms across all 12 leads for both C and F) and on P/T for most leads (8–16ms; p_off occasionally hits 28–40ms on V1–V3).

## Setup

```bash
uv sync
$env:UV_LINK_MODE = "copy"     # Windows + OneDrive workaround
$env:OPENECG_LUDB_ZIP = "<path-to-LUDB-zip>"

uv run pytest                              # 65 tests (50 unit + 15 stage2 + LUDB integration if env set)

# Stage 1 (NK baseline tokenization)
uv run python scripts/tokenize_ludb.py     # → data/ludb_tokens.npz
uv run python scripts/validate_v1.py       # → out/validation_v1_*.json
uv run python scripts/ablate_methods.py    # → out/ablation_*.json
uv run python scripts/validate_pacer.py    # console only

# Stage 2 (neural frame classifier, requires CUDA)
uv run python scripts/train_stage2.py      # → data/checkpoints/stage2_v1.pt (~35s on RTX 4090)
uv run python scripts/validate_stage2.py   # → out/validation_stage2_*.json
```
