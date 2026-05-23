# openecg v56c vs NeuroKit2 / WTdelineator

Direct, reproducible comparison of openecg's current boundary detector
(**v56c** — 0.99M-param 1-channel Conv+Transformer, soft-T α=0.9) against
two widely used open-source ECG delineation baselines:

- **NeuroKit2** (`nk.ecg_delineate(method="dwt")`) — DWT-based.
- **WTdelineator** (Ledezma reimplementation of Martínez 2004) —
  multi-scale wavelet delineation, the canonical academic baseline.

All three see the same inputs, are scored with the same Martínez 2004
per-boundary tolerances (P 50 ms, QRS 40 ms, T_on 50 ms, T_off 100 ms),
and metrics are aggregated identically. The script is in
[`scripts/benchmark_v56c.py`](../../scripts/benchmark_v56c.py).

## TL;DR

| Dataset | n records | openecg v56c | NeuroKit2 DWT | WTdelineator |
|---|---:|---:|---:|---:|
| **LUDB val** (lead II) | 41 | **0.963** | 0.788 | 0.596 |
| **ISP test** (lead II) | 72 | **0.971** | 0.703 | 0.604 |
| **QTDB T-subset** (first lead) | 44 | **0.908** | 0.605 | 0.535 |

Numbers are macro-F1 across the six P / QRS / T on/off boundaries.

**openecg v56c outperforms NeuroKit2 DWT by +0.18 to +0.30 macro-F1
on every dataset**, and outperforms WTdelineator by +0.37 across the
board. The gap is largest on **T-wave boundaries**, which are notoriously
hard for wavelet methods.

## Visual comparison

Each figure overlays the four detectors on the same ECG strip from
each benchmark dataset, lead II. P = red, QRS = blue, T = green;
shaded spans are predicted wave durations, vertical ticks at the top
mark predicted onsets and offsets.

### LUDB val record 16 — clean sinus rhythm
![LUDB comparison](../figures/v56c_vs_baselines_ludb.png)

openecg's P / QRS / T spans line up with the cardiologist
annotation. NeuroKit2 systematically places P boundaries off the
true P wave; WTdelineator drops most P and T detections after the
first beat.

### ISP test record 2 — biphasic T waves
![ISP comparison](../figures/v56c_vs_baselines_isp.png)

openecg locks onto every beat. NeuroKit2 misses the first beat
entirely and shifts QRS / T positions on most subsequent beats.
WTdelineator's T spans run far past the actual T wave end, which
is what drives its 40 ms median T_off error.

### QTDB record sel100 (MLII) — low-amplitude T
![QTDB comparison](../figures/v56c_vs_baselines_qtdb.png)

The regime where wavelet methods struggle most: T waves are small
and partially buried in baseline noise. openecg keeps tight P + QRS
spans on every beat; NeuroKit2 produces sporadic narrow T detections
far from where the T wave actually sits; WTdelineator drops two of
the four annotated beats outright.

Reproduce:

```bash
python -m scripts.viz_benchmark_v56c
# writes docs/figures/v56c_vs_baselines_{ludb,isp,qtdb}.png
```

## Setup

- **Datasets**:
  - LUDB validation split, lead II only, labeled-range filtered
    ±100 ms (so edge beats the cardiologist did not annotate don't
    inflate FP).
  - ISP test split, lead II only.
  - QTDB T-subset (records with ≥80% of QRSes having matched T
    annotations), first lead per record (Martínez 2004 convention).
- **Tolerances**: Martínez 2004 per-boundary —
  P 50 ms, QRS 40 ms, T_on 50 ms, T_off 100 ms.
- **Metric**: F1 per boundary; macro-F1 = mean of the six boundary F1s.
- **openecg v56c**: 1-channel `vit_transformer_noaux_1ch`, L8/d=128,
  trained on LUDB + QTDB + ISP + synthetic AV-block mix, soft-T α=0.9,
  rank-normalised input. Reg head refines boundaries.
- **NeuroKit2**: v0.2.13.
- **WTdelineator**: Ledezma reimplementation
  ([`third_party/WTdelineator/`](../../third_party/WTdelineator)).

## Per-boundary F1

### LUDB val (lead II, 41 records)

| Model | P_on | P_off | QRS_on | QRS_off | T_on | T_off |
|---|---:|---:|---:|---:|---:|---:|
| **openecg v56c**   | **0.969** | **0.922** | **0.988** | **0.965** | **0.942** | **0.994** |
| NeuroKit2 DWT      | 0.808 | 0.789 | 0.824 | 0.835 | 0.524 | 0.949 |
| WTdelineator       | 0.620 | 0.481 | 0.746 | 0.720 | 0.393 | 0.616 |

### ISP test (lead II, 72 records)

| Model | P_on | P_off | QRS_on | QRS_off | T_on | T_off |
|---|---:|---:|---:|---:|---:|---:|
| **openecg v56c**   | **0.959** | **0.962** | **0.996** | **0.994** | **0.932** | **0.983** |
| NeuroKit2 DWT      | 0.644 | 0.649 | 0.916 | 0.827 | 0.256 | 0.925 |
| WTdelineator       | 0.653 | 0.575 | 0.748 | 0.696 | 0.405 | 0.548 |

### QTDB T-subset (first lead, 44 records)

| Model | P_on | P_off | QRS_on | QRS_off | T_on | T_off |
|---|---:|---:|---:|---:|---:|---:|
| **openecg v56c**   | **0.908** | **0.907** | **0.966** | **0.959** | **0.797** | **0.912** |
| NeuroKit2 DWT      | 0.627 | 0.773 | 0.629 | 0.770 | 0.323 | 0.511 |
| WTdelineator       | 0.506 | 0.458 | 0.660 | 0.690 | 0.456 | 0.437 |

## Timing accuracy

Beyond F1, clinically the question is: how close are the predicted
boundary samples to ground truth? Median absolute error per boundary on
ISP test, lead II:

| Boundary | openecg v56c | NeuroKit2 DWT | WTdelineator |
|---|---:|---:|---:|
| P_on   | 4 ms  | 32 ms | 32 ms |
| P_off  | 12 ms | 24 ms | 36 ms |
| QRS_on | 4 ms  | 12 ms | 12 ms |
| QRS_off| 12 ms | 12 ms | 12 ms |
| T_on   | 12 ms | 32 ms | 32 ms |
| T_off  | 16 ms | 44 ms | 40 ms |

openecg's median timing error is **≤16 ms on every boundary**, hitting
the clinical 20 ms spec target. NeuroKit2 misses on T_off (44 ms) and
on every P boundary; WTdelineator misses on P_on, P_off, T_on, T_off.

## Deploy footprint

The exact same v56c weights are also shipped as a TFLite int8 model
bundled in the wheel (`openecg/models/boundary_int8.tflite`, 1.48 MB).
Boundary-F1 evaluated through the TFLite int8 path matches the torch
fp32 path within **0.0025 macro-F1** — quantisation is essentially
lossless on this model thanks to weight-only int8.

| Path | Size | F1 mean (LUDB+ISP+QTDB) | Latency / 10s window |
|---|---:|---:|---:|
| Torch fp32 | 4.0 MB | 0.9299 | 44 ms |
| TFLite fp32 | 4.4 MB | (matches torch) | 44 ms |
| TFLite int8 | **1.5 MB** | **0.9274** | 44 ms |

See [`memory/project_deploy_tflite.md`](../../../.claude/projects/C--Users-lucid-OneDrive-Projects-openecg/memory/project_deploy_tflite.md)
for the deploy decision and the head-to-head with ExecuTorch
(TFLite won 3.5× on latency and -0.004 less F1 loss).

## Reproduce

```bash
# Install with loaders + stage2 backbone
pip install "openecg[loaders,stage2]"

# Run the benchmark (lead II, ~30s on CPU)
python -m scripts.benchmark_v56c \
    --ckpt data/checkpoints/stage2_v45k_noaux_L8_d128_1ch_v56c.pt \
    --leads ii \
    --out out/benchmark_v56c_leadii.json

# All 12 leads (longer)
python -m scripts.benchmark_v56c --leads all --out out/benchmark_v56c_all.json
```

JSON output includes per-boundary F1 / Se / P+ / mean signed error /
SD / median absolute error / n_true / n_pred for each model × dataset,
so any sub-table here can be rebuilt from the raw file.
