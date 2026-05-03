# ECGCode

*Clinically-grounded discrete tokenization for electrocardiograms.*

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

## Setup

```bash
uv sync
$env:UV_LINK_MODE = "copy"     # Windows + OneDrive workaround
$env:ECGCODE_LUDB_ZIP = "<path-to-LUDB-zip>"

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
