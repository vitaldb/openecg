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

## Setup

```bash
uv sync
$env:UV_LINK_MODE = "copy"     # Windows + OneDrive workaround
$env:ECGCODE_LUDB_ZIP = "<path-to-LUDB-zip>"

uv run pytest                              # 53 tests
uv run python scripts/tokenize_ludb.py     # → data/ludb_tokens.npz
uv run python scripts/validate_v1.py       # → out/validation_v1_*.json
uv run python scripts/ablate_methods.py    # → out/ablation_*.json
uv run python scripts/validate_pacer.py    # console only
```
