# openecg codec — progression note (2026-06-11)

**Production model: `codec_v4` (openecg 0.7.0, PyPI + GitHub).** This note records the
codec_v5/v6/v7 frame-improvement exploration and the KHDP data work done this cycle.
Bottom line: **no candidate beat codec_v4 on frame; v4 remains shipped.** The
exploration produced durable diagnoses (below) that should stop us repeating dead ends.

---

## 1. Shipped — codec_v4 (0.7.0)
codec_v3 backbone + frame + rhythm heads **frozen** (byte-identical), **beat head
retrained** on vitaldb VPC (rhythm masked). Zero regression, one win:
- frame: boundary macro-F1 **0.829** / median **11.6 ms** / per-sample **0.827** (LUDB, all-12-lead)
- hospital rhythm (lydus-test): macro **0.790** (avb 0.76 / paced 0.76 / afib 0.88 / bbb 0.62)
- beat: **VPC F1 0.858 → 0.935** (MIT-BIH DS2; precision 0.78 → 0.94), sinus 0.992
- `openecg.load_codec()` default = codec_v4. int8 ONNX shipped.

## 2. KHDP data redistribution (done — admin_review)
Licenses reviewed; **8 datasets submitted to KHDP with raw data** via the submission
API (`scripts/khdp_submit.ps1`): rdb-resting-ecg-segmentation (CC-BY-4.0),
idiopathic-va-12lead-outflow-tract (CC0), leipzig-heart-center-ecg, code-test-827
(CC-BY-4.0), mit-bih-pwave-annotations, spontaneous-vt-arrhythmia-db,
ecg-fragment-dangerous-arrhythmia, european-st-t-database (ODC-By-1.0).
Excluded: CPSC2020 (no redistribution license), Mendeley 6jd4rn2z9x (signals only,
labels never published), CODE-II (unreleased), IRIDIA-AF (10.5 GB, deferred).
New expert dataset survey + vitaldb integration recorded in agent memory.

## 3. Frame-improvement attempts — all FAILED to beat codec_v4
| attempt | recipe | result vs v4 (boundary-F1 0.829) |
|---|---|---|
| **v5 full** | real+lydus+vitaldb+RDB (hard) | 0.829, but rhythm 0.769↓ vpc 0.900↓ |
| **v5c** two-stage | RDB-frame-only + frozen beat | 0.816↓ |
| **v5f** frame-only-from-v4 | freeze backbone, retrain frame head on RDB | **0.777↓↓** (frame ≠ beat: frozen backbone can't help frame) |
| **v6** bundle | +RDB-12lead +augment +focal +smooth | **0.763↓↓** |
| **v6b** corrected | no-RDB-frame +hard-negatives +smooth | 0.823, over-seg WORSE (20.5%) |
| **v7** soft-boundary | gaussian-soft targets (T wide, QRS sharp) +RDB | 0.801; calibration corr ≈0.10 (class-level only) |
| **v7lv** logvar | heteroscedastic aleatoric (Kendall-Gal) | 0.804; **σ collapsed to ~0**, corr 0.04 |

## 4. Durable diagnoses (do not repeat these)
- **The frame weakness is over-segmentation + boundary precision, not detection.** Worst
  LUDB windows over-predict waves (55% pred-wave vs 29% GT); P recall 0.90 but precision 0.72.
- **codec_v4 is at the annotation-noise ceiling for LUDB hard boundary-F1.** Adding frame
  data / head tricks / loss tweaks does not exceed it.
- **RDB hurts frame via annotation-CONVENTION mismatch, not lead-mixing.** RDB annotates
  T-waves ~20% narrower than LUDB (145 vs 181 ms). Damage is concentrated in the FUZZY
  channels (P_off +0.090, P_on +0.086, T_on +0.076 recovery when RDB removed) and barely
  touches QRS (+0.05). QRS boundary is robust/agreed; P/T are inherently ambiguous
  (clinician-confirmed). `ludb_val.npz` is already all-12-lead, so this is not a lead issue.
- **Hard-negative synthetic noise does NOT fix LUDB over-seg** — LUDB's over-seg is boundary
  leakage on CLEAN signal, not noise-firing.
- **soft-boundary works only at class level** (T transition width 34 ms vs QRS 14 ms,
  matching clinical fuzziness) — not per-instance (corr ≈0.10).
- **logvar / heteroscedastic aleatoric COLLAPSES (σ→0)** on single-label delineation: the
  model fits the one per-sample label, so there is no aleatoric signal to attribute. The
  true uncertainty (inter-annotator disagreement) is not in the per-sample loss.
- **Single-channel, lead-agnostic contract** (use_lead_emb=False, no lead-id input): the
  model must delineate ANY lead; train on all leads, but cross-dataset frame conventions
  must agree or be modeled as uncertainty.

## 5. Trainer tooling added (pod_stage/kgpu_train.py, for future use)
`--soft-boundary` (+ per-class σ), `--logvar` (+ `--logvar-mc`), `--augment`,
`--frame-focal-gamma`, `--frame-smooth`, `--beat-only` / `--frame-only` (frozen-head
upgrades), `--d-model/--n-layers/--n-heads/--ff` (arch overrides). Shape-filtered
warm-start. model_variants MH gained an optional `use_logvar` head (4-tuple forward,
backward-compatible). Build/eval scripts: `scripts/build_rdb_cache.py`,
`build_vitaldb_cache.py`, `eval_boundary_sweep.py`, `eval_beat_sweep.py`,
`viz_worst_frame.py`, `pod_stage/eval_sweep.py`.

## 6. Open directions (if frame is revisited)
1. **Uncertainty as a product feature, not a boundary-F1 win** — ship a soft-trained model
   whose 4-channel softmax (entropy/transition-width) IS the boundary uncertainty
   (T fuzzy, QRS sharp). Clinically honest; does not beat v4 on argmax F1.
2. **Per-instance uncertainty needs EPISTEMIC** (MC-dropout / ensembles), not aleatoric logvar.
3. **Rhythm/beat equity** (the thesis): transformer rare-class heads (validated in v56c,
   unshipped); more expert vent data. Frame is at ceiling — invest here instead.

**Decision: finish on codec_v4 (0.7.0). Frame ceiling reached; future effort → rhythm/beat
equity + uncertainty-as-feature, not frame boundary-F1.**
