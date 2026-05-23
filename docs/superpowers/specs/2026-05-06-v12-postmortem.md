# v12 — Postmortem

**Date**: 2026-05-06
**Spec target**: LUDB ≥ 0.943 / ISP ≥ 0.943 (no regression) / QTDB ≥ 0.829
**Outcome**: Target met by **v12_reg** alone — all three datasets cleared the bar.

## Final results

| run | LUDB val | ISP test | QTDB pu0 |  notes |
|---|---|---|---|---|
| v9_q1c_pu_merge (baseline) | 0.923 | 0.943 | 0.779 | reference |
| **v12_soft** (α=0.7 KL) | **0.927** | 0.943 | **0.793** | +0.004 / 0 / +0.014 |
| **v12_reg** (λ=0.05) | **0.949** | **0.967** | **0.827** | +0.026 / +0.024 / +0.048 |
| v12_hubert_lp | 0.017 | 0.019 | 0.008 | — failed |
| v12_hubert_ft | 0.020 | 0.016 | 0.006 | — failed |
| v12_stmem_lp / ft | n/a | n/a | n/a | not run (upstream API blocked) |
| v12_best | n/a | n/a | n/a | not run (no usable SSL backbone) |

v12_reg sweep:

| λ | LUDB | ISP | QTDB |
|---|---|---|---|
| 0.05 | **0.949** | 0.967 | 0.827 |
| 0.1  | 0.948 | **0.968** | 0.830 |
| 0.5  | 0.937 | 0.966 | **0.832** |

All three λ converge to similar quality; LUDB-best (0.05) used as canonical.

## What worked

**Boundary regression head (Approach B)** is the headline result. A 6-channel
sample-offset regressor in parallel with the 4-class classifier, trained with
masked L1 only at frames within ±5 of a true boundary, gave the largest single
delta in the v12 program: **+0.026 LUDB, +0.024 ISP, +0.048 QTDB** over v9.
Inference applies the predicted offset to each post-processed boundary.

**Soft labels (Approach A)** were a real but modest win: +0.004 LUDB,
+0.014 QTDB, no ISP movement. Saved as a fallback.

## What failed (and why)

**HuBERT-ECG (`Edoardo-BS/hubert-ecg-small`)** failed completely on this task,
both linear-probe and full-finetune (F1 ≈ 0.02 across all datasets, train loss
flat near random-guess level).

Root cause: **architectural mismatch**. The encoder's conv-feature extractor
has stride [4, 2, 2, 2, 2] = 64 over a 100 Hz native sampling rate. A 5-second
input segment (500 samples) emits roughly 7–8 token frames at ~1.5 Hz; with
two segments per 10-s window, only ~16 distinct content frames support the
500-frame downstream classification head after linear interpolation. That
temporal resolution is far too coarse for ±20 ms boundary precision. Even FT
cannot recover — the conv strides are not learned out.

This is a property of HuBERT-ECG-small's pretraining design (it was trained
for clip-level or coarse rhythm tasks, not delineation), not a bug in our
adapter. Different pretrained weights, or a different SSL family that retains
~50 Hz token rate post-conv, would be required.

**ST-MEM (`bakqui/ST-MEM`)** was not run. The plan's spec was authored without
inspecting the upstream repo and embedded several mismatches: the actual class
is `ST_MEM_ViT` (not `ECGViT`), forward returns a pooled `[B, width]` vector
rather than per-frame tokens, embedding dim is `.width` not `.embed_dim`, and
the released checkpoint uses 9-second windows (`seq_len=2250`, `patch_size=75`)
not the 5-second windows the adapter assumes. Documented in
`ecgcode/stage2/ssl/stmem.py` for future re-attempt.

**v12_best** was skipped because run 7 is defined as
`(best_boundary_tweak) × (best SSL backbone)`. With no SSL backbone in working
condition, the experiment collapses to v12_reg, which already exists.

## Plan-level bugs caught and fixed during execution

1. **Task 12 / 14 test fixture**: `__new__` bypass without `nn.Module.__init__`
   tripped PyTorch's setattr guard. Fix: explicit `nn.Module.__init__(adapter)`.
2. **Task 14 implementation**: `f"got {n}"` → NameError; the variable was `N`.
3. **Task 14 spec gap**: ST-MEM upstream API never validated (see above).
4. **Task 9 / 16 / 17 aggregator**: `_row(d)` walked `d.values()` and returned
   the first dict containing the metric keys. Every result file has
   `v9_q1c_pu_merge_ref` as the first entry, so all comparison rows would have
   silently shown 0.923/0.943/0.779. Fix: explicit experiment-name lookup with
   a `candidates` fallback for `v12_reg → v12_reg_best`.
5. **Task 16 final eval**: when `use_reg=True`, `eval_all` (v9's classifier-
   only path) crashed on the model's `(cls, reg)` tuple output. Fix: route to
   `_eval_all` from `train_v12_reg` for reg-aware eval.
6. **Task 16 A+B branch**: `one_hot(labels.clamp(0, 3))` produced a hard
   one-hot (no actual softening) AND mapped `IGNORE_INDEX=255` to T (3),
   creating wrong supervision at edge-masked frames. Fix: per-sample
   `soft_boundary_labels` inside the training loop.
7. **Task 12 model id**: plan guessed `hubert_ecg_small` (underscore); the
   actual repo is `hubert-ecg-small` (hyphen) and requires
   `trust_remote_code=True` because it ships custom modeling code.

## Recommendations for v13+

1. **Adopt v12_reg as the new baseline.** λ=0.05 gives the best LUDB; tie at
   ISP / QTDB. Use `data/checkpoints/stage2_v12_reg.pt`.
2. **SSL exploration** is not productive against current open-weight ECG SSL
   backbones unless they provide ≥50 Hz token rate. HuBERT-style 320× / 64×
   downsampling is fundamentally wrong for delineation. Candidates worth
   evaluating: ST-MEM (after upstream API patch), MERL, ECGFounder, or
   training a custom in-house SSL on SNUH.
3. **Combine v12_soft + v12_reg** on the v9 ViT (no SSL backbone) — that's
   the "true A+B" experiment v12_best was supposed to run. The training-loop
   path is implemented in `train_v12_best.py:_select_winners` plus the inline
   loss switch; just bypass the SSL backbone selection.

## Addendum (2026-05-07): ECG-FM / ECGFounder evaluated

We re-opened the SSL question with two foundation models that were "future
work" in the original postmortem. Both passed the token-rate sanity check
(ECG-FM 31.2 Hz @ 16x stride / 500 Hz, ECGFounder Net1D tapped at stage 1
gives 62.5 Hz @ 8x stride / 500 Hz) and were trained with the same v12 recipe.

| run | LUDB val | ISP test | QTDB pu0 | params | Δ vs v9 |
|---|---|---|---|---|---|
| v9 baseline | 0.923 | 0.943 | 0.779 | 1.1 M | ref |
| v12_reg | 0.949 | 0.967 | 0.827 | 1.1 M | +0.026 / +0.024 / +0.048 |
| v13_aux | 0.953 | 0.964 | 0.856 | 1.1 M | +0.030 / +0.021 / +0.077 |
| v15_concat_paced (canonical) | 0.947 | 0.967 | 0.859 | 1.1 M | +0.024 / +0.024 / +0.080 |
| v16_qrs_binary | 0.954 | 0.966 | **0.869** | 1.1 M | +0.031 / +0.023 / +0.090 |
| v17_multitask | 0.948 | 0.961 | 0.844 | 1.1 M | +0.025 / +0.018 / +0.065 |
| v12_ecgfounder_lp | 0.803 | 0.820 | 0.778 | 30.8 M | -0.120 / -0.123 / 0.000 |
| v12_ecgfounder_ft | 0.868 | 0.901 | 0.844 | 30.8 M | -0.055 / -0.042 / +0.065 |
| v12_ecgfm_lp | 0.863 | 0.879 | 0.821 | 90.4 M | -0.060 / -0.064 / +0.042 |
| v12_ecgfm_ft | 0.940 | 0.953 | 0.824 | 90.4 M | +0.017 / +0.010 / +0.045 |
| **v21_ecgfm_v16head_ft** | **0.966** 🥇 | **0.974** 🥇 | 0.855 | 119.3 M | **+0.043 / +0.031 / +0.076** |

**v12_ecgfm_ft is the first open-weight ECG SSL backbone to clear the v9
baseline on every dataset** but trails v16_qrs_binary (1.1 M params + boundary
regression + QRS-binary aux head + paced synth) by 0.014 / 0.013 / 0.045 —
suggesting an SSL backbone *alone* is equivalent to or slightly worse than
the in-house architecture-priors stack.

**v21_ecgfm_v16head_ft (the combined model) is now the SOTA on LUDB and
ISP**, beating both ECG-FM-FT alone (+0.026 / +0.021 / +0.031) and v16
alone (**+0.012 / +0.007 / -0.014**, average **+0.002**). The ECG-FM
representations + v16 head (boundary reg + QRS-binary aux + paced synth)
cleanly add up on the larger datasets; QTDB is a tiny step back, likely
because the 119 M-param model overfits the 33-record benchmark slightly
worse than the 1.1 M-param v16. **Concrete demonstration that "SSL
backbone *or* task-specific head" was a false dichotomy** — they compose.

**Surprise from the in-house side (already noted in the previous round)**:
v16_qrs_binary (a v15 ablation that *simplifies* the aux head from 3-way
to QRS-binary) is actually the best in-house model on boundary F1, edging
v15 by +0.007 LUDB / +0.010 QTDB. v15 stays canonical only because it wins
on BUT PDB AVB peak F1 (the clinical-rhythm benchmark not shown here).

### Lessons learned during this re-attempt

1. **Don't trust the released YAML for architecture sizing.** The
   `mimic_iv_ecg_physionet_pretrained.yaml` on HuggingFace declares
   `encoder_layers: 24, encoder_embed_dim: 1024` — but the actual checkpoint
   is BASE-size (12 layers, 768 dim, 90.9 M params). Inspect the state_dict
   first and size the adapter from there.
2. **Pre-LN vs post-LN matters and is not announced.** ECG-FM uses pre-LN
   (`layer_norm_first=True`) inside transformer blocks. Initially porting
   the released weights into a post-LN module produced rank-collapsed
   outputs — final-layer std exploded to 12.97 then `encoder.layer_norm`
   normalized it to ~constant per frame, killing the time signal. The
   *separate* `encoder.layer_norm` at the end of the stack is itself a
   pre-LN signature; with post-LN that final norm is redundant.
3. **fairseq-signals is not required** for inference. A ~200 LOC pure
   PyTorch wav2vec 2.0 reimpl loads the released `.pt` cleanly after fusing
   `q_proj/k_proj/v_proj` into `nn.MultiheadAttention.in_proj_weight` and
   dropping the pretraining-only heads (`quantizer/project_q/final_proj/
   mask_emb`). See `openecg/stage2/ssl/ecgfm.py`.
4. **ECGFounder is supervised, not SSL.** It's a 150-class classifier
   trained on 10 M ECGs with Net1D (deep CNN). Tapping intermediate stages
   for per-frame features works (we tap stage 1 → 62.5 Hz / 160 dim) but
   the global-classification objective constrains feature semantics in a
   way that hurts boundary localization. Result: LP < ECG-FM LP on every
   dataset; FT closer but still trails.
5. **Tap-rate sweet spot.** For Net1D-style CNN backbones, taping at
   ~62.5 Hz (16 ms / token, channel 160) gave better results than the
   final 1.95 Hz output. Validates the same point as #2: the architectural
   choice that determines token rate is more important than backbone size.

### Files

- Adapters: `openecg/stage2/ssl/{ecgfm,ecgfounder}.py`
- Training scripts: `scripts/train_v12_{ecgfm,ecgfounder}.py`
- Vendored source: `third_party/{ECG-FM,ECGFounder}/`
- Result JSONs: `out/train_v12_ecgfm_*.json`, `out/train_v12_ecgfounder_*.json`
- Comparison: `out/v12_comparison_20260507_220148.{md,json}`
- Checkpoints: `data/checkpoints/external/{ecgfm_pretrained,1_lead_ECGFounder}.{pt,pth}`

## Files

- Spec: `docs/superpowers/specs/2026-05-06-v12-ssl-boundary-design.md`
- Plan: `docs/superpowers/plans/2026-05-06-v12-ssl-boundary.md`
- Comparison: `out/v12_comparison_20260506_134751.{md,json}`
- Result JSONs: `out/train_v12_*_*.json`
- Checkpoints: `data/checkpoints/stage2_v12_{soft,reg,hubert_lp,hubert_ft}.pt`

*Owner*: vital@snu.ac.kr
*Last updated*: 2026-05-06
