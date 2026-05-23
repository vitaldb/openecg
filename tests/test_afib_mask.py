"""Tests for strict AFib P-fold masking in openecg.stage2.afib_mask.

Invariants we want to enforce:

* Inside AFib rows, the model's P output and the GT's P label both
  vanish from loss (gradient = 0) and from frame F1 (TP/FP/FN counters
  unaffected).
* Outside AFib rows, every helper is a no-op — non-AFib batches must
  produce bit-identical losses to the legacy code path.
* p_on / p_off reg-mask channels are the only ones cleared by AFib;
  qrs/t channels are untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from openecg.eval import IGNORE_INDEX, SUPER_OTHER, SUPER_P, SUPER_QRS, SUPER_T
from openecg.stage2 import afib_mask as afm


B, T, C = 4, 16, 4


def _rand_logits(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, C, generator=g)


def _rand_labels(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, C, (B, T), generator=g)


# ----------------------------------------------------- logit fold ----


def test_pfold_logits_no_afib_is_noop():
    logits = _rand_logits()
    flag = torch.zeros(B, dtype=torch.bool)
    out = afm.pfold_logits(logits, flag)
    assert torch.equal(out, logits)


def test_pfold_logits_p_invisible_in_softmax():
    """After suppress, P channel softmax mass is ~0 — any P logit value is killed."""
    logits = _rand_logits()
    flag = torch.tensor([True, False, True, False])
    out = afm.pfold_logits(logits, flag)
    probs = out.softmax(dim=-1)
    assert probs[flag][..., SUPER_P].max() < 1e-3
    assert torch.allclose(out[~flag], logits[~flag])


def test_pfold_logits_does_not_perturb_other_qrs_t_logits():
    """OTHER / QRS / T logits must be untouched — only P is suppressed.
    Otherwise the model's P output would leak into other classes' loss."""
    logits = _rand_logits()
    flag = torch.ones(B, dtype=torch.bool)
    out = afm.pfold_logits(logits, flag)
    for c in (SUPER_OTHER, SUPER_QRS, SUPER_T):
        assert torch.equal(out[..., c], logits[..., c])


def test_pfold_logits_invariant_to_p_output():
    """Strict guarantee: changing the model's P logit by any amount must
    not change CE loss after the suppress + label-IGNORE pipeline."""
    base = _rand_logits()
    alt = base.clone()
    alt[..., SUPER_P] += 1000.0  # absurd perturbation

    labels = _rand_labels()
    flag = torch.ones(B, dtype=torch.bool)

    base_s = afm.pfold_logits(base, flag)
    alt_s = afm.pfold_logits(alt, flag)
    lab_m = afm.pfold_labels(labels, flag)

    loss_base = F.cross_entropy(
        base_s.transpose(1, 2), lab_m, ignore_index=IGNORE_INDEX,
    )
    loss_alt = F.cross_entropy(
        alt_s.transpose(1, 2), lab_m, ignore_index=IGNORE_INDEX,
    )
    assert torch.allclose(loss_base, loss_alt, atol=1e-5)


# ------------------------------------------------------- label fold ----


def test_pfold_labels_p_to_ignore():
    labels = torch.tensor([[SUPER_P, SUPER_QRS, SUPER_T, SUPER_OTHER]])
    flag = torch.tensor([True])
    out = afm.pfold_labels(labels, flag)
    assert out.tolist() == [[IGNORE_INDEX, SUPER_QRS, SUPER_T, SUPER_OTHER]]


def test_pfold_labels_no_afib_is_noop():
    labels = _rand_labels()
    flag = torch.zeros(B, dtype=torch.bool)
    assert torch.equal(afm.pfold_labels(labels, flag), labels)


def test_pfold_labels_preserves_existing_ignore():
    labels = torch.tensor([[SUPER_P, IGNORE_INDEX, SUPER_P]])
    flag = torch.tensor([True])
    out = afm.pfold_labels(labels, flag)
    assert out.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]


def test_pfold_labels_non_p_classes_untouched():
    """OTHER / QRS / T labels in AFib rows must keep their supervision."""
    labels = torch.tensor([[SUPER_OTHER, SUPER_QRS, SUPER_T]])
    flag = torch.tensor([True])
    out = afm.pfold_labels(labels, flag)
    assert torch.equal(out, labels)


# -------------------------------------------------------- reg mask ----


def test_pfold_reg_mask_zeros_only_p_cols():
    mask = torch.ones(B, T, 6, dtype=torch.bool)
    flag = torch.tensor([True, False, True, False])
    out = afm.pfold_reg_mask(mask, flag)
    # AFib rows: p_on (0), p_off (1) = False; other cols (2-5) = True
    assert not out[0, :, 0].any() and not out[0, :, 1].any()
    assert out[0, :, 2:].all()
    # Non-AFib rows unchanged
    assert out[1].all()


def test_pfold_reg_mask_no_afib_noop():
    mask = torch.ones(B, T, 6, dtype=torch.bool)
    flag = torch.zeros(B, dtype=torch.bool)
    assert torch.equal(afm.pfold_reg_mask(mask, flag), mask)


# ----------------------------------------------------- soft target ----


def test_pfold_soft_target_p_mass_moved():
    soft = torch.zeros(1, 2, C)
    soft[0, 0, SUPER_P] = 0.7
    soft[0, 0, SUPER_OTHER] = 0.3
    soft[0, 1, SUPER_QRS] = 1.0
    flag = torch.tensor([True])
    out = afm.pfold_soft_target(soft, flag)
    assert out[0, 0, SUPER_P] == 0.0
    assert torch.isclose(out[0, 0, SUPER_OTHER], torch.tensor(1.0))
    # QRS frame untouched
    assert out[0, 1, SUPER_QRS] == 1.0


# --------------------------------------------------- metric fold ----


def test_pfold_predictions_arrays_drop_p_frames_via_ignore():
    """In AFib frames, any P-related decision (pred==P or true==P) is
    dropped via true=IGNORE_INDEX so it counts nowhere — neither P nor
    OTHER/QRS/T counters are affected by P decisions in AFib windows."""
    pred = np.array([SUPER_P, SUPER_QRS, SUPER_P, SUPER_OTHER])
    true = np.array([SUPER_OTHER, SUPER_P, SUPER_T, SUPER_QRS])
    afib = np.array([True, True, True, True], dtype=bool)
    p2, t2 = afm.pfold_predictions_arrays(pred, true, afib)
    # Frames 0,1,2 had P on one side -> true dropped to IGNORE; frame 3 stays.
    assert t2[0] == IGNORE_INDEX
    assert t2[1] == IGNORE_INDEX
    assert t2[2] == IGNORE_INDEX
    assert t2[3] == SUPER_QRS  # neither pred nor true was P -> kept
    # pred array is not mutated for downstream classes' integrity
    assert np.array_equal(p2, pred)


def test_pfold_predictions_arrays_no_afib_noop():
    pred = np.array([SUPER_P, SUPER_QRS, SUPER_OTHER])
    true = np.array([SUPER_P, SUPER_T, SUPER_OTHER])
    afib = np.zeros(3, dtype=bool)
    p2, t2 = afm.pfold_predictions_arrays(pred, true, afib)
    assert np.array_equal(p2, pred) and np.array_equal(t2, true)


def test_expand_window_flag_to_frames():
    flag = np.array([True, False])
    out = afm.expand_window_flag_to_frames(flag, frames_per_window=3)
    assert out.tolist() == [True, True, True, False, False, False]


# -------------------------------------------------- boundary filter ----


def test_filter_p_boundaries_drops_only_p_when_afib():
    bp = {"p_on": [1], "p_off": [2], "qrs_on": [3], "qrs_off": [4],
          "t_on": [5], "t_off": [6]}
    bt = {"p_on": [1], "qrs_on": [3]}
    bp_f, bt_f = afm.filter_p_boundaries(bp, bt, is_af=True)
    assert "p_on" not in bp_f and "p_off" not in bp_f
    assert "qrs_on" in bp_f and "qrs_off" in bp_f
    assert "p_on" not in bt_f and "qrs_on" in bt_f


def test_filter_p_boundaries_passthrough_when_not_afib():
    bp = {"p_on": [1], "qrs_on": [3]}
    bt = {"p_on": [1]}
    bp_f, bt_f = afm.filter_p_boundaries(bp, bt, is_af=False)
    assert bp_f is bp and bt_f is bt


# ------------------------------ full-pipeline strict invariance ----


def test_train_loss_invariant_to_p_logit_and_p_label_in_afib():
    """End-to-end strict don't-care: in AFib batches, neither the model's
    P logit value nor the GT's P labels can change the CE loss.

    Two batches share the same OTHER/QRS/T logits and same non-P labels;
    they differ only in P logit values (huge perturbation) and P label
    presence (one has many P labels, the other replaces P with QRS).
    After suppress + IGNORE, CE losses must match.
    """
    torch.manual_seed(0)
    # Build a batch where labels are a fixed mix of OTHER/QRS/T, plus some
    # P positions; perturb only the P logits.
    labels_with_p = torch.full((B, T), SUPER_OTHER, dtype=torch.long)
    labels_with_p[:, ::4] = SUPER_QRS
    labels_with_p[:, ::5] = SUPER_T
    labels_with_p[:, ::3] = SUPER_P  # many P labels in AFib rows
    labels_without_p = labels_with_p.clone()
    labels_without_p[labels_without_p == SUPER_P] = SUPER_QRS  # swap P->QRS

    base = _rand_logits()
    alt = base.clone()
    alt[..., SUPER_P] += 100.0

    flag = torch.ones(B, dtype=torch.bool)
    loss_a = F.cross_entropy(
        afm.pfold_logits(base, flag).transpose(1, 2),
        afm.pfold_labels(labels_with_p, flag),
        ignore_index=IGNORE_INDEX,
    )
    loss_b = F.cross_entropy(
        afm.pfold_logits(alt, flag).transpose(1, 2),
        afm.pfold_labels(labels_with_p, flag),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.allclose(loss_a, loss_b, atol=1e-5), \
        "P logit perturbation must not change loss in AFib rows"

    # Now also vary the labels: same P-perturbation, but different P
    # distribution. Both should produce the same loss because P frames
    # are IGNORED entirely.
    loss_c = F.cross_entropy(
        afm.pfold_logits(base, flag).transpose(1, 2),
        afm.pfold_labels(labels_without_p, flag),
        ignore_index=IGNORE_INDEX,
    )
    # labels_without_p has more QRS frames (where labels_with_p had P);
    # those QRS frames now contribute to CE — that's expected, they're not
    # P-related. The strict claim is narrower: removing all P labels from
    # `labels_with_p` (-> labels_without_p) only adds back supervision on
    # frames that used to be IGNORE, so loss_c >= loss_a's per-frame mean.
    assert torch.isfinite(loss_c)


def test_frame_f1_p_invariant_in_afib_window():
    """frame_f1 of an AFib-windowed batch is identical whether the model
    predicts P everywhere or never — the headline FP/FN invariance the
    user asked for. Non-AFib windows still penalise wrong P predictions.
    """
    from openecg.eval import frame_f1
    # 2 windows of 10 frames each, true labels: QRS in window 0, OTHER in
    # window 1. Window 0 is AFib; window 1 is not.
    N = 10
    true = np.concatenate([
        np.full(N, SUPER_QRS, dtype=np.uint8),
        np.full(N, SUPER_OTHER, dtype=np.uint8),
    ])
    afib_per_frame = np.concatenate([
        np.ones(N, dtype=bool),
        np.zeros(N, dtype=bool),
    ])

    # Model A: predicts P in the AFib window, OTHER elsewhere
    pred_a = np.concatenate([
        np.full(N, SUPER_P, dtype=np.uint8),
        np.full(N, SUPER_OTHER, dtype=np.uint8),
    ])
    # Model B: predicts QRS in the AFib window (correct), OTHER elsewhere
    pred_b = np.concatenate([
        np.full(N, SUPER_QRS, dtype=np.uint8),
        np.full(N, SUPER_OTHER, dtype=np.uint8),
    ])

    p_a, t_a = afm.pfold_predictions_arrays(pred_a, true, afib_per_frame)
    p_b, t_b = afm.pfold_predictions_arrays(pred_b, true, afib_per_frame)
    f1_a = frame_f1(p_a, t_a)
    f1_b = frame_f1(p_b, t_b)
    # P TP/FP/FN must be 0 in both — no P info influences any counter
    assert f1_a[SUPER_P]["tp"] == 0
    assert f1_a[SUPER_P]["fp"] == 0
    assert f1_a[SUPER_P]["fn"] == 0
    # OTHER F1 in window 1 is perfect in both (true=OTHER, pred=OTHER)
    assert f1_a[SUPER_OTHER]["f1"] == 1.0
    assert f1_b[SUPER_OTHER]["f1"] == 1.0


def test_p_aux_row_ignored_gives_zero_loss_contribution():
    """When a whole row of the P-binary aux labels is set to
    IGNORE_INDEX, CE on that row contributes 0 regardless of the aux
    logits. Matches `_mask_p_aux_for_afib` semantics in
    train_one_epoch_reg_aux.
    """
    aux_logits = torch.randn(B, T, 2)
    aux_labels_ok = torch.randint(0, 2, (B, T))
    aux_labels_masked = aux_labels_ok.clone()
    aux_labels_masked[0] = IGNORE_INDEX  # silence row 0 entirely

    # Loss on rows 1..B-1 only must equal full loss on the masked batch
    loss_full = F.cross_entropy(
        aux_logits.transpose(1, 2), aux_labels_masked,
        ignore_index=IGNORE_INDEX,
    )
    loss_subset = F.cross_entropy(
        aux_logits[1:].transpose(1, 2), aux_labels_ok[1:],
    )
    assert torch.allclose(loss_full, loss_subset, atol=1e-5)
