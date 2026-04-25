"""
Tests for ``models/fusion.py::soft_iou_loss`` and ``compute_loss``.

Covers monotonicity, boundary behaviour, weight scaling, and gradient flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.fusion import compute_loss, soft_iou_loss


# ── soft_iou_loss — boundary behaviour ──────────────────────────────────────

def test_soft_iou_perfect_overlap_near_zero() -> None:
    target = torch.zeros(1, 14, 14)
    target[0, 5:9, 5:9] = 1.0
    logits = torch.where(target > 0, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    assert soft_iou_loss(logits, target).item() < 0.01


def test_soft_iou_complete_miss_near_one() -> None:
    target = torch.zeros(1, 14, 14)
    target[0, 0:4, 0:4] = 1.0
    logits = torch.where(target > 0, torch.full_like(target, -10.0), torch.full_like(target, 10.0))
    assert soft_iou_loss(logits, target).item() > 0.9


def test_soft_iou_partial_overlap_between_bounds() -> None:
    target = torch.zeros(1, 14, 14)
    target[0, 0:4, 0:4] = 1.0
    # Roughly half the patches correct
    logits = torch.randn(1, 14, 14)
    loss = soft_iou_loss(logits, target).item()
    assert 0.0 <= loss <= 1.0


def test_soft_iou_empty_target() -> None:
    """With target=0 everywhere, any nonzero prediction → high loss (union = pred area)."""
    target = torch.zeros(1, 14, 14)
    logits = torch.ones(1, 14, 14) * 5.0
    loss = soft_iou_loss(logits, target).item()
    assert loss > 0.9


def test_soft_iou_empty_pred_nonzero_target() -> None:
    """Pred all zero, target nonzero → loss ≈ 1."""
    target = torch.zeros(1, 14, 14)
    target[0, 3:8, 3:8] = 1.0
    logits = torch.ones(1, 14, 14) * -10.0
    loss = soft_iou_loss(logits, target).item()
    assert loss > 0.9


# ── soft_iou_loss — structural invariants ───────────────────────────────────

def test_soft_iou_returns_scalar_tensor() -> None:
    logits = torch.randn(2, 14, 14)
    target = torch.zeros(2, 14, 14)
    loss = soft_iou_loss(logits, target)
    assert loss.dim() == 0


def test_soft_iou_bounded_zero_to_one() -> None:
    """Over many random inputs, soft_iou must stay in [0, 1]."""
    torch.manual_seed(0)
    for _ in range(10):
        logits = torch.randn(4, 14, 14) * 3
        target = (torch.rand(4, 14, 14) > 0.5).float()
        loss = soft_iou_loss(logits, target).item()
        assert 0.0 - 1e-6 <= loss <= 1.0 + 1e-6


def test_soft_iou_differentiable() -> None:
    logits = torch.randn(1, 14, 14, requires_grad=True)
    target = torch.zeros(1, 14, 14)
    target[0, 3:8, 3:8] = 1.0
    loss = soft_iou_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_soft_iou_decreases_as_pred_improves() -> None:
    """Moving the heatmap closer to the target should lower the loss."""
    target = torch.zeros(1, 14, 14)
    target[0, 3:8, 3:8] = 1.0
    wrong = torch.randn(1, 14, 14) * 0.1   # near uniform 0.5 after sigmoid
    better = target * 5 - 2                # strongly correlated with target
    loss_wrong = soft_iou_loss(wrong, target).item()
    loss_better = soft_iou_loss(better, target).item()
    assert loss_better < loss_wrong


# ── compute_loss — dict contract ────────────────────────────────────────────

def test_compute_loss_returns_required_keys() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True)
    assert set(out.keys()) == {"total", "bce", "loc"}


def test_compute_loss_components_are_scalars() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True)
    for k, v in out.items():
        assert v.dim() == 0, f"{k} must be scalar, got shape {tuple(v.shape)}"


# ── compute_loss — weighting ────────────────────────────────────────────────

def test_compute_loss_bce_weight_scales_total() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    labels = torch.tensor([0, 1])
    base = compute_loss(scs, hm, labels, tm, has_heatmap=False, bce_weight=1.0)
    doubled = compute_loss(scs, hm, labels, tm, has_heatmap=False, bce_weight=2.0)
    assert abs(doubled["total"].item() - 2 * base["total"].item()) < 1e-5


def test_compute_loss_loc_weight_scales_total() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    tm[1, 3:8, 3:8] = 1.0    # incoherent sample has a target
    labels = torch.tensor([0, 1])
    base = compute_loss(scs, hm, labels, tm, has_heatmap=True, bce_weight=0.0, loc_weight=1.0)
    doubled = compute_loss(scs, hm, labels, tm, has_heatmap=True, bce_weight=0.0, loc_weight=2.0)
    assert abs(doubled["total"].item() - 2 * base["total"].item()) < 1e-5


def test_compute_loss_zero_weights_zero_total() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True, bce_weight=0.0, loc_weight=0.0)
    assert out["total"].item() == 0.0


# ── compute_loss — localization gating ─────────────────────────────────────

def test_compute_loss_skips_loc_when_flag_off() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.ones(2, 14, 14)   # would produce large loc loss if active
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=False)
    assert out["loc"].item() == 0.0
    assert out["total"].item() == out["bce"].item()


def test_compute_loss_skips_loc_all_coherent_batch() -> None:
    scs = torch.zeros(3)
    hm = torch.zeros(3, 14, 14)
    tm = torch.zeros(3, 14, 14)
    labels = torch.zeros(3, dtype=torch.long)
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True)
    assert out["loc"].item() == 0.0


def test_compute_loss_loc_nonzero_when_incoherent_sample_present() -> None:
    scs = torch.zeros(2)
    hm = torch.randn(2, 14, 14)  # random predictions
    tm = torch.zeros(2, 14, 14)
    tm[1, 3:8, 3:8] = 1.0
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True, loc_weight=1.0)
    assert out["loc"].item() > 0.0


def test_compute_loss_gradient_flows_from_total() -> None:
    scs = torch.randn(2, requires_grad=True)
    hm = torch.randn(2, 14, 14, requires_grad=True)
    tm = torch.zeros(2, 14, 14)
    tm[1, 3:8, 3:8] = 1.0
    labels = torch.tensor([0, 1])
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True)
    out["total"].backward()
    assert scs.grad is not None and scs.grad.abs().sum().item() > 0
    assert hm.grad is not None and hm.grad.abs().sum().item() > 0


def test_compute_loss_bce_reacts_to_wrong_prediction() -> None:
    """High BCE when logits contradict labels; low BCE when they match."""
    labels = torch.tensor([1, 0], dtype=torch.long)
    wrong_scs = torch.tensor([-10.0, 10.0])  # opposite of labels
    right_scs = torch.tensor([10.0, -10.0])  # match labels
    hm = torch.zeros(2, 14, 14); tm = torch.zeros(2, 14, 14)
    loss_wrong = compute_loss(wrong_scs, hm, labels, tm, has_heatmap=False)["bce"].item()
    loss_right = compute_loss(right_scs, hm, labels, tm, has_heatmap=False)["bce"].item()
    assert loss_wrong > loss_right
    assert loss_right < 0.01


def test_compute_loss_bce_with_long_labels_works() -> None:
    """compute_loss internally casts labels.float() — must accept long labels."""
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14); tm = torch.zeros(2, 14, 14)
    labels = torch.tensor([0, 1], dtype=torch.long)
    out = compute_loss(scs, hm, labels, tm, has_heatmap=False)
    assert out["bce"].item() > 0  # log(2) ≈ 0.693


def test_compute_loss_total_equals_weighted_sum() -> None:
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    tm = torch.zeros(2, 14, 14)
    tm[1, 3:8, 3:8] = 1.0
    labels = torch.tensor([0, 1])
    bw, lw = 0.7, 1.3
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True, bce_weight=bw, loc_weight=lw)
    expected = bw * out["bce"].item() + lw * out["loc"].item()
    assert abs(out["total"].item() - expected) < 1e-5
