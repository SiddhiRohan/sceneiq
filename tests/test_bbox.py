"""
Tests for ``scripts/data.py::bbox_to_patch_mask``.

Exhaustively exercises the pixel-bbox → patch-mask conversion:
  * area conservation for multiple bbox sizes
  * clamping on out-of-image / negative coordinates
  * correctness of the covered patch region
  * different grid sizes and image aspect ratios
  * edge alignment (bbox sitting exactly on patch boundaries)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import torch

from data import bbox_to_patch_mask


# ── Area conservation ───────────────────────────────────────────────────────

def test_bbox_quarter_area() -> None:
    """25% bbox → sum ≈ 0.25 * 14 * 14 = 49."""
    m = bbox_to_patch_mask([50, 50, 100, 100], 200, 200, grid=14)
    assert abs(m.sum().item() - 49.0) < 1e-3


def test_bbox_half_area_horizontal() -> None:
    """Top half of the image → sum ≈ 0.5 * 14 * 14 = 98."""
    m = bbox_to_patch_mask([0, 0, 200, 100], 200, 200, grid=14)
    assert abs(m.sum().item() - 98.0) < 1e-3


def test_bbox_half_area_vertical() -> None:
    """Left half of the image → sum ≈ 98."""
    m = bbox_to_patch_mask([0, 0, 100, 200], 200, 200, grid=14)
    assert abs(m.sum().item() - 98.0) < 1e-3


def test_bbox_ninth_area() -> None:
    """1/9 area → sum ≈ 14*14/9 ≈ 21.78."""
    m = bbox_to_patch_mask([0, 0, 100, 100], 300, 300, grid=14)
    assert abs(m.sum().item() - (196.0 / 9.0)) < 1.0


def test_bbox_full_image() -> None:
    """bbox = full image → sum = grid^2, every patch = 1.0."""
    m = bbox_to_patch_mask([0, 0, 100, 100], 100, 100, grid=14)
    assert abs(m.sum().item() - 196.0) < 1e-3
    assert torch.allclose(m, torch.ones_like(m))


def test_bbox_area_scales_linearly() -> None:
    """Doubling bbox area must roughly double the mask sum."""
    small = bbox_to_patch_mask([0, 0, 50, 50], 200, 200, grid=14).sum().item()
    large = bbox_to_patch_mask([0, 0, 100, 50], 200, 200, grid=14).sum().item()
    assert abs(large / small - 2.0) < 0.1


# ── Shape / grid ────────────────────────────────────────────────────────────

def test_bbox_grid_7_shape() -> None:
    m = bbox_to_patch_mask([0, 0, 100, 100], 200, 200, grid=7)
    assert m.shape == (7, 7)


def test_bbox_grid_14_shape() -> None:
    m = bbox_to_patch_mask([0, 0, 100, 100], 200, 200, grid=14)
    assert m.shape == (14, 14)


def test_bbox_grid_28_shape() -> None:
    m = bbox_to_patch_mask([0, 0, 100, 100], 200, 200, grid=28)
    assert m.shape == (28, 28)


def test_bbox_grid_16_area_conserved() -> None:
    m = bbox_to_patch_mask([0, 0, 100, 100], 200, 200, grid=16)
    assert abs(m.sum().item() - 0.25 * 16 * 16) < 1e-3


# ── Degenerate / edge cases ─────────────────────────────────────────────────

def test_bbox_none_returns_zeros() -> None:
    m = bbox_to_patch_mask(None, 100, 100, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_empty_w_and_h() -> None:
    m = bbox_to_patch_mask([10, 10, 0, 0], 100, 100, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_zero_width() -> None:
    m = bbox_to_patch_mask([10, 10, 0, 20], 100, 100, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_zero_height() -> None:
    m = bbox_to_patch_mask([10, 10, 20, 0], 100, 100, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_entirely_off_image() -> None:
    m = bbox_to_patch_mask([500, 500, 10, 10], 100, 100, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_zero_image_dimensions() -> None:
    """image_width or image_height = 0 must not divide by zero."""
    m = bbox_to_patch_mask([0, 0, 10, 10], 0, 100, grid=14)
    assert m.sum().item() == 0.0
    m = bbox_to_patch_mask([0, 0, 10, 10], 100, 0, grid=14)
    assert m.sum().item() == 0.0


def test_bbox_negative_coords_clamped() -> None:
    """A bbox starting at negative coords is clamped to 0 — no crash."""
    m = bbox_to_patch_mask([-20, -20, 50, 50], 100, 100, grid=14)
    # Effective bbox [0, 0, 30, 30] → 0.09 of image area → ~17.64
    assert m.sum().item() > 0


def test_bbox_extending_past_image_clamped() -> None:
    """bbox extending past image boundary must be clamped, not expanded."""
    m = bbox_to_patch_mask([50, 50, 200, 200], 100, 100, grid=14)
    # Effective bbox [50, 50, 50, 50] → 25% of image → 49.0
    assert abs(m.sum().item() - 49.0) < 1.0


# ── Spatial correctness ─────────────────────────────────────────────────────

def test_bbox_top_left_quarter_covers_right_region() -> None:
    """Top-left quarter → patches (0..6, 0..6) non-zero, rest zero."""
    m = bbox_to_patch_mask([0, 0, 100, 100], 200, 200, grid=14)
    assert (m[:7, :7] > 0).all()
    # Row 8+ and col 8+ must be all zero
    assert m[8:, :].sum().item() == 0
    assert m[:, 8:].sum().item() == 0


def test_bbox_bottom_right_quarter() -> None:
    """Bottom-right quarter → patches (7..13, 7..13) non-zero."""
    m = bbox_to_patch_mask([100, 100, 100, 100], 200, 200, grid=14)
    assert (m[7:, 7:] > 0).all()
    assert m[:6, :].sum().item() == 0
    assert m[:, :6].sum().item() == 0


def test_bbox_center_region() -> None:
    """Central bbox should illuminate the center patches only."""
    m = bbox_to_patch_mask([60, 60, 80, 80], 200, 200, grid=14)
    assert m[0, 0].item() == 0.0
    assert m[13, 13].item() == 0.0
    center = m[5:9, 5:9].sum().item()
    assert center > 1.0


# ── Patch-boundary alignment ────────────────────────────────────────────────

def test_bbox_aligned_to_patch_grid() -> None:
    """Bbox that fills the first patch exactly → m[0, 0] == 1.0, others ~0."""
    # At 14×14 grid on a 224×224 image, patch 0 spans pixels [0, 16).
    m = bbox_to_patch_mask([0, 0, 16, 16], 224, 224, grid=14)
    assert abs(m[0, 0].item() - 1.0) < 1e-4
    m[0, 0] = 0.0
    assert m.sum().item() < 1e-4


def test_bbox_aligned_covers_two_patches_row() -> None:
    """Bbox spanning exactly two horizontal patches lights up m[0, 0..1]."""
    m = bbox_to_patch_mask([0, 0, 32, 16], 224, 224, grid=14)
    assert abs(m[0, 0].item() - 1.0) < 1e-4
    assert abs(m[0, 1].item() - 1.0) < 1e-4
    m[0, :2] = 0.0
    assert m.sum().item() < 1e-4


# ── Non-square images ──────────────────────────────────────────────────────

def test_bbox_wide_image_area_conserved() -> None:
    """400×100 image, 200×50 bbox → 25% area → sum ≈ 49.0."""
    m = bbox_to_patch_mask([0, 0, 200, 50], 400, 100, grid=14)
    assert abs(m.sum().item() - 49.0) < 1.0


def test_bbox_tall_image_area_conserved() -> None:
    """100×400 image, 50×200 bbox → 25% area → sum ≈ 49.0."""
    m = bbox_to_patch_mask([0, 0, 50, 200], 100, 400, grid=14)
    assert abs(m.sum().item() - 49.0) < 1.0


def test_bbox_tiny_subpixel_still_produces_mask() -> None:
    """Sub-patch bbox → at most 1 patch lights up, sum ≤ 1."""
    m = bbox_to_patch_mask([5, 5, 2, 2], 224, 224, grid=14)
    assert 0 < m.sum().item() <= 1.0


# ── Mask value bounds ──────────────────────────────────────────────────────

def test_bbox_mask_values_in_unit_interval() -> None:
    """Every mask value must be in [0, 1]."""
    m = bbox_to_patch_mask([30, 30, 60, 60], 100, 100, grid=14)
    assert (m >= 0).all()
    assert (m <= 1.0 + 1e-6).all()


def test_bbox_mask_is_float32() -> None:
    m = bbox_to_patch_mask([0, 0, 50, 50], 100, 100, grid=14)
    assert m.dtype == torch.float32
