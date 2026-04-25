"""
Tests for the ViT branch, fusion module, and build_model factory.

Uses the FakeBackbone from conftest.py so nothing hits HuggingFace.
Verifies shape invariants, gradient flow, factory behaviour, and
interface uniformity across the three modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import torch
import torch.nn as nn

from conftest import install_fake_vit, make_small_gat


def _setup():
    install_fake_vit()


# ── ViT branch ───────────────────────────────────────────────────────────────

def test_vit_scs_logit_is_1d() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(3, 3, 224, 224))
    assert out.scs_logit.dim() == 1
    assert out.scs_logit.shape[0] == 3


def test_vit_heatmap_shape_matches_patch_grid() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT(patch_grid=14)
    out = vit(torch.randn(2, 3, 224, 224))
    assert out.heatmap_logits.shape == (2, 14, 14)


def test_vit_cls_embedding_hidden_dim() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(2, 3, 224, 224))
    assert out.cls_embedding.shape == (2, vit.hidden_dim)


def test_vit_patch_embeddings_count() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT(patch_grid=14)
    out = vit(torch.randn(1, 3, 224, 224))
    assert out.patch_embeddings.shape == (1, 14 * 14, vit.hidden_dim)


def test_vit_batch_size_1_works() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(1, 3, 224, 224))
    assert out.scs_logit.shape == (1,)
    assert out.heatmap_logits.shape == (1, 14, 14)


def test_vit_larger_batch_works() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(8, 3, 224, 224))
    assert out.scs_logit.shape == (8,)
    assert out.heatmap_logits.shape == (8, 14, 14)


def test_vit_gradient_flows_through_scs_head() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(2, 3, 224, 224))
    out.scs_logit.sum().backward()
    assert vit.scs_head.weight.grad is not None
    assert vit.scs_head.weight.grad.abs().sum().item() > 0


def test_vit_gradient_flows_through_loc_head() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    vit = SceneIQViT()
    out = vit(torch.randn(2, 3, 224, 224))
    out.heatmap_logits.sum().backward()
    assert vit.loc_head.weight.grad is not None
    assert vit.loc_head.weight.grad.abs().sum().item() > 0


# ── _ViTOnly wrapper ─────────────────────────────────────────────────────────

def test_vitonly_has_heatmap_flag_true() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import _ViTOnly
    model = _ViTOnly(SceneIQViT())
    out = model(torch.randn(2, 3, 224, 224), None)
    assert out.has_heatmap is True


def test_vitonly_accepts_none_graph_batch() -> None:
    """_ViTOnly must not crash when graph_batch is None."""
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import _ViTOnly
    model = _ViTOnly(SceneIQViT())
    out = model(torch.randn(1, 3, 224, 224), None)
    assert out.scs_logit.shape == (1,)


# ── _GATOnly wrapper ────────────────────────────────────────────────────────

def test_gatonly_has_heatmap_flag_false() -> None:
    from models.fusion import _GATOnly
    gat = make_small_gat()
    model = _GATOnly(gat, patch_grid=14)
    batch = gat.collate_graphs([{"objects": [2], "edges": []}], n_max=2)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.has_heatmap is False


def test_gatonly_zero_heatmap() -> None:
    """GAT-only mode emits an all-zero heatmap of the correct shape."""
    from models.fusion import _GATOnly
    gat = make_small_gat()
    model = _GATOnly(gat, patch_grid=14)
    batch = gat.collate_graphs([{"objects": [2], "edges": []}], n_max=2)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.heatmap_logits.shape == (1, 14, 14)
    assert out.heatmap_logits.abs().sum().item() == 0.0


def test_gatonly_ignores_pixel_values_for_scs() -> None:
    """Two different images with the same graph must give the same SCS in GAT mode."""
    from models.fusion import _GATOnly
    gat = make_small_gat()
    gat.eval()
    model = _GATOnly(gat, patch_grid=14)
    batch = gat.collate_graphs([{"objects": [2, 3], "edges": []}], n_max=3)
    with torch.no_grad():
        out1 = model(torch.randn(1, 3, 224, 224), batch)
        out2 = model(torch.randn(1, 3, 224, 224), batch)
    assert torch.allclose(out1.scs_logit, out2.scs_logit)


# ── Fusion ───────────────────────────────────────────────────────────────────

def test_fusion_has_heatmap_true() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    model = SceneIQFusion(SceneIQViT(), make_small_gat())
    batch = model.gat.collate_graphs([{"objects": [2], "edges": []}], n_max=2)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.has_heatmap is True


def test_fusion_scs_shape_matches_batch() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    model = SceneIQFusion(SceneIQViT(), make_small_gat())
    batch = model.gat.collate_graphs(
        [{"objects": [2], "edges": []}, {"objects": [3], "edges": []}, {"objects": [4], "edges": []}],
        n_max=2,
    )
    out = model(torch.randn(3, 3, 224, 224), batch)
    assert out.scs_logit.shape == (3,)


def test_fusion_heatmap_shape_matches_patch_grid() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    model = SceneIQFusion(SceneIQViT(patch_grid=14), make_small_gat())
    batch = model.gat.collate_graphs([{"objects": [2], "edges": []}], n_max=2)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.heatmap_logits.shape == (1, 14, 14)


def test_fusion_backward_reaches_vit_and_gat() -> None:
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    vit = SceneIQViT()
    gat = make_small_gat()
    model = SceneIQFusion(vit, gat)
    batch = model.gat.collate_graphs([{"objects": [2], "edges": [[0, 0, 3]]}], n_max=2)
    out = model(torch.randn(1, 3, 224, 224), batch)
    (out.scs_logit.sum() + out.heatmap_logits.sum()).backward()
    n_vit = sum(1 for p in vit.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_gat = sum(1 for p in gat.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_vit > 0, "Fusion backward did not reach ViT parameters"
    assert n_gat > 0, "Fusion backward did not reach GAT parameters"


def test_fusion_single_node_graph_does_not_crash() -> None:
    """A graph with exactly one node must still produce valid output."""
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    model = SceneIQFusion(SceneIQViT(), make_small_gat())
    batch = model.gat.collate_graphs([{"objects": [2], "edges": []}], n_max=1)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.scs_logit.shape == (1,)


def test_fusion_large_graph() -> None:
    """Fusion must handle graphs near SG_MAX_OBJECTS."""
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    model = SceneIQFusion(SceneIQViT(), make_small_gat())
    graph = {"objects": list(range(2, 20)), "edges": []}
    batch = model.gat.collate_graphs([graph], n_max=18)
    out = model(torch.randn(1, 3, 224, 224), batch)
    assert out.scs_logit.shape == (1,)


def test_fusion_graph_proj_matches_vit_hidden() -> None:
    """The fusion's graph_proj output dimension must equal the ViT hidden size."""
    _setup()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion
    vit = SceneIQViT()
    gat = make_small_gat(hidden_dim=32)
    model = SceneIQFusion(vit, gat)
    assert model.graph_proj.out_features == vit.hidden_dim


# ── build_model factory ──────────────────────────────────────────────────────

def test_build_model_vit_mode_returns_vit_only() -> None:
    _setup()
    from models.fusion import _ViTOnly, build_model
    m = build_model(mode="vit", vit_model_name="mock",
                    n_objects=50, n_predicates=20)
    assert isinstance(m, _ViTOnly)


def test_build_model_gat_mode_returns_gat_only() -> None:
    from models.fusion import _GATOnly, build_model
    m = build_model(mode="gat", vit_model_name="mock",
                    n_objects=50, n_predicates=20,
                    gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1)
    assert isinstance(m, _GATOnly)


def test_build_model_fusion_mode_returns_fusion() -> None:
    _setup()
    from models.fusion import SceneIQFusion, build_model
    m = build_model(mode="fusion", vit_model_name="mock",
                    n_objects=50, n_predicates=20,
                    gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1)
    assert isinstance(m, SceneIQFusion)


def test_build_model_unknown_mode_raises() -> None:
    from models.fusion import build_model
    try:
        build_model(mode="quantum", vit_model_name="mock",
                    n_objects=50, n_predicates=20)
    except ValueError:
        return
    raise AssertionError("build_model should raise ValueError for an unknown mode")


def test_build_model_case_insensitive_mode() -> None:
    """Mode string should be matched case-insensitively."""
    _setup()
    from models.fusion import _ViTOnly, build_model
    m = build_model(mode="ViT", vit_model_name="mock",
                    n_objects=50, n_predicates=20)
    assert isinstance(m, _ViTOnly)


def test_build_model_fusion_has_both_branches() -> None:
    _setup()
    from models.fusion import build_model
    m = build_model(mode="fusion", vit_model_name="mock",
                    n_objects=50, n_predicates=20,
                    gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1)
    assert hasattr(m, "vit")
    assert hasattr(m, "gat")
    assert hasattr(m, "cross_attn")


def test_build_model_gat_only_has_gat_attribute() -> None:
    from models.fusion import build_model
    m = build_model(mode="gat", vit_model_name="mock",
                    n_objects=50, n_predicates=20,
                    gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1)
    assert hasattr(m, "gat")


def test_build_model_returns_nn_module() -> None:
    _setup()
    from models.fusion import build_model
    for mode in ("vit", "gat", "fusion"):
        m = build_model(mode=mode, vit_model_name="mock",
                        n_objects=50, n_predicates=20,
                        gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1)
        assert isinstance(m, nn.Module)
