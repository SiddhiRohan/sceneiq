"""
Tests for ``models/gat_branch.py``.

Covers ``SceneIQGAT.collate_graphs`` batching logic and the forward pass:
  * mask, padding, and offset correctness
  * empty / missing / single-graph batches
  * edge-index rebasing
  * forward shape invariants
  * gradient flow through the object embedding
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import torch

from models.gat_branch import SceneIQGAT


def _gat(hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 1):
    return SceneIQGAT(
        n_objects=50, n_predicates=20, embed_dim=16,
        hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=num_layers, dropout=0.0,
    )


# ── Collation ───────────────────────────────────────────────────────────────

def test_gat_collate_batch_dim_matches_input() -> None:
    gat = _gat()
    graphs = [{"objects": [2, 3], "edges": []}, {"objects": [5], "edges": []}]
    batch = gat.collate_graphs(graphs, n_max=3)
    assert batch["object_ids"].shape == (2, 3)
    assert batch["node_mask"].shape == (2, 3)


def test_gat_collate_object_ids_are_long() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=3)
    assert batch["object_ids"].dtype == torch.long


def test_gat_collate_node_mask_is_bool() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=3)
    assert batch["node_mask"].dtype == torch.bool


def test_gat_collate_pads_with_zero() -> None:
    """Unused node slots must be 0 (PAD id)."""
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [7], "edges": []}], n_max=4)
    assert batch["object_ids"][0].tolist() == [7, 0, 0, 0]


def test_gat_collate_node_mask_matches_real_nodes() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [2, 3, 4], "edges": []}], n_max=5)
    mask = batch["node_mask"][0].tolist()
    assert mask == [True, True, True, False, False]


def test_gat_collate_missing_graph_becomes_single_unk() -> None:
    """None graph → 1 UNK node (id=1), rest padded."""
    gat = _gat()
    batch = gat.collate_graphs([None], n_max=3)
    assert batch["object_ids"][0].tolist() == [1, 0, 0]
    assert batch["node_mask"][0].tolist() == [True, False, False]


def test_gat_collate_empty_objects_list_becomes_unk() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [], "edges": []}], n_max=2)
    assert batch["object_ids"][0].tolist() == [1, 0]


def test_gat_collate_truncates_to_n_max() -> None:
    """A graph with more than n_max nodes must be truncated to n_max."""
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2, 3, 4, 5], "edges": []}], n_max=3)
    assert batch["object_ids"].shape[1] == 3
    assert batch["object_ids"][0].tolist() == [1, 2, 3]


def test_gat_collate_edge_index_offsets_across_batch() -> None:
    """Edge indices for batch item i must be offset by i * n_max."""
    gat = _gat()
    graphs = [
        {"objects": [1, 2], "edges": [[0, 1, 3]]},
        {"objects": [4, 5], "edges": [[0, 1, 3]]},
    ]
    batch = gat.collate_graphs(graphs, n_max=2)
    # Edges from second graph should point at indices 2, 3 — not 0, 1.
    src, dst = batch["edge_index"].tolist()
    assert 2 in src or 2 in dst or 3 in src or 3 in dst


def test_gat_collate_edges_beyond_n_max_dropped() -> None:
    """Edges whose endpoints exceed the truncated node set are dropped."""
    gat = _gat()
    graphs = [{"objects": [1, 2, 3, 4], "edges": [[3, 0, 2]]}]  # endpoint 3 won't fit in n_max=2
    batch = gat.collate_graphs(graphs, n_max=2)
    # Either edge survived (but only if both endpoints < 2) or edge_index is a self-loop-only fallback.
    edges = batch["edge_index"]
    # The explicit edge [3, 0] must NOT appear
    for s, d in zip(edges[0].tolist(), edges[1].tolist()):
        assert s < 2 and d < 2


def test_gat_collate_empty_edges_gives_self_loop_fallback() -> None:
    """PyG needs non-empty edge_index; we inject a self-loop fallback."""
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1], "edges": []}], n_max=2)
    assert batch["edge_index"].shape[0] == 2
    assert batch["edge_index"].shape[1] >= 1


def test_gat_collate_edge_attr_matches_edge_count() -> None:
    gat = _gat()
    graphs = [
        {"objects": [1, 2], "edges": [[0, 1, 3], [1, 0, 4]]},
    ]
    batch = gat.collate_graphs(graphs, n_max=2)
    assert batch["edge_attr"].shape[0] == batch["edge_index"].shape[1]


# ── Forward ─────────────────────────────────────────────────────────────────

def test_gat_forward_scs_shape_batch_1() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=2)
    out = gat(batch)
    assert out.scs_logit.shape == (1,)


def test_gat_forward_scs_shape_batch_many() -> None:
    gat = _gat()
    graphs = [{"objects": [1, 2], "edges": []}] * 8
    batch = gat.collate_graphs(graphs, n_max=2)
    out = gat(batch)
    assert out.scs_logit.shape == (8,)


def test_gat_forward_graph_embedding_shape() -> None:
    gat = _gat(hidden_dim=32)
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=2)
    out = gat(batch)
    assert out.graph_embedding.shape == (1, 32)


def test_gat_forward_node_embeddings_shape() -> None:
    gat = _gat(hidden_dim=32)
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}] * 2, n_max=3)
    out = gat(batch)
    assert out.node_embeddings.shape == (2, 3, 32)


def test_gat_forward_node_mask_passthrough() -> None:
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=4)
    out = gat(batch)
    assert torch.equal(out.node_mask, batch["node_mask"])


def test_gat_forward_output_deterministic_with_seed() -> None:
    torch.manual_seed(42)
    gat1 = _gat()
    torch.manual_seed(42)
    gat2 = _gat()
    batch = gat1.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=2)
    out1 = gat1(batch)
    out2 = gat2(batch)
    assert torch.allclose(out1.scs_logit, out2.scs_logit, atol=1e-5)


def test_gat_forward_gradient_flows() -> None:
    """Backward from scs_logit must reach the object embedding parameters."""
    gat = _gat()
    batch = gat.collate_graphs([{"objects": [1, 2], "edges": [[0, 1, 3]]}], n_max=2)
    out = gat(batch)
    out.scs_logit.sum().backward()
    assert gat.object_embed.weight.grad is not None
    assert gat.object_embed.weight.grad.abs().sum().item() > 0


def test_gat_output_changes_with_different_inputs() -> None:
    """Different graphs must produce different graph embeddings."""
    gat = _gat()
    a = gat.collate_graphs([{"objects": [1, 2], "edges": []}], n_max=2)
    b = gat.collate_graphs([{"objects": [5, 6], "edges": []}], n_max=2)
    ea = gat(a).graph_embedding
    eb = gat(b).graph_embedding
    assert not torch.allclose(ea, eb)


def test_gat_parameter_count_scales_with_hidden() -> None:
    small = sum(p.numel() for p in _gat(hidden_dim=32).parameters())
    large = sum(p.numel() for p in _gat(hidden_dim=64).parameters())
    assert large > small
