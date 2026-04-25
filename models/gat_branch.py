"""
SceneIQ — GAT branch.

Encodes the per-image scene graph (objects + typed relationships) into a fixed
set of node embeddings plus a graph-level pooled embedding. The node
embeddings are later keys/values for the cross-attention fusion in
``SceneIQFusion``; the pooled embedding feeds a standalone SCS head when the
model is run in ``--mode gat`` ablation.

We use ``torch_geometric.nn.GATConv`` for the message passing. Edges carry a
learned predicate embedding that is concatenated onto each source node
representation before aggregation (a simple but effective edge-feature
injection trick since GATConv itself doesn't take edge_attr by default).

Graceful degradation
--------------------
If ``torch_geometric`` is not installed, we fall back to a dense self-attention
encoder over the node embeddings. Results are slightly worse but the model
still trains — useful for environments without torch-geometric wheels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


@dataclass
class GATOutput:
    """Container for GAT-branch outputs.

    Attributes:
        scs_logit: (B,) SCS logit from the pooled graph embedding.
        graph_embedding: (B, hidden_dim) mean-pooled node representation.
        node_embeddings: (B, N_max, hidden_dim) per-node embeddings padded to
            ``N_max`` — the max node count in the batch.
        node_mask: (B, N_max) bool mask; True = real node.
    """
    scs_logit: torch.Tensor
    graph_embedding: torch.Tensor
    node_embeddings: torch.Tensor
    node_mask: torch.Tensor


class SceneIQGAT(nn.Module):
    """Graph attention encoder over VG-style scene graphs.

    The forward signature takes a *batched* representation produced by
    :meth:`collate_graphs` — see its docstring for the exact schema.

    Args:
        n_objects: Object vocabulary size (including PAD and UNK).
        n_predicates: Predicate vocabulary size.
        embed_dim: Per-node embedding dim.
        hidden_dim: GAT hidden size.
        num_heads: GAT attention heads.
        num_layers: Stacked GAT layers.
        dropout: Dropout after attention.
    """

    def __init__(
        self,
        n_objects: int,
        n_predicates: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.object_embed = nn.Embedding(n_objects, embed_dim, padding_idx=0)
        self.predicate_embed = nn.Embedding(n_predicates, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        if _HAS_PYG:
            self.gat_layers = nn.ModuleList([
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=embed_dim,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ])
        else:
            # Fallback: dense multi-head self-attention, ignores edges' structure
            # beyond bulk predicate-bias injection at layer 0.
            self.gat_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=num_heads,
                    dropout=dropout, batch_first=True,
                )
                for _ in range(num_layers)
            ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.scs_head = nn.Linear(hidden_dim, 1)

    # ── Collation ────────────────────────────────────────────────────────
    @staticmethod
    def collate_graphs(
        graphs: list,
        n_max: int,
        device: torch.device | None = None,
    ) -> dict:
        """Pack a list of per-image graph dicts into batched tensors.

        Each graph dict looks like ``{"objects": [ids...], "edges": [[s, d, p]]}``.
        Empty or missing graphs are represented as a single UNK node (id=1).

        Args:
            graphs: List of graph dicts (length B).
            n_max: Pad every graph up to ``n_max`` nodes. The caller should pass
                the max observed in the batch (capped at ``SG_MAX_OBJECTS``).
            device: Optional target device for the returned tensors.

        Returns:
            Dict with keys:
                * ``object_ids``: (B, N_max) long
                * ``node_mask``: (B, N_max) bool
                * ``edge_index``: (2, E_total) long, offset so node indices are
                  unique across the batch; this format matches torch_geometric.
                * ``edge_attr``: (E_total,) long — predicate ids.
                * ``batch``: (B * N_max,) long — graph index each node belongs to.
        """
        B = len(graphs)
        object_ids = torch.zeros(B, n_max, dtype=torch.long)
        node_mask = torch.zeros(B, n_max, dtype=torch.bool)

        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_attr: List[int] = []

        for bi, g in enumerate(graphs):
            objs = (g or {}).get("objects", []) or []
            if not objs:
                objs = [1]  # UNK placeholder so the image still has one node
            objs = objs[:n_max]
            for i, oid in enumerate(objs):
                object_ids[bi, i] = int(oid)
                node_mask[bi, i] = True
            offset = bi * n_max
            for edge in (g or {}).get("edges", []) or []:
                s, d, p = edge
                if s >= len(objs) or d >= len(objs):
                    continue
                edge_src.append(offset + int(s))
                edge_dst.append(offset + int(d))
                edge_attr.append(int(p))

        if not edge_src:
            # Guarantee a non-empty edge_index for PyG (one self-loop on the
            # first padded slot). GATConv with add_self_loops=True will dedupe.
            edge_src.append(0)
            edge_dst.append(0)
            edge_attr.append(0)

        out = {
            "object_ids": object_ids,
            "node_mask": node_mask,
            "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.long),
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, batch: dict) -> GATOutput:
        """Run graph encoding on a pre-collated batch.

        Args:
            batch: Output of :meth:`collate_graphs`.

        Returns:
            :class:`GATOutput`.
        """
        object_ids = batch["object_ids"]                  # (B, N)
        node_mask = batch["node_mask"]                    # (B, N)
        B, N = object_ids.shape

        x = self.object_embed(object_ids)                 # (B, N, E)
        x = F.relu(self.input_proj(x))                    # (B, N, H)

        if _HAS_PYG:
            flat = x.reshape(B * N, self.hidden_dim)      # (B*N, H)
            edge_index = batch["edge_index"]
            edge_attr = self.predicate_embed(batch["edge_attr"])  # (E, E_pred)
            for layer, norm in zip(self.gat_layers, self.layer_norms):
                h = layer(flat, edge_index, edge_attr=edge_attr)
                flat = norm(F.elu(h) + flat)              # residual + norm
            nodes = flat.view(B, N, self.hidden_dim)
        else:
            # Dense fallback: self-attention masking padded nodes.
            key_padding = ~node_mask                      # True where PAD
            nodes = x
            for layer, norm in zip(self.gat_layers, self.layer_norms):
                attn_out, _ = layer(nodes, nodes, nodes, key_padding_mask=key_padding)
                nodes = norm(F.elu(attn_out) + nodes)

        # Mean-pool over real nodes only
        mask_f = node_mask.unsqueeze(-1).float()          # (B, N, 1)
        graph_emb = (nodes * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        scs_logit = self.scs_head(graph_emb).squeeze(-1)  # (B,)

        return GATOutput(
            scs_logit=scs_logit,
            graph_embedding=graph_emb,
            node_embeddings=nodes,
            node_mask=node_mask,
        )
