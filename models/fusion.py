"""
SceneIQ — Cross-attention fusion + unified model wrapper.

Combines the ViT and GAT branches via a single multi-head cross-attention block
(ViT patches as query, GAT node embeddings as key/value). Produces a unified
SCS logit and the per-patch localization heatmap.

The :func:`build_model` factory returns the right architecture for each
ablation mode: ``vit``, ``gat``, or ``fusion``. Every variant exposes the same
output interface (:class:`SceneIQOutput`) so training/eval code is
mode-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_branch import SceneIQViT
from .gat_branch import SceneIQGAT


@dataclass
class SceneIQOutput:
    """Unified output for all three model modes.

    Attributes:
        scs_logit: (B,) pre-sigmoid coherence logit. Apply ``torch.sigmoid``
            for SCS in [0, 1].
        heatmap_logits: (B, H, W) per-patch inconsistency logits. For the GAT
            mode this will be a zero tensor of the correct shape (localization
            is not meaningful without pixel features).
        has_heatmap: True when ``heatmap_logits`` is a trained output.
    """
    scs_logit: torch.Tensor
    heatmap_logits: torch.Tensor
    has_heatmap: bool


# ────────────────────────────────────────────────────────────────────────────
# Fusion
# ────────────────────────────────────────────────────────────────────────────

class SceneIQFusion(nn.Module):
    """ViT + GAT + cross-attention fusion producing SCS and localization.

    Args:
        vit: A :class:`SceneIQViT` instance (owned).
        gat: A :class:`SceneIQGAT` instance (owned).
        num_heads: Attention heads in the fusion block.
        dropout: Dropout inside the fusion block.
    """

    def __init__(
        self,
        vit: SceneIQViT,
        gat: SceneIQGAT,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vit = vit
        self.gat = gat

        d = vit.hidden_dim

        # Project GAT node embeddings to ViT hidden dim for cross-attention
        self.graph_proj = nn.Linear(gat.hidden_dim, d)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d)

        # Heads operate on the fused CLS / fused patches
        self.scs_head = nn.Linear(d + gat.hidden_dim, 1)
        self.loc_head = nn.Linear(d, 1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        graph_batch: dict,
    ) -> SceneIQOutput:
        """Run fusion forward pass.

        Args:
            pixel_values: (B, 3, 224, 224) image tensor.
            graph_batch: Output of :meth:`SceneIQGAT.collate_graphs`.

        Returns:
            :class:`SceneIQOutput` with SCS logit + patch heatmap.
        """
        vit_out = self.vit(pixel_values)            # ViTOutput
        gat_out = self.gat(graph_batch)             # GATOutput

        B = pixel_values.size(0)
        patches = vit_out.patch_embeddings          # (B, P, D)
        nodes = self.graph_proj(gat_out.node_embeddings)   # (B, N, D)

        # Cross-attention: patches attend to scene-graph nodes
        key_padding = ~gat_out.node_mask            # True on PAD
        fused_patches, _ = self.cross_attn(
            query=patches, key=nodes, value=nodes,
            key_padding_mask=key_padding,
        )
        fused_patches = self.cross_norm(fused_patches + patches)   # residual

        # SCS from [fused CLS-proxy (mean-pool over fused patches)  ⊕  graph_emb]
        pooled_patches = fused_patches.mean(dim=1)                 # (B, D)
        scs_in = torch.cat([pooled_patches, gat_out.graph_embedding], dim=-1)
        scs_logit = self.scs_head(scs_in).squeeze(-1)              # (B,)

        # Localization on fused patches
        loc_logits_flat = self.loc_head(fused_patches).squeeze(-1)  # (B, P)
        grid = self.vit.patch_grid
        heatmap = loc_logits_flat.view(B, grid, grid)

        return SceneIQOutput(
            scs_logit=scs_logit,
            heatmap_logits=heatmap,
            has_heatmap=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# Mode-specific wrappers that share the :class:`SceneIQOutput` interface
# ────────────────────────────────────────────────────────────────────────────

class _ViTOnly(nn.Module):
    """Thin wrapper that adapts :class:`SceneIQViT` to :class:`SceneIQOutput`."""

    def __init__(self, vit: SceneIQViT):
        super().__init__()
        self.vit = vit

    def forward(self, pixel_values, graph_batch=None) -> SceneIQOutput:  # noqa: ARG002
        out = self.vit(pixel_values)
        return SceneIQOutput(
            scs_logit=out.scs_logit,
            heatmap_logits=out.heatmap_logits,
            has_heatmap=True,
        )


class _GATOnly(nn.Module):
    """Thin wrapper that adapts :class:`SceneIQGAT` to :class:`SceneIQOutput`.

    Localization is not meaningful in this mode, so we emit a zero heatmap of
    the ViT grid shape to keep downstream code uniform, and flag
    ``has_heatmap=False``.
    """

    def __init__(self, gat: SceneIQGAT, patch_grid: int = 14):
        super().__init__()
        self.gat = gat
        self.patch_grid = patch_grid

    def forward(self, pixel_values, graph_batch) -> SceneIQOutput:
        out = self.gat(graph_batch)
        B = out.scs_logit.size(0)
        zeros = torch.zeros(B, self.patch_grid, self.patch_grid, device=out.scs_logit.device)
        return SceneIQOutput(
            scs_logit=out.scs_logit,
            heatmap_logits=zeros,
            has_heatmap=False,
        )


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

def build_model(
    mode: str,
    *,
    vit_model_name: str,
    n_objects: int,
    n_predicates: int,
    patch_grid: int = 14,
    sg_embed_dim: int = 128,
    gat_hidden_dim: int = 256,
    gat_num_heads: int = 4,
    gat_num_layers: int = 2,
    fusion_num_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """Return the model for a given ablation mode.

    Args:
        mode: One of ``"vit"``, ``"gat"``, ``"fusion"``.
        vit_model_name: HF model id for the ViT backbone.
        n_objects: Object vocab size (PAD+UNK+labels).
        n_predicates: Predicate vocab size.
        patch_grid: ViT patch grid side (14 for patch16/224).
        sg_embed_dim: Per-node embedding dim in the GAT.
        gat_hidden_dim: GAT hidden size.
        gat_num_heads: GAT attention heads.
        gat_num_layers: GAT layers.
        fusion_num_heads: Cross-attention heads (fusion only).
        dropout: Dropout in all heads.

    Returns:
        An ``nn.Module`` whose ``forward(pixel_values, graph_batch=None)``
        returns a :class:`SceneIQOutput`.
    """
    mode = mode.lower()
    if mode not in {"vit", "gat", "fusion"}:
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of vit/gat/fusion.")

    if mode == "vit":
        vit = SceneIQViT(model_name=vit_model_name, patch_grid=patch_grid, dropout=dropout)
        return _ViTOnly(vit)

    gat = SceneIQGAT(
        n_objects=n_objects,
        n_predicates=n_predicates,
        embed_dim=sg_embed_dim,
        hidden_dim=gat_hidden_dim,
        num_heads=gat_num_heads,
        num_layers=gat_num_layers,
        dropout=dropout,
    )
    if mode == "gat":
        return _GATOnly(gat, patch_grid=patch_grid)

    vit = SceneIQViT(model_name=vit_model_name, patch_grid=patch_grid, dropout=dropout)
    return SceneIQFusion(vit=vit, gat=gat, num_heads=fusion_num_heads, dropout=dropout)


# ────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ────────────────────────────────────────────────────────────────────────────

def soft_iou_loss(
    heatmap_logits: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft-IoU loss on the patch heatmap.

    Args:
        heatmap_logits: (B, H, W) pre-sigmoid heatmap.
        target_mask: (B, H, W) in [0, 1] — the fraction of each patch covered
            by the ground-truth bbox.
        eps: Numerical stabilizer.

    Returns:
        Scalar loss: mean over the batch of ``1 - soft_iou``.
    """
    probs = torch.sigmoid(heatmap_logits)
    probs = probs.view(probs.size(0), -1)
    target = target_mask.view(target_mask.size(0), -1)

    inter = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def compute_loss(
    scs_logit: torch.Tensor,
    heatmap_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor | None,
    has_heatmap: bool,
    bce_weight: float = 1.0,
    loc_weight: float = 1.0,
) -> dict:
    """Combined BCE (SCS) + soft-IoU (localization) loss.

    Args:
        scs_logit: (B,) pre-sigmoid coherence logit.
        heatmap_logits: (B, H, W) pre-sigmoid heatmap.
        labels: (B,) binary labels (0 = coherent, 1 = incoherent).
        target_mask: (B, H, W) soft patch-coverage mask (may contain rows of
            zeros for coherent samples — localization loss ignores those).
        has_heatmap: If False, skip the localization term.
        bce_weight: Weight on the BCE term.
        loc_weight: Weight on the soft-IoU term.

    Returns:
        Dict with ``total``, ``bce``, ``loc`` tensors (scalars).
    """
    bce = F.binary_cross_entropy_with_logits(scs_logit, labels.float())
    out = {"bce": bce, "loc": torch.zeros_like(bce)}

    if has_heatmap and target_mask is not None:
        # Only compute localization loss on incoherent samples (those with a real bbox)
        mask_sum = target_mask.view(target_mask.size(0), -1).sum(dim=1)
        has_target = mask_sum > 0
        if has_target.any():
            out["loc"] = soft_iou_loss(
                heatmap_logits[has_target], target_mask[has_target]
            )

    out["total"] = bce_weight * out["bce"] + loc_weight * out["loc"]
    return out
