"""
SceneIQ — Phase 2 model architectures.

Defines the scene-graph GAT encoder, GAT-only classifier,
cross-attention fusion module, and the full ViT+GAT fusion model
used for the ablation study.

Classes:
    SceneGraphGAT:       GAT encoder over scene graphs → graph embedding.
    GATClassifier:       GAT + linear head (GAT-only ablation).
    CrossAttentionFusion: ViT patches attend to GAT node embeddings.
    FusionModel:         Full ViT + GAT + cross-attention + localization.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import ViTModel


class SceneGraphGAT(nn.Module):
    """GAT encoder over a scene graph.

    Embeds node labels, applies stacked GATConv layers, and produces
    a fixed-size graph-level embedding via global mean pooling.

    Args:
        num_categories: Size of the object name vocabulary.
        num_predicates: Size of the predicate vocabulary (unused in this
            version; edges are untyped in GATConv).
        embed_dim: Dimensionality of the node embedding.
        hidden_dim: Hidden dimension of GATConv layers.
        num_heads: Number of attention heads in each GATConv.
        num_layers: Number of stacked GATConv layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_categories: int,
        num_predicates: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(num_categories, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.out_dim = hidden_dim

    def forward(
        self,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batched scene graph.

        Args:
            node_labels: (N,) integer node labels.
            edge_index: (2, E) edge index in COO format.
            batch: (N,) batch assignment vector.

        Returns:
            (B, hidden_dim) graph-level embedding.
        """
        x = self.node_embed(node_labels)
        x = self.dropout(x)

        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return global_mean_pool(x, batch)

    def forward_nodes(
        self,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-node embeddings instead of pooled graph embedding.

        Args:
            node_labels: (N,) integer node labels.
            edge_index: (2, E) edge index in COO format.
            batch: (N,) batch assignment vector.

        Returns:
            (N, hidden_dim) per-node embeddings.
        """
        x = self.node_embed(node_labels)
        x = self.dropout(x)

        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        return x


class GATClassifier(nn.Module):
    """GAT encoder with a linear classification head.

    Used for the GAT-only ablation (no image features).

    Args:
        num_categories: Object name vocabulary size.
        num_predicates: Predicate vocabulary size.
        hidden_dim: GAT hidden dimension.
        num_heads: GAT attention heads.
        num_layers: Number of GAT layers.
        dropout: Dropout probability.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        num_categories: int,
        num_predicates: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.gat = SceneGraphGAT(
            num_categories, num_predicates, embed_dim=hidden_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Classify based on scene graph alone.

        Args:
            node_labels: (N,) integer node labels.
            edge_index: (2, E) edge index in COO format.
            batch: (N,) batch assignment vector.

        Returns:
            (B, num_classes) logits.
        """
        graph_emb = self.gat(node_labels, edge_index, batch)
        return self.classifier(graph_emb)


class CrossAttentionFusion(nn.Module):
    """Cross-attention: ViT patch tokens attend to GAT node embeddings.

    Projects both modalities to a shared dimension, then applies
    multi-head attention where ViT patches are queries and GAT nodes
    are keys/values.

    Args:
        vit_dim: ViT hidden dimension (768 for vit-base).
        gat_dim: GAT hidden dimension.
        fusion_dim: Shared projection dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vit_dim: int = 768,
        gat_dim: int = 128,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.gat_proj = nn.Linear(gat_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.fusion_dim = fusion_dim

    def forward(
        self,
        vit_tokens: torch.Tensor,
        gat_nodes: torch.Tensor,
        gat_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse ViT patch tokens with GAT node embeddings.

        Args:
            vit_tokens: (B, 196, vit_dim) ViT patch tokens (no CLS).
            gat_nodes: (B, max_nodes, gat_dim) padded GAT node embeddings.
            gat_mask: (B, max_nodes) True for padded positions.

        Returns:
            (B, 196, fusion_dim) fused patch representations.
        """
        q = self.vit_proj(vit_tokens)
        kv = self.gat_proj(gat_nodes)
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=gat_mask)
        return self.norm(q + attn_out)


class FusionModel(nn.Module):
    """Full ViT + GAT fusion model with localization head.

    Combines a frozen (or fine-tuned) ViT backbone with a GAT scene-graph
    encoder via cross-attention. Produces classification logits and
    per-patch inconsistency scores for localization.

    Args:
        vit_model_name: HuggingFace ViT model ID.
        num_categories: Object name vocabulary size.
        num_predicates: Predicate vocabulary size.
        gat_hidden: GAT hidden dimension.
        gat_heads: GAT attention heads.
        gat_layers: Number of GAT layers.
        fusion_dim: Cross-attention fusion dimension.
        fusion_heads: Cross-attention heads.
        dropout: Dropout probability.
        freeze_vit: If True, freeze ViT backbone weights.
    """

    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        num_categories: int = 17110,
        num_predicates: int = 5501,
        gat_hidden: int = 128,
        gat_heads: int = 4,
        gat_layers: int = 2,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        dropout: float = 0.3,
        freeze_vit: bool = False,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.gat = SceneGraphGAT(
            num_categories, num_predicates, embed_dim=gat_hidden,
            hidden_dim=gat_hidden, num_heads=gat_heads,
            num_layers=gat_layers, dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            vit_dim=self.vit.config.hidden_size,
            gat_dim=gat_hidden,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )

        self.patch_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        node_counts: list[int],
    ) -> dict:
        """Forward pass returning logits and patch-level scores.

        Args:
            pixel_values: (B, 3, 224, 224) input images.
            node_labels: (N_total,) node label indices.
            edge_index: (2, E_total) batched edge index.
            graph_batch: (N_total,) batch assignment for nodes.
            node_counts: List of node counts per sample in the batch.

        Returns:
            Dict with ``logits`` (B, 2) and ``patch_scores`` (B, 196).
        """
        # ViT: extract patch tokens (skip CLS at position 0)
        vit_out = self.vit(pixel_values=pixel_values)
        patch_tokens = vit_out.last_hidden_state[:, 1:, :]  # (B, 196, 768)

        # GAT: per-node embeddings
        node_embs = self.gat.forward_nodes(node_labels, edge_index, graph_batch)

        # Pad node embeddings to (B, max_nodes, gat_hidden) for cross-attention
        B = pixel_values.size(0)
        max_nodes = max(node_counts)
        gat_dim = node_embs.size(-1)
        device = pixel_values.device

        padded_nodes = torch.zeros(B, max_nodes, gat_dim, device=device)
        gat_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)

        offset = 0
        for i, nc in enumerate(node_counts):
            padded_nodes[i, :nc] = node_embs[offset:offset + nc]
            gat_mask[i, :nc] = False
            offset += nc

        # Cross-attention fusion
        fused = self.fusion(patch_tokens, padded_nodes, gat_mask)  # (B, 196, fusion_dim)

        # Classification: mean-pool fused patch representations
        cls_emb = fused.mean(dim=1)  # (B, fusion_dim)
        logits = self.classifier(cls_emb)  # (B, 2)

        # Localization: per-patch inconsistency score
        patch_scores = self.patch_head(fused).squeeze(-1)  # (B, 196)

        return {"logits": logits, "patch_scores": patch_scores}
