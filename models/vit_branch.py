"""
SceneIQ — ViT branch.

Wraps HuggingFace's ``google/vit-base-patch16-224`` and replaces the default
classification head with two task-specific heads:

  * **SCS head** — a sigmoid scalar giving the Scene Coherence Score from the
    CLS token embedding (Eq. 1 in the proposal).
  * **Localization head** — a linear layer on every patch token producing a
    per-patch inconsistency logit, reshaped into a 14×14 heatmap.

This module is used both standalone (``--mode vit`` ablation) and as the
visual branch inside ``SceneIQFusion``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import ViTModel


@dataclass
class ViTOutput:
    """Container for ViT-branch outputs.

    Attributes:
        scs_logit: (B,) raw logit for the SCS (pre-sigmoid).
        heatmap_logits: (B, H, W) per-patch inconsistency logits.
        cls_embedding: (B, D) CLS-token embedding for fusion consumers.
        patch_embeddings: (B, H*W, D) patch embeddings for cross-attention.
    """
    scs_logit: torch.Tensor
    heatmap_logits: torch.Tensor
    cls_embedding: torch.Tensor
    patch_embeddings: torch.Tensor


class SceneIQViT(nn.Module):
    """ViT feature extractor + SCS head + localization heatmap head.

    Args:
        model_name: HuggingFace model id (default ``google/vit-base-patch16-224``).
        patch_grid: Patch grid side length (14 for 224px at patch16).
        dropout: Dropout applied before the SCS projection.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        patch_grid: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_grid = patch_grid

        self.dropout = nn.Dropout(dropout)
        # SCS head: CLS -> scalar logit
        self.scs_head = nn.Linear(self.hidden_dim, 1)
        # Localization head: per-patch scalar logit
        self.loc_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, pixel_values: torch.Tensor) -> ViTOutput:
        """Run a batch of images through ViT and the two heads.

        Args:
            pixel_values: (B, 3, 224, 224) image tensor normalized by the HF
                processor.

        Returns:
            A :class:`ViTOutput` with logits and embeddings.
        """
        out = self.backbone(pixel_values=pixel_values)
        hidden = out.last_hidden_state             # (B, 1 + P, D)  P = patch_grid**2
        cls = hidden[:, 0, :]
        patches = hidden[:, 1:, :]                 # (B, P, D)

        scs_logit = self.scs_head(self.dropout(cls)).squeeze(-1)        # (B,)
        loc_logits_flat = self.loc_head(patches).squeeze(-1)            # (B, P)
        heatmap = loc_logits_flat.view(-1, self.patch_grid, self.patch_grid)

        return ViTOutput(
            scs_logit=scs_logit,
            heatmap_logits=heatmap,
            cls_embedding=cls,
            patch_embeddings=patches,
        )
