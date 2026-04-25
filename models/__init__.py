"""
SceneIQ — model package.

Exposes the three architecture variants used in the ablation and a single
factory ``build_model(mode, ...)`` so training/eval code never instantiates
a backbone directly.
"""

from .vit_branch import SceneIQViT
from .gat_branch import SceneIQGAT
from .fusion import SceneIQFusion, build_model, SceneIQOutput

__all__ = [
    "SceneIQViT",
    "SceneIQGAT",
    "SceneIQFusion",
    "SceneIQOutput",
    "build_model",
]
