"""
Shared test helpers and fixtures.

Installs a fake ViT backbone once per process so every test module can build
``SceneIQViT`` / ``SceneIQFusion`` without downloading HuggingFace weights.
Also provides a ``_FakeProcessor`` that mimics ``ViTImageProcessor`` just well
enough for Dataset / collate tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── sys.path so every test file can ``import models``, ``import data`` etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ── Fake ViT backbone ────────────────────────────────────────────────────────

class FakeBackbone(nn.Module):
    """Stand-in for HuggingFace ViTModel.

    Produces a (B, 1+P, D) hidden state that is a function of ``pixel_values``
    (via a 16×16 conv stem), so autograd can flow back to every parameter.
    """

    def __init__(self, hidden_dim: int = 768, patch_grid: int = 14):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden_dim})()
        self.hidden_dim = hidden_dim
        self.patch_grid = patch_grid
        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, pixel_values):
        feats = self.stem(pixel_values)
        B, D, H, W = feats.shape
        patches = feats.flatten(2).transpose(1, 2)  # (B, P, D)
        cls = self.cls.expand(B, 1, -1)
        hidden = torch.cat([cls, patches], dim=1)
        return type("O", (), {"last_hidden_state": hidden})()


_FAKE_INSTALLED = False


def install_fake_vit(patch_grid: int = 14, hidden_dim: int = 768) -> None:
    """Monkey-patch ``SceneIQViT.__init__`` so tests work offline.

    Idempotent: the second call is a no-op.
    """
    global _FAKE_INSTALLED
    if _FAKE_INSTALLED:
        return
    from models.vit_branch import SceneIQViT

    def _init(self, model_name="mock", patch_grid=patch_grid, dropout=0.0):
        nn.Module.__init__(self)
        self.backbone = FakeBackbone(hidden_dim=hidden_dim, patch_grid=patch_grid)
        self.hidden_dim = hidden_dim
        self.patch_grid = patch_grid
        self.dropout = nn.Dropout(dropout)
        self.scs_head = nn.Linear(hidden_dim, 1)
        self.loc_head = nn.Linear(hidden_dim, 1)

    SceneIQViT.__init__ = _init
    _FAKE_INSTALLED = True


# ── Fake image processor ─────────────────────────────────────────────────────

class FakeProcessor:
    """Minimal stand-in for ``ViTImageProcessor``.

    Resizes to 224×224, normalizes to [0, 1], and returns a (1, 3, 224, 224)
    tensor under ``pixel_values`` so SceneIQDataset can consume it.
    """

    def __call__(self, images, return_tensors=None):
        img = images if not isinstance(images, list) else images[0]
        arr = np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": t}


# ── Synthetic-image helper ───────────────────────────────────────────────────

def write_grey_jpeg(path: Path, size=(64, 64), colour=(128, 128, 128)) -> Path:
    """Write a uniform RGB JPEG of ``size`` to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=colour).save(path, "JPEG")
    return path


# ── Small GAT factory used all over the tests ────────────────────────────────

def make_small_gat(
    n_objects: int = 50,
    n_predicates: int = 20,
    embed_dim: int = 16,
    hidden_dim: int = 32,
    num_heads: int = 2,
    num_layers: int = 1,
):
    """Build a deliberately tiny SceneIQGAT so forward tests run fast."""
    from models.gat_branch import SceneIQGAT
    return SceneIQGAT(
        n_objects=n_objects, n_predicates=n_predicates,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers, dropout=0.0,
    )
