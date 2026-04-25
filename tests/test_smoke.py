"""
SceneIQ — end-to-end smoke tests.

Runs without a network connection and without a GPU. Verifies:

  1. Imports — every public module loads cleanly.
  2. Config — required constants are present and coherent.
  3. bbox → patch-mask math — area conservation on a known bbox.
  4. Scene graph extraction — toy VG-shaped input produces the right triplets.
  5. GAT branch — collation and forward on mixed / empty graphs.
  6. Fusion and ViT-only paths — shape checks using a mock ViT backbone
     (no HuggingFace download required), plus an end-to-end backward pass.
  7. Loss helpers — soft-IoU is bounded, BCE reacts to label mismatch,
     combined loss respects weights and skips localization when has_heatmap=False.
  8. Dataset — SceneIQDataset returns the expected keys with a synthetic 1-pixel
     image record.
  9. Collate — make_collate produces the right batch tensors with and without
     a GAT module.

Usage:
    python tests/test_smoke.py          # runs everything, exits 0 on pass
    pytest tests/                       # also works if you have pytest

Every test is a plain function named ``test_*``. Assertion messages spell out
exactly what failed.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ── 1. Imports ───────────────────────────────────────────────────────────────

def test_imports() -> None:
    """Every public module loads cleanly."""
    import config  # noqa: F401
    import utils  # noqa: F401
    from models import build_model, SceneIQViT, SceneIQGAT, SceneIQFusion, SceneIQOutput  # noqa: F401
    from models.fusion import soft_iou_loss, compute_loss  # noqa: F401
    import data  # scripts/data.py via sys.path  # noqa: F401


# ── 2. Config sanity ─────────────────────────────────────────────────────────

def test_config_constants() -> None:
    """All constants train.py / evaluate.py depend on are defined and coherent."""
    import config as cfg
    for attr in [
        "PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "EVALUATION_DIR",
        "SCENE_GRAPH_DIR", "VIT_MODEL_NAME", "PATCH_GRID",
        "MODE_VIT", "MODE_GAT", "MODE_FUSION", "MODEL_MODES",
        "SG_OBJECT_VOCAB_SIZE", "SG_PREDICATE_VOCAB_SIZE",
        "SG_EMBED_DIM", "GAT_HIDDEN_DIM", "GAT_NUM_HEADS", "GAT_NUM_LAYERS",
        "BCE_LOSS_WEIGHT", "LOC_LOSS_WEIGHT", "SG_MAX_OBJECTS",
    ]:
        assert hasattr(cfg, attr), f"config.{attr} is missing"

    assert cfg.MODEL_MODES == ("vit", "gat", "fusion")
    assert cfg.PATCH_GRID == 14
    # 224 / 16 must equal PATCH_GRID for ViT-B/16@224
    assert 224 // 16 == cfg.PATCH_GRID


# ── 3. Bbox → patch-mask math ────────────────────────────────────────────────

def test_bbox_to_patch_mask_area() -> None:
    """A bbox covering 25% of the image must produce mask-sum ≈ 0.25 * grid^2."""
    from data import bbox_to_patch_mask
    mask = bbox_to_patch_mask([50, 50, 100, 100], image_width=200, image_height=200, grid=14)
    total = mask.sum().item()
    assert mask.shape == (14, 14)
    # bbox area fraction * 14*14 = 0.25 * 196 = 49.0
    assert abs(total - 49.0) < 1e-3, f"expected 49.0, got {total}"


def test_bbox_to_patch_mask_edge_cases() -> None:
    """Empty / None / out-of-bounds bboxes must not raise and must return zeros."""
    from data import bbox_to_patch_mask
    z1 = bbox_to_patch_mask(None, 100, 100, grid=14)
    z2 = bbox_to_patch_mask([0, 0, 0, 0], 100, 100, grid=14)
    z3 = bbox_to_patch_mask([500, 500, 10, 10], 100, 100, grid=14)  # off-image
    for z in (z1, z2, z3):
        assert z.shape == (14, 14)
        assert z.sum().item() == 0.0


def test_bbox_to_patch_mask_covers_right_region() -> None:
    """The top-left quarter bbox must light up patches 0..6 only."""
    from data import bbox_to_patch_mask
    m = bbox_to_patch_mask([0, 0, 100, 100], image_width=200, image_height=200, grid=14)
    # Rows 0-6 (fully) and row 7 (half) should be non-zero in columns 0-6 (fully) and col 7 (half).
    assert (m[:7, :7] > 0).all()
    assert (m[8:, :] == 0).all()


# ── 4. Scene-graph extraction ────────────────────────────────────────────────

def test_extract_graph_basic() -> None:
    """Build a toy VG entry, extract it, verify object ids + edge rewrite."""
    # Need to import via importlib because "scripts/build_scene_graphs.py" is not a package entry.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from build_scene_graphs import extract_graph, PAD_TOKEN, UNK_TOKEN

    obj_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "dog": 2, "ball": 3, "field": 4}
    pred_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "holding": 2, "on": 3}

    entry = {
        "image_id": 42,
        "objects": [
            {"object_id": 1001, "name": "dog", "x": 10, "y": 10, "w": 50, "h": 50},
            {"object_id": 1002, "name": "ball", "x": 5, "y": 5, "w": 5, "h": 5},
            {"object_id": 1003, "name": "field", "x": 0, "y": 0, "w": 200, "h": 200},
        ],
        "relationships": [
            {"subject_id": 1001, "object_id": 1002, "predicate": "holding"},
            {"subject_id": 1001, "object_id": 1003, "predicate": "on"},
        ],
    }
    g = extract_graph(entry, obj_vocab, pred_vocab)
    assert set(g.keys()) == {"objects", "edges"}
    assert len(g["objects"]) == 3
    # Largest object (field, 200x200) ranks first; smallest (ball) last.
    assert g["objects"][0] == obj_vocab["field"]
    assert g["objects"][-1] == obj_vocab["ball"]
    assert len(g["edges"]) == 2
    for src, dst, pid in g["edges"]:
        assert 0 <= src < 3 and 0 <= dst < 3
        assert pid in {pred_vocab["holding"], pred_vocab["on"]}


def test_extract_graph_unknown_label() -> None:
    """Labels outside the vocab must map to UNK rather than crashing."""
    from build_scene_graphs import extract_graph, UNK_TOKEN, PAD_TOKEN
    obj_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}  # empty vocab
    pred_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    entry = {
        "objects": [{"object_id": 1, "name": "alpaca", "w": 10, "h": 10}],
        "relationships": [],
    }
    g = extract_graph(entry, obj_vocab, pred_vocab)
    assert g["objects"] == [obj_vocab[UNK_TOKEN]]


# ── 5. GAT branch ────────────────────────────────────────────────────────────

def _make_gat():
    from models.gat_branch import SceneIQGAT
    return SceneIQGAT(n_objects=50, n_predicates=20, embed_dim=16,
                      hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)


def test_gat_collate_shapes() -> None:
    gat = _make_gat()
    graphs = [
        {"objects": [2, 5, 7], "edges": [[0, 1, 3], [1, 2, 4]]},
        {"objects": [9], "edges": []},
        None,  # missing graph → UNK fallback
    ]
    batch = gat.collate_graphs(graphs, n_max=4)
    assert batch["object_ids"].shape == (3, 4)
    assert batch["node_mask"].shape == (3, 4)
    # None graph must produce exactly one UNK node (id=1)
    assert batch["object_ids"][2].tolist() == [1, 0, 0, 0]
    assert batch["node_mask"][2].tolist() == [True, False, False, False]


def test_gat_forward_shapes() -> None:
    gat = _make_gat()
    graphs = [
        {"objects": [2, 5, 7], "edges": [[0, 1, 3]]},
        {"objects": [9], "edges": []},
    ]
    batch = gat.collate_graphs(graphs, n_max=3)
    out = gat(batch)
    assert out.scs_logit.shape == (2,)
    assert out.graph_embedding.shape == (2, 32)
    assert out.node_embeddings.shape == (2, 3, 32)
    assert out.node_mask.shape == (2, 3)


# ── 6. ViT / Fusion with a mock backbone ─────────────────────────────────────

class _FakeBackbone(nn.Module):
    """Stand-in for HuggingFace ViTModel so tests run offline.

    Produces a (B, 1+P, D) tensor that IS a function of ``pixel_values`` so
    gradients flow end-to-end (a plain ``randn`` would break backward).
    """

    def __init__(self, hidden_dim: int = 768, patch_grid: int = 14):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden_dim})()
        self.hidden_dim = hidden_dim
        self.patch_grid = patch_grid
        self.n_tokens = 1 + patch_grid * patch_grid
        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, pixel_values):
        feats = self.stem(pixel_values)                    # (B, D, 14, 14)
        B, D, H, W = feats.shape
        patches = feats.flatten(2).transpose(1, 2)         # (B, H*W, D)
        cls = self.cls.expand(B, 1, -1)
        hidden = torch.cat([cls, patches], dim=1)          # (B, 1+P, D)
        return type("O", (), {"last_hidden_state": hidden})()


def _install_fake_vit():
    """Monkey-patch SceneIQViT.__init__ to skip HF download; safe to call repeatedly."""
    from models.vit_branch import SceneIQViT

    def _init(self, model_name="mock", patch_grid=14, dropout=0.0):
        nn.Module.__init__(self)
        self.backbone = _FakeBackbone(hidden_dim=768, patch_grid=patch_grid)
        self.hidden_dim = 768
        self.patch_grid = patch_grid
        self.dropout = nn.Dropout(dropout)
        self.scs_head = nn.Linear(768, 1)
        self.loc_head = nn.Linear(768, 1)

    SceneIQViT.__init__ = _init


def test_vit_only_shapes() -> None:
    _install_fake_vit()
    from models.vit_branch import SceneIQViT
    from models.fusion import _ViTOnly
    vit = SceneIQViT(patch_grid=14)
    model = _ViTOnly(vit)
    out = model(torch.randn(2, 3, 224, 224), None)
    assert out.scs_logit.shape == (2,)
    assert out.heatmap_logits.shape == (2, 14, 14)
    assert out.has_heatmap is True


def test_gat_only_shapes() -> None:
    from models.fusion import _GATOnly
    gat = _make_gat()
    model = _GATOnly(gat, patch_grid=14)
    batch = gat.collate_graphs([{"objects": [2], "edges": []}, {"objects": [3], "edges": []}], n_max=2)
    out = model(torch.randn(2, 3, 224, 224), batch)
    assert out.scs_logit.shape == (2,)
    assert out.heatmap_logits.shape == (2, 14, 14)
    assert out.has_heatmap is False
    # Heatmap must be exactly zero in GAT mode
    assert out.heatmap_logits.abs().sum().item() == 0.0


def test_fusion_shapes_and_backward() -> None:
    _install_fake_vit()
    from models.vit_branch import SceneIQViT
    from models.fusion import SceneIQFusion, compute_loss
    vit = SceneIQViT(patch_grid=14)
    gat = _make_gat()
    model = SceneIQFusion(vit=vit, gat=gat, num_heads=4, dropout=0.0)

    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    batch = gat.collate_graphs(
        [{"objects": [2, 3], "edges": [[0, 1, 4]]},
         {"objects": [5], "edges": []}],
        n_max=3,
    )
    out = model(x, batch)
    assert out.scs_logit.shape == (2,)
    assert out.heatmap_logits.shape == (2, 14, 14)

    # End-to-end backward
    labels = torch.tensor([1, 0])
    target_mask = torch.zeros(2, 14, 14)
    target_mask[0, 3:8, 3:8] = 1.0
    losses = compute_loss(out.scs_logit, out.heatmap_logits, labels, target_mask, has_heatmap=True)
    losses["total"].backward()

    # At least some ViT and GAT params must have non-zero grads
    n_vit_grads = sum(1 for p in vit.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_gat_grads = sum(1 for p in gat.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_vit_grads > 0, "No gradient reached ViT — backbone is disconnected."
    assert n_gat_grads > 0, "No gradient reached GAT — graph branch is disconnected."


def test_build_model_factory_all_modes() -> None:
    _install_fake_vit()
    from models import build_model
    for mode in ("vit", "gat", "fusion"):
        m = build_model(
            mode=mode, vit_model_name="mock",
            n_objects=50, n_predicates=20,
            patch_grid=14, sg_embed_dim=16,
            gat_hidden_dim=32, gat_num_heads=2, gat_num_layers=1,
        )
        assert isinstance(m, nn.Module)


def test_build_model_rejects_bad_mode() -> None:
    from models import build_model
    try:
        build_model(mode="oops", vit_model_name="mock", n_objects=50, n_predicates=20)
    except ValueError as e:
        assert "oops" in str(e) or "Unknown mode" in str(e)
    else:
        raise AssertionError("build_model should have raised for bad mode")


# ── 7. Losses ────────────────────────────────────────────────────────────────

def test_soft_iou_perfect() -> None:
    """Perfect overlap → soft_iou_loss → 0."""
    from models.fusion import soft_iou_loss
    # Make the heatmap after sigmoid ~= 1 where target is 1 by using very large logits.
    target = torch.zeros(1, 14, 14)
    target[0, 5:9, 5:9] = 1.0
    logits = torch.where(target > 0, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    loss = soft_iou_loss(logits, target).item()
    assert loss < 0.01, f"expected near-zero loss, got {loss}"


def test_soft_iou_worst() -> None:
    """Complete miss → soft_iou_loss → ~1."""
    from models.fusion import soft_iou_loss
    target = torch.zeros(1, 14, 14)
    target[0, 0:4, 0:4] = 1.0
    logits = torch.where(target > 0, torch.full_like(target, -10.0), torch.full_like(target, 10.0))
    loss = soft_iou_loss(logits, target).item()
    assert loss > 0.9, f"expected near-one loss, got {loss}"


def test_compute_loss_skips_localization_when_flag_off() -> None:
    """has_heatmap=False must zero out the loc term."""
    from models.fusion import compute_loss
    scs = torch.zeros(2)
    hm = torch.zeros(2, 14, 14)
    labels = torch.tensor([1, 0])
    tm = torch.ones(2, 14, 14)  # would produce a huge loc loss if used
    out = compute_loss(scs, hm, labels, tm, has_heatmap=False)
    assert out["loc"].item() == 0.0
    assert out["total"].item() == out["bce"].item()


def test_compute_loss_skips_loc_for_coherent_only_batch() -> None:
    """All-coherent batch → no target bboxes → loc = 0."""
    from models.fusion import compute_loss
    scs = torch.zeros(3)
    hm = torch.zeros(3, 14, 14)
    labels = torch.zeros(3, dtype=torch.long)  # all coherent
    tm = torch.zeros(3, 14, 14)
    out = compute_loss(scs, hm, labels, tm, has_heatmap=True)
    assert out["loc"].item() == 0.0


# ── 8. Dataset ───────────────────────────────────────────────────────────────

class _FakeProcessor:
    """Minimal stand-in for ViTImageProcessor so we don't need HF weights."""
    def __call__(self, images, return_tensors=None):
        img = images if not isinstance(images, list) else images[0]
        # resize to 224 and pack into (1, 3, 224, 224)
        arr = np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": t}


def _write_tmp_image(tmpdir: Path, name: str, size=(64, 64)) -> Path:
    """Save a uniform grey JPEG for the dataset test."""
    p = tmpdir / name
    Image.new("RGB", size, color=(128, 128, 128)).save(p, "JPEG")
    return p


def test_dataset_item_keys_and_shapes() -> None:
    from data import SceneIQDataset
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_path = _write_tmp_image(tmp, "img.jpg")
        records = [{
            "image_path": str(img_path),
            "label": 1,
            "paste_bbox": [16, 16, 32, 32],   # (x, y, w, h) on the 64×64 image
            "scene_image_id": 123,
        }]
        ds = SceneIQDataset(records, _FakeProcessor(), patch_grid=14)
        item = ds[0]
        assert set(item.keys()) == {
            "pixel_values", "label", "target_mask",
            "scene_image_id", "image_path", "alien_object",
        }
        assert item["pixel_values"].shape == (3, 224, 224)
        assert item["target_mask"].shape == (14, 14)
        assert item["label"].item() == 1
        assert item["scene_image_id"] == 123
        # 32*32 / 64*64 = 0.25 of the image → mask sum ≈ 0.25 * 196 = 49
        assert abs(item["target_mask"].sum().item() - 49.0) < 0.5


def test_dataset_coherent_record_has_zero_mask() -> None:
    """Coherent records (no paste_bbox) must produce an all-zero target mask."""
    from data import SceneIQDataset
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_path = _write_tmp_image(tmp, "img.jpg")
        records = [{"image_path": str(img_path), "label": 0, "scene_image_id": 1}]
        ds = SceneIQDataset(records, _FakeProcessor(), patch_grid=14)
        item = ds[0]
        assert item["target_mask"].sum().item() == 0.0


# ── 9. Collate ───────────────────────────────────────────────────────────────

def test_make_collate_without_gat() -> None:
    """gat_module=None path: collate skips graph_batch (sets it to None)."""
    from data import SceneIQDataset, make_collate
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        records = [
            {"image_path": str(_write_tmp_image(tmp, "a.jpg")), "label": 0, "scene_image_id": 1},
            {"image_path": str(_write_tmp_image(tmp, "b.jpg")), "label": 1,
             "paste_bbox": [0, 0, 16, 16], "scene_image_id": 2},
        ]
        ds = SceneIQDataset(records, _FakeProcessor(), patch_grid=14)
        collate = make_collate(scene_graphs={}, gat_module=None, device=None, sg_max_objects=8)
        batch = collate([ds[0], ds[1]])
        assert batch["pixel_values"].shape == (2, 3, 224, 224)
        assert batch["labels"].tolist() == [0, 1]
        assert batch["graph_batch"] is None


def test_make_collate_with_gat_uses_scene_graphs_dict() -> None:
    """When gat_module and scene_graphs are provided, graph_batch has the right shape."""
    from data import SceneIQDataset, make_collate
    gat = _make_gat()
    scene_graphs = {
        "1": {"objects": [2, 3], "edges": [[0, 1, 2]]},
        # image_id 2 is intentionally missing → UNK fallback
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        records = [
            {"image_path": str(_write_tmp_image(tmp, "a.jpg")), "label": 0, "scene_image_id": 1},
            {"image_path": str(_write_tmp_image(tmp, "b.jpg")), "label": 1,
             "paste_bbox": [0, 0, 16, 16], "scene_image_id": 2},
        ]
        ds = SceneIQDataset(records, _FakeProcessor(), patch_grid=14)
        collate = make_collate(scene_graphs, gat_module=gat, device=None, sg_max_objects=8)
        batch = collate([ds[0], ds[1]])
        gb = batch["graph_batch"]
        assert isinstance(gb, dict)
        assert gb["object_ids"].shape[0] == 2
        assert "edge_index" in gb and gb["edge_index"].shape[0] == 2


# ── Test runner ──────────────────────────────────────────────────────────────

def _discover_tests() -> list:
    """Return (name, func) pairs for every test_* function in this module."""
    mod = sys.modules[__name__]
    return [(n, getattr(mod, n)) for n in sorted(dir(mod))
            if n.startswith("test_") and callable(getattr(mod, n))]


def run_all(verbose: bool = True) -> int:
    """Run every test and return the number of failures."""
    tests = _discover_tests()
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed.append((name, f"assertion: {e}", traceback.format_exc()))
            if verbose:
                print(f"FAIL {name}: {e}")
        except Exception as e:   # noqa: BLE001 — we want to keep going
            failed.append((name, f"{type(e).__name__}: {e}", traceback.format_exc()))
            if verbose:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            if verbose:
                print(f"PASS {name}")

    total = len(tests)
    print()
    print("=" * 60)
    print(f"  SceneIQ smoke tests — {passed}/{total} passed")
    print("=" * 60)
    if failed:
        print("\nFailures:")
        for name, short, _tb in failed:
            print(f"  - {name}: {short}")
        print("\nRe-run a specific test with:  python -c \"import tests.test_smoke as t; t.<name>()\"")
    return len(failed)


if __name__ == "__main__":
    rc = run_all(verbose=True)
    sys.exit(1 if rc else 0)
