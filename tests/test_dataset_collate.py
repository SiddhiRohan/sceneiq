"""
Tests for ``scripts/data.py::SceneIQDataset`` and ``make_collate``.

Verifies that each record makes it from disk to tensors intact, that the
patch-mask is derived correctly, and that the collate function assembles
both the image batch and the graph batch as train.py expects.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import torch
from PIL import Image

from data import SceneIQDataset, make_collate
from conftest import FakeProcessor, make_small_gat, write_grey_jpeg


# ── Small fixture factory ───────────────────────────────────────────────────

def _coherent_record(path: Path, sid: int = 1) -> dict:
    return {"image_path": str(path), "label": 0, "scene_image_id": sid}


def _incoherent_record(path: Path, bbox=(0, 0, 16, 16), sid: int = 2, alien: str = "boat") -> dict:
    return {
        "image_path": str(path),
        "label": 1,
        "paste_bbox": list(bbox),
        "scene_image_id": sid,
        "alien_object": alien,
    }


# ── Dataset — structural ─────────────────────────────────────────────────────

def test_dataset_len_matches_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / f"{i}.jpg")) for i in range(5)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert len(ds) == 5


def test_dataset_item_has_all_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        item = ds[0]
        for k in ("pixel_values", "label", "target_mask",
                  "scene_image_id", "image_path", "alien_object"):
            assert k in item, f"missing key {k!r}"


def test_dataset_pixel_values_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["pixel_values"].shape == (3, 224, 224)


def test_dataset_target_mask_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["target_mask"].shape == (14, 14)


def test_dataset_label_is_long_tensor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        label = ds[0]["label"]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


# ── Dataset — mask semantics ────────────────────────────────────────────────

def test_dataset_coherent_has_zero_mask() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["target_mask"].sum().item() == 0.0


def test_dataset_incoherent_has_nonzero_mask() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg", size=(64, 64))
        recs = [_incoherent_record(p, bbox=(16, 16, 32, 32))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["target_mask"].sum().item() > 0.0


def test_dataset_bbox_area_preserved() -> None:
    """25% bbox area on the source image → mask sum ≈ 49.0."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg", size=(64, 64))
        recs = [_incoherent_record(p, bbox=(16, 16, 32, 32))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        mask_sum = ds[0]["target_mask"].sum().item()
        assert abs(mask_sum - 49.0) < 1.0


def test_dataset_preserves_alien_object_string() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg")
        recs = [_incoherent_record(p, alien="penguin")]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["alien_object"] == "penguin"


def test_dataset_preserves_image_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg")
        recs = [_coherent_record(p)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["image_path"] == str(p)


def test_dataset_preserves_scene_image_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg")
        recs = [_coherent_record(p, sid=42)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["scene_image_id"] == 42


def test_dataset_scene_image_id_falls_back_to_image_id() -> None:
    """If scene_image_id is missing, image_id is used; else -1."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = write_grey_jpeg(tmp / "a.jpg")
        recs = [{"image_path": str(p), "label": 0, "image_id": 99}]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["scene_image_id"] == 99


def test_dataset_multiple_records_independent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p_c = write_grey_jpeg(tmp / "c.jpg")
        p_i = write_grey_jpeg(tmp / "i.jpg")
        recs = [_coherent_record(p_c), _incoherent_record(p_i)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        it0 = ds[0]
        it1 = ds[1]
        assert it0["label"].item() == 0
        assert it1["label"].item() == 1
        assert it0["target_mask"].sum().item() == 0
        assert it1["target_mask"].sum().item() > 0


def test_dataset_rgba_image_handled() -> None:
    """RGBA → RGB conversion must not crash the processor path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / "rgba.png"
        Image.new("RGBA", (64, 64), color=(128, 128, 128, 255)).save(p)
        recs = [{"image_path": str(p), "label": 0, "scene_image_id": 1}]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        assert ds[0]["pixel_values"].shape == (3, 224, 224)


# ── make_collate — no GAT ───────────────────────────────────────────────────

def test_collate_without_gat_has_no_graph_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, gat_module=None, device=None, sg_max_objects=4)
        batch = collate([ds[0]])
        assert batch["graph_batch"] is None


def test_collate_pixel_values_stacked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / f"{i}.jpg")) for i in range(3)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[i] for i in range(3)])
        assert batch["pixel_values"].shape == (3, 3, 224, 224)


def test_collate_labels_stacked_as_1d() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg")),
                _incoherent_record(write_grey_jpeg(tmp / "b.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[0], ds[1]])
        assert batch["labels"].shape == (2,)
        assert batch["labels"].tolist() == [0, 1]


def test_collate_target_masks_stacked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))] * 2
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[0], ds[1]])
        assert batch["target_mask"].shape == (2, 14, 14)


def test_collate_preserves_image_paths_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p0 = write_grey_jpeg(tmp / "first.jpg")
        p1 = write_grey_jpeg(tmp / "second.jpg")
        recs = [_coherent_record(p0), _coherent_record(p1)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[0], ds[1]])
        assert batch["image_paths"] == [str(p0), str(p1)]


def test_collate_preserves_scene_image_ids_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / f"{i}.jpg"), sid=i * 10) for i in range(3)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[i] for i in range(3)])
        assert batch["scene_image_ids"] == [0, 10, 20]


def test_collate_preserves_alien_objects_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [
            _incoherent_record(write_grey_jpeg(tmp / "a.jpg"), alien="boat"),
            _incoherent_record(write_grey_jpeg(tmp / "b.jpg"), alien="penguin"),
        ]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[0], ds[1]])
        assert batch["alien_objects"] == ["boat", "penguin"]


def test_collate_batch_size_1() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"))]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate({}, None, None, 4)
        batch = collate([ds[0]])
        assert batch["pixel_values"].shape == (1, 3, 224, 224)
        assert batch["labels"].shape == (1,)


# ── make_collate — with GAT ─────────────────────────────────────────────────

def test_collate_with_gat_produces_graph_batch_dict() -> None:
    gat = make_small_gat()
    graphs = {"1": {"objects": [2, 3], "edges": [[0, 1, 2]]}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"), sid=1)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate(graphs, gat, None, 8)
        batch = collate([ds[0]])
        assert isinstance(batch["graph_batch"], dict)
        for key in ("object_ids", "node_mask", "edge_index", "edge_attr"):
            assert key in batch["graph_batch"]


def test_collate_with_gat_missing_scene_graph_unk_fallback() -> None:
    """Scene id with no graph entry gets the 1-UNK-node fallback."""
    gat = make_small_gat()
    graphs = {}   # no graph entries at all
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"), sid=999)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate(graphs, gat, None, 8)
        batch = collate([ds[0]])
        # UNK id is 1
        assert batch["graph_batch"]["object_ids"][0, 0].item() == 1


def test_collate_with_gat_graph_batch_size_matches() -> None:
    gat = make_small_gat()
    graphs = {"1": {"objects": [2], "edges": []}, "2": {"objects": [3], "edges": []}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recs = [_coherent_record(write_grey_jpeg(tmp / "a.jpg"), sid=1),
                _coherent_record(write_grey_jpeg(tmp / "b.jpg"), sid=2)]
        ds = SceneIQDataset(recs, FakeProcessor(), patch_grid=14)
        collate = make_collate(graphs, gat, None, 8)
        batch = collate([ds[0], ds[1]])
        assert batch["graph_batch"]["object_ids"].shape[0] == 2
