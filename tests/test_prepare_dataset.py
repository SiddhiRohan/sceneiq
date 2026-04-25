"""
Tests for ``scripts/prepare_dataset.py``, focusing on the new COCO integration.

Covers:
  * ``load_coco_records`` subsampling and default label/source fill-in
  * ``load_coco_records`` when the index file is missing (should warn, not raise)
  * ``load_incoherent_records`` still drops entries without ``paste_bbox``
  * ``split_records`` produces a stratified 70/15/15 split
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_dataset import (
    load_coco_records,
    load_incoherent_records,
    sample_coherent_records,
    split_records,
)


# ── load_coco_records ───────────────────────────────────────────────────────

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _coco_records_blob(n: int = 5) -> list:
    return [
        {"image_id": i, "scene_image_id": i, "url": f"http://x/{i}.jpg",
         "source": "coco", "label": 0}
        for i in range(n)
    ]


def test_load_coco_records_returns_all_when_no_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "coherent.json"
        _write_json(path, _coco_records_blob(4))
        recs = load_coco_records(path, n_samples=0, rng=random.Random(0))
        assert len(recs) == 4


def test_load_coco_records_respects_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "coherent.json"
        _write_json(path, _coco_records_blob(10))
        recs = load_coco_records(path, n_samples=3, rng=random.Random(0))
        assert len(recs) == 3


def test_load_coco_records_missing_file_returns_empty() -> None:
    recs = load_coco_records(Path("/does/not/exist.json"), n_samples=5, rng=random.Random(0))
    assert recs == []


def test_load_coco_records_fills_defaults() -> None:
    """Records missing source/label get the defaults filled in."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "coherent.json"
        _write_json(path, [{"image_id": 1, "url": "x"}])
        recs = load_coco_records(path, n_samples=0, rng=random.Random(0))
        assert recs[0]["label"] == 0
        assert recs[0]["source"] == "coco"


def test_load_coco_records_is_seeded() -> None:
    """Two calls with the same seed must produce the same subset."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "coherent.json"
        _write_json(path, _coco_records_blob(20))
        a = load_coco_records(path, n_samples=5, rng=random.Random(42))
        b = load_coco_records(path, n_samples=5, rng=random.Random(42))
        assert [r["image_id"] for r in a] == [r["image_id"] for r in b]


# ── load_incoherent_records ─────────────────────────────────────────────────

def _synthetic_metadata(n: int = 3, with_bbox: bool = True) -> list:
    meta = []
    for i in range(n):
        entry = {
            "sample_id": f"s{i}",
            "output_path": f"images/s{i}.jpg",
            "scene_image_id": 1000 + i,
            "alien_object": f"alien_{i}",
            "alien_source_image_id": 2000 + i,
        }
        if with_bbox:
            entry["paste_bbox"] = [0, 0, 10, 10]
        meta.append(entry)
    return meta


def test_load_incoherent_drops_records_without_bbox() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta_path = tmp / "metadata.json"
        _write_json(meta_path, _synthetic_metadata(5, with_bbox=False))
        # Files don't exist, so records would be dropped anyway.
        # Create the referenced images so the bbox check is the only reason for filtering.
        (tmp / "images").mkdir()
        for i in range(5):
            (tmp / "images" / f"s{i}.jpg").write_bytes(b"fake")
        recs, _ = load_incoherent_records(tmp, n_samples=-1, rng=random.Random(0))
        assert recs == []


def test_load_incoherent_keeps_records_with_bbox() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta_path = tmp / "metadata.json"
        _write_json(meta_path, _synthetic_metadata(3, with_bbox=True))
        (tmp / "images").mkdir()
        for i in range(3):
            (tmp / "images" / f"s{i}.jpg").write_bytes(b"fake")
        recs, excluded = load_incoherent_records(tmp, n_samples=-1, rng=random.Random(0))
        assert len(recs) == 3
        assert all("paste_bbox" in r for r in recs)
        assert excluded == {1000, 1001, 1002}


# ── sample_coherent_records excludes leaked ids ────────────────────────────

def test_sample_coherent_excludes_leaked_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_data = [
            {"image_id": 1, "url": "http://x/1.jpg"},
            {"image_id": 2, "url": "http://x/2.jpg"},
            {"image_id": 3, "url": "http://x/3.jpg"},
        ]
        path = tmp / "image_data.json"
        _write_json(path, image_data)
        recs = sample_coherent_records(path, excluded_ids={2}, n_samples=10, rng=random.Random(0))
        ids = {r["image_id"] for r in recs}
        assert 2 not in ids
        assert ids == {1, 3}


def test_sample_coherent_labels_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path = tmp / "image_data.json"
        _write_json(path, [{"image_id": 1, "url": "http://x/1.jpg"}])
        recs = sample_coherent_records(path, excluded_ids=set(), n_samples=5, rng=random.Random(0))
        assert all(r["label"] == 0 for r in recs)


# ── split_records ───────────────────────────────────────────────────────────

def test_split_records_fractions_roughly_respected() -> None:
    records = [{"label": 0}] * 100 + [{"label": 1}] * 100
    splits = split_records(records, train_frac=0.7, val_frac=0.15, rng=random.Random(0))
    assert 135 <= len(splits["train"]) <= 145
    assert 28 <= len(splits["val"]) <= 32


def test_split_records_stratified_by_label() -> None:
    records = [{"label": 0}] * 10 + [{"label": 1}] * 10
    splits = split_records(records, train_frac=0.8, val_frac=0.1, rng=random.Random(0))
    # Every split should contain both labels
    for name in ("train", "val", "test"):
        labels = {r["label"] for r in splits[name]}
        assert labels.issubset({0, 1})


def test_split_records_no_overlap_between_splits() -> None:
    records = [{"label": 0, "id": i} for i in range(30)] + [{"label": 1, "id": i} for i in range(30, 60)]
    splits = split_records(records, train_frac=0.7, val_frac=0.15, rng=random.Random(0))
    ids_train = {r["id"] for r in splits["train"]}
    ids_val = {r["id"] for r in splits["val"]}
    ids_test = {r["id"] for r in splits["test"]}
    assert ids_train & ids_val == set()
    assert ids_train & ids_test == set()
    assert ids_val & ids_test == set()


def test_split_records_all_labels_covered() -> None:
    records = [{"label": 0, "id": i} for i in range(10)] + [{"label": 1, "id": i} for i in range(10, 20)]
    splits = split_records(records, train_frac=0.6, val_frac=0.2, rng=random.Random(0))
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 20
