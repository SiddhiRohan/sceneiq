"""
Tests for ``scripts/build_coco_index.py``.

Feeds toy COCO-shaped annotation blobs through each helper and verifies the
record format, category mapping, caption grouping, and co-occurrence counts.
Also integration-tests the full ``main()`` entrypoint on a tempdir.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_coco_index import (
    assemble_records,
    build_category_index,
    build_image_index,
    build_pair_counts,
    captions_per_image,
    main as coco_main,
    objects_per_image,
)


# ── Toy COCO blobs ───────────────────────────────────────────────────────────

def _toy_instances() -> dict:
    return {
        "images": [
            {"id": 1, "coco_url": "http://x/1.jpg", "width": 640, "height": 480,
             "file_name": "000000000001.jpg"},
            {"id": 2, "flickr_url": "http://y/2.jpg", "width": 300, "height": 200,
             "file_name": "000000000002.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "dog"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1},
            {"image_id": 1, "category_id": 2},
            {"image_id": 2, "category_id": 1},
            {"image_id": 2, "category_id": 3},
            {"image_id": 1, "category_id": 1},   # duplicate category on image 1
        ],
    }


def _toy_captions() -> dict:
    return {
        "annotations": [
            {"image_id": 1, "caption": "A person on a bicycle"},
            {"image_id": 1, "caption": "Someone riding a bike"},
            {"image_id": 2, "caption": "A dog runs through the park"},
        ],
    }


# ── Category / image / caption builders ─────────────────────────────────────

def test_build_category_index_maps_id_to_name() -> None:
    idx = build_category_index(_toy_instances()["categories"])
    assert idx == {1: "person", 2: "bicycle", 3: "dog"}


def test_build_category_index_empty_list() -> None:
    assert build_category_index([]) == {}


def test_build_image_index_uses_coco_url_when_present() -> None:
    idx = build_image_index(_toy_instances()["images"])
    assert idx[1]["url"] == "http://x/1.jpg"


def test_build_image_index_falls_back_to_flickr_url() -> None:
    idx = build_image_index(_toy_instances()["images"])
    assert idx[2]["url"] == "http://y/2.jpg"


def test_build_image_index_preserves_dims() -> None:
    idx = build_image_index(_toy_instances()["images"])
    assert idx[1]["width"] == 640 and idx[1]["height"] == 480


# ── objects_per_image ───────────────────────────────────────────────────────

def test_objects_per_image_deduplicates_categories() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    per_img = objects_per_image(inst["annotations"], cat)
    assert per_img[1] == sorted({"person", "bicycle"})


def test_objects_per_image_respects_categories_per_image() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    per_img = objects_per_image(inst["annotations"], cat)
    assert per_img[2] == sorted({"person", "dog"})


def test_objects_per_image_skips_unknown_category() -> None:
    cat = {1: "person"}
    annotations = [{"image_id": 1, "category_id": 999}]  # id not in cat
    per_img = objects_per_image(annotations, cat)
    assert per_img.get(1, []) == []


def test_objects_per_image_handles_missing_keys() -> None:
    annotations = [{}, {"image_id": 1}, {"category_id": 1}]
    per_img = objects_per_image(annotations, {1: "person"})
    assert per_img == {}


# ── captions_per_image ──────────────────────────────────────────────────────

def test_captions_per_image_groups_by_image_id() -> None:
    per_img = captions_per_image(_toy_captions()["annotations"])
    assert len(per_img[1]) == 2
    assert len(per_img[2]) == 1


def test_captions_per_image_strips_whitespace() -> None:
    anns = [{"image_id": 1, "caption": "   hello world  "}]
    per_img = captions_per_image(anns)
    assert per_img[1] == ["hello world"]


def test_captions_per_image_skips_blank() -> None:
    anns = [{"image_id": 1, "caption": "   "}]
    per_img = captions_per_image(anns)
    assert per_img == {}


# ── Pair counts ─────────────────────────────────────────────────────────────

def test_build_pair_counts_symmetric_sort() -> None:
    per = {1: ["dog", "ball"], 2: ["ball", "cat"]}
    counts = build_pair_counts(per)
    # Keys are sorted "a|b" with a <= b
    assert "ball|dog" in counts
    assert "ball|cat" in counts
    assert "cat|dog" not in counts   # never co-occurred


def test_build_pair_counts_accumulates_across_images() -> None:
    per = {1: ["a", "b"], 2: ["a", "b"]}
    counts = build_pair_counts(per)
    assert counts["a|b"] == 2


def test_build_pair_counts_single_category_no_pair() -> None:
    per = {1: ["only"]}
    counts = build_pair_counts(per)
    assert counts == {}


def test_build_pair_counts_triangle() -> None:
    per = {1: ["a", "b", "c"]}
    counts = build_pair_counts(per)
    assert set(counts.keys()) == {"a|b", "a|c", "b|c"}


# ── assemble_records ────────────────────────────────────────────────────────

def test_assemble_records_labels_all_coherent() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    img_idx = build_image_index(inst["images"])
    per_obj = objects_per_image(inst["annotations"], cat)
    per_cap = captions_per_image(_toy_captions()["annotations"])
    recs = assemble_records(img_idx, per_obj, per_cap)
    assert all(r["label"] == 0 for r in recs)


def test_assemble_records_source_is_coco() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    img_idx = build_image_index(inst["images"])
    per_obj = objects_per_image(inst["annotations"], cat)
    recs = assemble_records(img_idx, per_obj, {})
    assert all(r["source"] == "coco" for r in recs)


def test_assemble_records_scene_image_id_mirrors_image_id() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    img_idx = build_image_index(inst["images"])
    per_obj = objects_per_image(inst["annotations"], cat)
    recs = assemble_records(img_idx, per_obj, {})
    for r in recs:
        assert r["image_id"] == r["scene_image_id"]


def test_assemble_records_empty_for_empty_index() -> None:
    assert assemble_records({}, {}, {}) == []


def test_assemble_records_captions_attached_only_where_present() -> None:
    inst = _toy_instances()
    cat = build_category_index(inst["categories"])
    img_idx = build_image_index(inst["images"])
    per_obj = objects_per_image(inst["annotations"], cat)
    per_cap = captions_per_image(_toy_captions()["annotations"])
    recs = {r["image_id"]: r for r in assemble_records(img_idx, per_obj, per_cap)}
    assert len(recs[1]["captions"]) == 2
    assert len(recs[2]["captions"]) == 1


# ── Integration: main() ─────────────────────────────────────────────────────

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_coco_main_end_to_end() -> None:
    """main() consumes on-disk COCO files and writes both output JSONs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coco_dir = tmp / "mscoco"
        output_dir = tmp / "processed"
        ann_dir = coco_dir / "annotations"
        _write_json(ann_dir / "instances_val2017.json", _toy_instances())
        _write_json(ann_dir / "captions_val2017.json", _toy_captions())

        coco_main(coco_dir=coco_dir, split="val2017", output_dir=output_dir)

        rec_path = output_dir / "coherent_records_val2017.json"
        pc_path = output_dir / "coco_pair_counts_val2017.json"
        assert rec_path.exists()
        assert pc_path.exists()
        recs = json.loads(rec_path.read_text())
        assert len(recs) == 2
        pc = json.loads(pc_path.read_text())
        # Each image had ≥2 objects, so there's at least one pair
        assert len(pc) >= 1


def test_coco_main_missing_instances_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coco_dir = tmp / "mscoco"
        output_dir = tmp / "processed"
        (coco_dir / "annotations").mkdir(parents=True)
        try:
            coco_main(coco_dir=coco_dir, split="val2017", output_dir=output_dir)
        except FileNotFoundError:
            return
        raise AssertionError("coco_main should raise FileNotFoundError when instances JSON is missing")


def test_coco_main_works_without_captions_file() -> None:
    """Captions file is optional — absence should not block the run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coco_dir = tmp / "mscoco"
        output_dir = tmp / "processed"
        _write_json(coco_dir / "annotations" / "instances_val2017.json", _toy_instances())
        coco_main(coco_dir=coco_dir, split="val2017", output_dir=output_dir)
        recs = json.loads((output_dir / "coherent_records_val2017.json").read_text())
        # captions list empty
        assert all(r["captions"] == [] for r in recs)
