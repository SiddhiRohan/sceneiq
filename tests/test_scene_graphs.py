"""
Tests for ``scripts/build_scene_graphs.py``.

Covers the vocab builder and the per-image scene-graph encoder:
  * PAD + UNK always reserved
  * frequency-sorted, cap-respecting vocabs
  * area-ranked object truncation
  * edge rewrite onto local indices
  * graceful handling of malformed VG entries
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_scene_graphs import (
    PAD_TOKEN,
    UNK_TOKEN,
    _object_size,
    build_vocab,
    extract_graph,
)


# ── Vocab builder ────────────────────────────────────────────────────────────

def test_build_vocab_reserves_pad_and_unk() -> None:
    vocab = build_vocab(Counter({"dog": 10, "cat": 5}), max_size=10)
    assert vocab[PAD_TOKEN] == 0
    assert vocab[UNK_TOKEN] == 1


def test_build_vocab_respects_max_size() -> None:
    counts = Counter({f"w{i}": 100 - i for i in range(20)})
    vocab = build_vocab(counts, max_size=5)
    assert len(vocab) == 5


def test_build_vocab_ordered_by_frequency() -> None:
    counts = Counter({"rare": 1, "common": 100, "mid": 10})
    vocab = build_vocab(counts, max_size=10)
    assert vocab["common"] < vocab["mid"] < vocab["rare"]


def test_build_vocab_empty_counter() -> None:
    vocab = build_vocab(Counter(), max_size=10)
    assert vocab == {PAD_TOKEN: 0, UNK_TOKEN: 1}


def test_build_vocab_with_zero_counts_ignored() -> None:
    vocab = build_vocab(Counter({"only": 1}), max_size=10)
    assert "only" in vocab


def test_build_vocab_ids_are_contiguous() -> None:
    """IDs must be 0..N-1 with no gaps."""
    vocab = build_vocab(Counter({"a": 3, "b": 2, "c": 1}), max_size=10)
    ids = sorted(vocab.values())
    assert ids == list(range(len(ids)))


# ── _object_size ─────────────────────────────────────────────────────────────

def test_object_size_product_of_wh() -> None:
    assert _object_size({"w": 10, "h": 20}) == 200


def test_object_size_missing_fields_returns_zero() -> None:
    assert _object_size({}) == 0


def test_object_size_negative_dimensions_clamped() -> None:
    assert _object_size({"w": -5, "h": 10}) == 0
    assert _object_size({"w": 10, "h": -5}) == 0


def test_object_size_zero_dimensions() -> None:
    assert _object_size({"w": 0, "h": 10}) == 0


# ── extract_graph — happy path ──────────────────────────────────────────────

def _obj_vocab_basic() -> dict:
    return {PAD_TOKEN: 0, UNK_TOKEN: 1, "dog": 2, "ball": 3, "field": 4}


def _pred_vocab_basic() -> dict:
    return {PAD_TOKEN: 0, UNK_TOKEN: 1, "holding": 2, "on": 3}


def test_extract_graph_returns_objects_and_edges() -> None:
    entry = {
        "objects": [{"object_id": 1, "name": "dog", "w": 10, "h": 10}],
        "relationships": [],
    }
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert set(g.keys()) == {"objects", "edges"}
    assert isinstance(g["objects"], list)
    assert isinstance(g["edges"], list)


def test_extract_graph_maps_known_labels() -> None:
    entry = {
        "objects": [{"object_id": 1, "name": "dog", "w": 10, "h": 10}],
        "relationships": [],
    }
    obj_vocab = _obj_vocab_basic()
    g = extract_graph(entry, obj_vocab, _pred_vocab_basic())
    assert g["objects"] == [obj_vocab["dog"]]


def test_extract_graph_unknown_label_maps_to_unk() -> None:
    entry = {
        "objects": [{"object_id": 1, "name": "alpaca", "w": 10, "h": 10}],
        "relationships": [],
    }
    obj_vocab = _obj_vocab_basic()
    g = extract_graph(entry, obj_vocab, _pred_vocab_basic())
    assert g["objects"] == [obj_vocab[UNK_TOKEN]]


def test_extract_graph_sorts_objects_by_area() -> None:
    """Largest-area object first."""
    entry = {
        "objects": [
            {"object_id": 1, "name": "ball", "w": 5, "h": 5},        # 25
            {"object_id": 2, "name": "dog", "w": 20, "h": 20},       # 400
            {"object_id": 3, "name": "field", "w": 100, "h": 100},   # 10000
        ],
        "relationships": [],
    }
    obj_vocab = _obj_vocab_basic()
    g = extract_graph(entry, obj_vocab, _pred_vocab_basic())
    assert g["objects"][0] == obj_vocab["field"]
    assert g["objects"][-1] == obj_vocab["ball"]


def test_extract_graph_edges_rewritten_to_local_indices() -> None:
    entry = {
        "objects": [
            {"object_id": 1001, "name": "dog", "w": 50, "h": 50},
            {"object_id": 1002, "name": "ball", "w": 10, "h": 10},
        ],
        "relationships": [
            {"subject_id": 1001, "object_id": 1002, "predicate": "holding"},
        ],
    }
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert len(g["edges"]) == 1
    src, dst, pid = g["edges"][0]
    assert 0 <= src < 2 and 0 <= dst < 2
    assert src != dst


def test_extract_graph_edges_use_predicate_vocab() -> None:
    entry = {
        "objects": [
            {"object_id": 1, "name": "dog", "w": 10, "h": 10},
            {"object_id": 2, "name": "ball", "w": 10, "h": 10},
        ],
        "relationships": [
            {"subject_id": 1, "object_id": 2, "predicate": "holding"},
        ],
    }
    pred_vocab = _pred_vocab_basic()
    g = extract_graph(entry, _obj_vocab_basic(), pred_vocab)
    assert g["edges"][0][2] == pred_vocab["holding"]


def test_extract_graph_unknown_predicate_maps_to_unk() -> None:
    entry = {
        "objects": [
            {"object_id": 1, "name": "dog", "w": 10, "h": 10},
            {"object_id": 2, "name": "ball", "w": 10, "h": 10},
        ],
        "relationships": [
            {"subject_id": 1, "object_id": 2, "predicate": "floofing"},
        ],
    }
    pred_vocab = _pred_vocab_basic()
    g = extract_graph(entry, _obj_vocab_basic(), pred_vocab)
    assert g["edges"][0][2] == pred_vocab[UNK_TOKEN]


# ── extract_graph — edge cases ──────────────────────────────────────────────

def test_extract_graph_empty_objects() -> None:
    g = extract_graph({"objects": [], "relationships": []}, _obj_vocab_basic(), _pred_vocab_basic())
    assert g["objects"] == []
    assert g["edges"] == []


def test_extract_graph_missing_objects_key() -> None:
    """A VG entry with no objects key should still produce something valid."""
    g = extract_graph({}, _obj_vocab_basic(), _pred_vocab_basic())
    assert g["objects"] == []
    assert g["edges"] == []


def test_extract_graph_empty_relationships() -> None:
    entry = {"objects": [{"object_id": 1, "name": "dog", "w": 10, "h": 10}]}
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert g["objects"] == [2]
    assert g["edges"] == []


def test_extract_graph_relationship_to_missing_object_dropped() -> None:
    """Edge referencing an object_id that isn't in the (truncated) node list is dropped."""
    entry = {
        "objects": [{"object_id": 1, "name": "dog", "w": 10, "h": 10}],
        "relationships": [
            {"subject_id": 1, "object_id": 999, "predicate": "holding"},  # obj 999 doesn't exist
        ],
    }
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert g["edges"] == []


def test_extract_graph_names_list_variant() -> None:
    """VG sometimes puts the label under "names": [...] instead of "name": ..."""
    entry = {
        "objects": [{"object_id": 1, "names": ["Dog"], "w": 10, "h": 10}],
        "relationships": [],
    }
    obj_vocab = _obj_vocab_basic()
    g = extract_graph(entry, obj_vocab, _pred_vocab_basic())
    assert g["objects"] == [obj_vocab["dog"]]


def test_extract_graph_nested_subject_object() -> None:
    """Some VG records nest subject/object dicts instead of using *_id."""
    entry = {
        "objects": [
            {"object_id": 1, "name": "dog", "w": 10, "h": 10},
            {"object_id": 2, "name": "ball", "w": 10, "h": 10},
        ],
        "relationships": [
            {
                "subject": {"object_id": 1},
                "object": {"object_id": 2},
                "predicate": "holding",
            }
        ],
    }
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert len(g["edges"]) == 1


def test_extract_graph_self_loop_kept() -> None:
    """Self-loops (same subject and object) are valid graph edges."""
    entry = {
        "objects": [{"object_id": 1, "name": "dog", "w": 10, "h": 10}],
        "relationships": [
            {"subject_id": 1, "object_id": 1, "predicate": "holding"},
        ],
    }
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert len(g["edges"]) == 1
    assert g["edges"][0][0] == g["edges"][0][1]


def test_extract_graph_caps_objects_at_sg_max_objects() -> None:
    """The top-K truncation must respect SG_MAX_OBJECTS."""
    from config import SG_MAX_OBJECTS
    many_objects = [
        {"object_id": i, "name": "dog", "w": 10 + i, "h": 10 + i}
        for i in range(SG_MAX_OBJECTS + 10)
    ]
    entry = {"objects": many_objects, "relationships": []}
    g = extract_graph(entry, _obj_vocab_basic(), _pred_vocab_basic())
    assert len(g["objects"]) == SG_MAX_OBJECTS
