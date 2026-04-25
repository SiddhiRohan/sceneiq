"""
Tests for ``scripts/build_visualcomet_index.py``.

Feeds toy VisualCOMET-shaped records through the helpers and the end-to-end
main() entrypoint. Verifies aggregation, dedup, string normalisation, and
file output shape.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_visualcomet_index import (
    _dedup_strip,
    aggregate_records,
    main as vc_main,
    merge_splits,
)


# ── _dedup_strip ────────────────────────────────────────────────────────────

def test_dedup_strip_preserves_order() -> None:
    assert _dedup_strip(["alpha", "beta", "alpha"]) == ["alpha", "beta"]


def test_dedup_strip_lowercases_and_strips() -> None:
    assert _dedup_strip(["  Alpha ", "alpha"]) == ["alpha"]


def test_dedup_strip_skips_empty() -> None:
    assert _dedup_strip(["", "  ", "x"]) == ["x"]


def test_dedup_strip_flattens_list_of_lists() -> None:
    """VisualCOMET sometimes stores a list[list[str]] — we flatten 1 level."""
    assert _dedup_strip([["one", "two"], ["two", "three"]]) == ["one", "two", "three"]


def test_dedup_strip_handles_none_input() -> None:
    assert _dedup_strip(None) == []


def test_dedup_strip_ignores_non_strings() -> None:
    assert _dedup_strip([1, {"x": 1}, "real"]) == ["real"]


# ── aggregate_records ───────────────────────────────────────────────────────

def _toy_vc(img: str = "img1.jpg") -> list:
    return [
        {
            "img_fn": img,
            "event": "a woman reads a book",
            "place": "living room",
            "intent": ["to relax", "to escape reality"],
            "before": ["picked up the book"],
            "after": ["turned the page"],
        },
        {
            "img_fn": img,
            "event": "A woman reads a book",     # case dup
            "place": "Living room",              # case dup
            "intent": ["to relax"],
            "before": [],
            "after": ["kept reading"],
        },
    ]


def test_aggregate_by_img_fn() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg") + _toy_vc("img2.jpg"))
    assert set(agg.keys()) == {"img1.jpg", "img2.jpg"}


def test_aggregate_counts_records_per_image() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert agg["img1.jpg"]["num_records"] == 2


def test_aggregate_dedupes_events() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert agg["img1.jpg"]["events"] == ["a woman reads a book"]


def test_aggregate_dedupes_places() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert agg["img1.jpg"]["places"] == ["living room"]


def test_aggregate_dedupes_intent_across_annotators() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert "to relax" in agg["img1.jpg"]["intent"]
    assert agg["img1.jpg"]["intent"].count("to relax") == 1


def test_aggregate_preserves_unique_intent() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert "to escape reality" in agg["img1.jpg"]["intent"]


def test_aggregate_handles_missing_img_fn() -> None:
    records = [{"event": "x", "intent": []}, {"img_fn": "a.jpg", "event": "y"}]
    agg = aggregate_records(records)
    assert list(agg.keys()) == ["a.jpg"]


def test_aggregate_handles_empty_input() -> None:
    assert aggregate_records([]) == {}


def test_aggregate_skips_none_value_fields() -> None:
    records = [{"img_fn": "a.jpg", "event": None, "intent": None}]
    agg = aggregate_records(records)
    assert agg["a.jpg"]["events"] == []
    assert agg["a.jpg"]["intent"] == []


def test_aggregate_before_and_after_collected() -> None:
    agg = aggregate_records(_toy_vc("img1.jpg"))
    assert "picked up the book" in agg["img1.jpg"]["before"]
    assert "turned the page" in agg["img1.jpg"]["after"]
    assert "kept reading" in agg["img1.jpg"]["after"]


# ── merge_splits ────────────────────────────────────────────────────────────

def test_merge_splits_combines_disjoint() -> None:
    a = {"img1.jpg": {"events": ["x"]}}
    b = {"img2.jpg": {"events": ["y"]}}
    merged = merge_splits(a, b)
    assert set(merged.keys()) == {"img1.jpg", "img2.jpg"}


def test_merge_splits_later_wins_on_collision() -> None:
    a = {"img1.jpg": {"events": ["old"]}}
    b = {"img1.jpg": {"events": ["new"]}}
    assert merge_splits(a, b)["img1.jpg"]["events"] == ["new"]


def test_merge_splits_no_args_returns_empty() -> None:
    assert merge_splits() == {}


# ── main() ─────────────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_vc_main_end_to_end() -> None:
    """main() consumes annot JSONs and writes commonsense.json with correct shape."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vc_dir = tmp / "visualcomet"
        output_dir = tmp / "processed"
        _write_json(vc_dir / "train_annots.json", _toy_vc("img1.jpg"))
        _write_json(vc_dir / "val_annots.json", _toy_vc("img2.jpg"))

        vc_main(vc_dir=vc_dir, output_dir=output_dir)

        out_path = output_dir / "commonsense.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert set(data.keys()) == {"img1.jpg", "img2.jpg"}
        # Required sub-fields
        for v in data.values():
            for key in ("events", "places", "intent", "before", "after", "num_records"):
                assert key in v


def test_vc_main_missing_all_splits_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vc_dir = tmp / "visualcomet"
        vc_dir.mkdir(parents=True)
        try:
            vc_main(vc_dir=vc_dir, output_dir=tmp / "out")
        except RuntimeError:
            return
        raise AssertionError("vc_main should raise RuntimeError when no split JSONs exist")


def test_vc_main_single_split_works() -> None:
    """Having just train_annots.json is enough — missing val/test is tolerated."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vc_dir = tmp / "visualcomet"
        out_dir = tmp / "out"
        _write_json(vc_dir / "train_annots.json", _toy_vc("img1.jpg"))
        vc_main(vc_dir=vc_dir, output_dir=out_dir)
        data = json.loads((out_dir / "commonsense.json").read_text())
        assert "img1.jpg" in data
