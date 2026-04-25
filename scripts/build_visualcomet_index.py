"""
SceneIQ — Preprocess VisualCOMET into a per-image commonsense index.

VisualCOMET ships JSON files (``{train,val,test}_annots.json``) where each
record is an ``(img_fn, event, place, before, after, intent)`` tuple. The
same ``img_fn`` appears in many records (one per annotator), so we aggregate
by image and deduplicate the inference text.

The resulting ``commonsense.json`` is a per-image dict::

    {
      img_fn: {
        "events": [...],
        "places": [...],
        "intent": [...],
        "before": [...],
        "after": [...],
        "num_records": int,
      },
      ...
    }

This feeds three use cases:
  1. An auxiliary coherence signal (images with many intent/after inferences
     tend to be dynamic scenes — useful prior for filtering).
  2. A future text-side commonsense embedding branch.
  3. Qualitative analysis: matching an image's predicted SCS against the
     annotators' expected event.

The preprocessing is light — we only do string normalisation + dedup.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VISUALCOMET_DIR, VISUALCOMET_INDEX_DIR
from utils import load_json, save_json, setup_logging, timer

logger = logging.getLogger("sceneiq")


# ── Aggregation ──────────────────────────────────────────────────────────────

def _dedup_strip(values: list) -> list:
    """Lowercase, strip, and dedup while preserving first-seen order."""
    seen = set()
    out = []
    for v in values or []:
        if isinstance(v, list):
            # Some VisualCOMET fields contain list-of-list — flatten 1 level.
            for inner in v:
                if not isinstance(inner, str):
                    continue
                s = inner.strip().lower()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
        elif isinstance(v, str):
            s = v.strip().lower()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def aggregate_records(records: list) -> dict:
    """Group raw VisualCOMET annotations by ``img_fn``.

    Args:
        records: List of annotation dicts from one of the VC JSON files.

    Returns:
        Dict ``{img_fn: {events, places, intent, before, after, num_records}}``.
    """
    by_image: dict = defaultdict(lambda: {
        "events": [], "places": [], "intent": [],
        "before": [], "after": [], "num_records": 0,
    })
    for rec in records:
        key = rec.get("img_fn")
        if not key:
            continue
        entry = by_image[key]
        entry["num_records"] += 1
        for field in ("event", "place"):
            val = rec.get(field)
            if val:
                entry[f"{field}s"].append(val)
        for field in ("intent", "before", "after"):
            val = rec.get(field)
            if val is not None:
                entry[field].append(val)

    # Dedup + normalise
    out: dict = {}
    for key, entry in by_image.items():
        out[key] = {
            "events": _dedup_strip(entry["events"]),
            "places": _dedup_strip(entry["places"]),
            "intent": _dedup_strip(entry["intent"]),
            "before": _dedup_strip(entry["before"]),
            "after": _dedup_strip(entry["after"]),
            "num_records": entry["num_records"],
        }
    return out


def merge_splits(*per_split: dict) -> dict:
    """Merge multiple aggregated dicts. Later splits win on collision."""
    merged: dict = {}
    for d in per_split:
        merged.update(d)
    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(vc_dir: Path, output_dir: Path) -> None:
    """Build the per-image commonsense index from VisualCOMET JSON files."""
    vc_dir = Path(vc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        path = vc_dir / f"{name}_annots.json"
        if not path.exists():
            logger.warning("Missing %s — skipping", path)
            continue
        logger.info("Loading %s", path)
        raw = load_json(path)
        splits[name] = aggregate_records(raw)
        logger.info("  %s: %d unique images", name, len(splits[name]))

    if not splits:
        raise RuntimeError(
            f"No VisualCOMET annotation files found in {vc_dir}. "
            "Run scripts/download_visualcomet.py first."
        )

    commonsense = merge_splits(*splits.values())
    save_json(commonsense, output_dir / "commonsense.json")

    # Summary stats
    total = len(commonsense)
    has_intent = sum(1 for v in commonsense.values() if v["intent"])
    has_before = sum(1 for v in commonsense.values() if v["before"])
    has_after = sum(1 for v in commonsense.values() if v["after"])

    print("\n" + "=" * 60)
    print("  SceneIQ — VisualCOMET Index Summary")
    print("=" * 60)
    print(f"  Images:               {total}")
    print(f"  With intent:          {has_intent}")
    print(f"  With before:          {has_before}")
    print(f"  With after:           {has_after}")
    print(f"  Output:               {output_dir / 'commonsense.json'}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VisualCOMET annotations into a per-image commonsense index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vc-dir", type=str, default=str(VISUALCOMET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(VISUALCOMET_INDEX_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="build_visualcomet_index.log")
    main(vc_dir=args.vc_dir, output_dir=args.output_dir)
