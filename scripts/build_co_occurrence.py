"""
SceneIQ — Build object co-occurrence and relationship statistics from Visual Genome.

Processes the Visual Genome ``objects.json`` and ``relationships.json`` files to
produce:

* ``object_counts``        — how often each object category appears across images
* ``pair_counts``          — how often each unordered object pair co-occurs in the
                             same image (useful for plausibility of co-presence)
* ``predicate_counts``     — global frequency of each relationship predicate
* ``triple_counts``        — frequency of (subject, predicate, object) triples

All tables are filtered by ``CO_OCCURRENCE_FREQ_THRESHOLD`` and saved as JSON
intermediates inside ``data/processed/co_occurrence/`` so later pipeline stages
can sample plausible vs. implausible object combinations without re-parsing VG.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CO_OCCURRENCE_FREQ_THRESHOLD,
    PROCESSED_DIR,
    VISUAL_GENOME_DIR,
)
from utils import extract_object_name, normalise_name, save_json, setup_logging, timer

logger = logging.getLogger("sceneiq")


# ── Object co-occurrence (from objects.json) ─────────────────────────────────

def build_object_cooccurrence(objects_path: Path) -> tuple[Counter, Counter]:
    """Count object frequencies and unordered object-pair co-occurrences.

    Args:
        objects_path: Path to Visual Genome ``objects.json``.

    Returns:
        Tuple of ``(object_counts, pair_counts)`` where ``pair_counts`` keys
        are ``"a|b"`` strings with ``a <= b`` lexicographically.
    """
    logger.info("Loading %s ...", objects_path.name)
    with open(objects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d image entries", len(data))

    object_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for entry in tqdm(data, desc="objects.json", unit="img"):
        names = {
            extract_object_name(o)
            for o in entry.get("objects", [])
        }
        names.discard("")
        # Per-image object presence
        for n in names:
            object_counts[n] += 1
        # Unordered pairs
        for a, b in combinations(sorted(names), 2):
            pair_counts[f"{a}|{b}"] += 1

    logger.info(
        "Objects: %d unique categories, %d unique co-occurring pairs",
        len(object_counts), len(pair_counts),
    )
    return object_counts, pair_counts


# ── Relationship triples (from relationships.json) ───────────────────────────

def build_relationship_stats(relationships_path: Path) -> tuple[Counter, Counter]:
    """Count predicate and (subject, predicate, object) triple frequencies.

    Args:
        relationships_path: Path to Visual Genome ``relationships.json``.

    Returns:
        Tuple of ``(predicate_counts, triple_counts)``. Triple keys are
        ``"subject|predicate|object"`` strings.
    """
    logger.info("Loading %s ...", relationships_path.name)
    with open(relationships_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d image entries", len(data))

    predicate_counts: Counter = Counter()
    triple_counts: Counter = Counter()

    for entry in tqdm(data, desc="relationships.json", unit="img"):
        for rel in entry.get("relationships", []):
            predicate = normalise_name(rel.get("predicate", ""))
            subj = extract_object_name(rel.get("subject", {}))
            obj = extract_object_name(rel.get("object", {}))
            if not (predicate and subj and obj):
                continue
            predicate_counts[predicate] += 1
            triple_counts[f"{subj}|{predicate}|{obj}"] += 1

    logger.info(
        "Relationships: %d unique predicates, %d unique triples",
        len(predicate_counts), len(triple_counts),
    )
    return predicate_counts, triple_counts


# ── Filtering & saving ───────────────────────────────────────────────────────

def filter_by_threshold(counter: Counter, threshold: int) -> dict:
    """Return a dict of entries whose count meets the threshold, sorted desc.

    Args:
        counter: Counter of string → count.
        threshold: Minimum count to retain.

    Returns:
        Ordered dict of ``{key: count}`` with ``count >= threshold``.
    """
    kept = {k: c for k, c in counter.most_common() if c >= threshold}
    return kept


def save_counter(counter: Counter, out_path: Path, threshold: int) -> dict:
    """Filter a Counter and persist it as JSON.

    Args:
        counter: Counter to save.
        out_path: Destination JSON path.
        threshold: Minimum count for inclusion.

    Returns:
        The filtered dict that was written.
    """
    filtered = filter_by_threshold(counter, threshold)
    save_json(filtered, out_path)
    logger.info(
        "%s — kept %d / %d entries (threshold=%d)",
        out_path.name, len(filtered), len(counter), threshold,
    )
    return filtered


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(vg_dir: Path, output_dir: Path, threshold: int) -> None:
    """Build co-occurrence and relationship statistics from Visual Genome.

    Args:
        vg_dir: Directory containing extracted VG JSON files.
        output_dir: Directory to write processed statistics into.
        threshold: Minimum frequency for an entry to be retained.
    """
    vg_dir = Path(vg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("VG input dir: %s", vg_dir)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Threshold:    %d", threshold)

    objects_path = vg_dir / "objects.json"
    relationships_path = vg_dir / "relationships.json"

    missing = [p for p in (objects_path, relationships_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required VG files missing: {missing}. Run scripts/download_vg.py first."
        )

    # Step 1 — Object co-occurrence
    logger.info("=" * 60)
    logger.info("STEP 1/2 — Object co-occurrence from objects.json")
    logger.info("=" * 60)
    object_counts, pair_counts = build_object_cooccurrence(objects_path)
    filt_objects = save_counter(object_counts, output_dir / "object_counts.json", threshold)
    filt_pairs = save_counter(pair_counts, output_dir / "pair_counts.json", threshold)

    # Step 2 — Relationship triples
    logger.info("=" * 60)
    logger.info("STEP 2/2 — Predicate & triple stats from relationships.json")
    logger.info("=" * 60)
    predicate_counts, triple_counts = build_relationship_stats(relationships_path)
    filt_predicates = save_counter(
        predicate_counts, output_dir / "predicate_counts.json", threshold
    )
    filt_triples = save_counter(
        triple_counts, output_dir / "triple_counts.json", threshold
    )

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    def _top(d: dict, k: int = 5) -> list:
        return list(d.items())[:k]

    print("\n" + "=" * 60)
    print("  Visual Genome — Co-occurrence Summary")
    print("=" * 60)
    print(f"  Threshold:                 {threshold}")
    print(f"  Object categories kept:    {len(filt_objects):,} / {len(object_counts):,}")
    print(f"  Object pairs kept:         {len(filt_pairs):,} / {len(pair_counts):,}")
    print(f"  Predicates kept:           {len(filt_predicates):,} / {len(predicate_counts):,}")
    print(f"  Triples kept:              {len(filt_triples):,} / {len(triple_counts):,}")
    print(f"  Output directory:          {output_dir}")
    print("-" * 60)
    print(f"  Top 5 objects:   {_top(filt_objects)}")
    print(f"  Top 5 pairs:     {_top(filt_pairs)}")
    print(f"  Top 5 predicates:{_top(filt_predicates)}")
    print(f"  Top 5 triples:   {_top(filt_triples)}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build object co-occurrence and relationship stats from Visual Genome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vg-dir",
        type=str,
        default=str(VISUAL_GENOME_DIR),
        help="Directory containing extracted Visual Genome JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DIR / "co_occurrence"),
        help="Directory to write processed co-occurrence JSON files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CO_OCCURRENCE_FREQ_THRESHOLD,
        help="Minimum frequency required to retain an entry.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="build_co_occurrence.log")
    main(
        vg_dir=args.vg_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
