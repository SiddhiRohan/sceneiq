"""
SceneIQ — Build a COCO coherent-image index.

Takes COCO's ``instances_{split}.json`` + ``captions_{split}.json`` and
flattens them into a single per-image record list that
``prepare_dataset.py`` can consume as a coherent-image pool:

    [
      {"image_id": int, "url": str, "width": int, "height": int,
       "object_categories": [str, ...],
       "captions": [str, ...],
       "source": "coco"},
      ...
    ]

We also emit a COCO-side co-occurrence table (``coco_pair_counts.json``) so
the synthetic generator can optionally mine implausible pairs from COCO too —
COCO's categorical taxonomy is cleaner than Visual Genome's free-text labels,
so pairs mined here are less noisy.

Outputs:
    data/processed/mscoco/
      ├── coherent_records_<split>.json
      └── coco_pair_counts_<split>.json
"""

import argparse
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MSCOCO_DEFAULT_SPLIT, MSCOCO_DIR, MSCOCO_INDEX_DIR
from utils import load_json, save_json, setup_logging, timer

logger = logging.getLogger("sceneiq")


# ── COCO -> per-image records ────────────────────────────────────────────────

def build_image_index(images: list) -> dict:
    """Return ``{image_id: {"url", "width", "height", "file_name"}}``."""
    out: dict = {}
    for entry in images:
        out[entry["id"]] = {
            "url": entry.get("coco_url", entry.get("flickr_url", "")),
            "width": int(entry.get("width", 0)),
            "height": int(entry.get("height", 0)),
            "file_name": entry.get("file_name", ""),
        }
    return out


def build_category_index(categories: list) -> dict:
    """Return ``{category_id: name}``."""
    return {c["id"]: c["name"] for c in categories}


def objects_per_image(annotations: list, cat_name_for_id: dict) -> dict:
    """Return ``{image_id: sorted set of category names}``."""
    by_image: dict = defaultdict(set)
    for ann in annotations:
        iid = ann.get("image_id")
        cid = ann.get("category_id")
        if iid is None or cid is None:
            continue
        name = cat_name_for_id.get(cid)
        if name:
            by_image[iid].add(name)
    return {iid: sorted(cats) for iid, cats in by_image.items()}


def captions_per_image(caption_entries: list) -> dict:
    """Return ``{image_id: [caption_text, ...]}``."""
    by_image: dict = defaultdict(list)
    for entry in caption_entries:
        iid = entry.get("image_id")
        cap = entry.get("caption", "").strip()
        if iid is not None and cap:
            by_image[iid].append(cap)
    return dict(by_image)


def build_pair_counts(per_image_categories: dict) -> dict:
    """Symmetric object co-occurrence counts keyed as ``"a|b"`` with a <= b."""
    counts: dict = defaultdict(int)
    for cats in per_image_categories.values():
        for a, b in combinations(sorted(cats), 2):
            counts[f"{a}|{b}"] += 1
    return dict(counts)


def assemble_records(
    image_index: dict,
    per_image_objects: dict,
    per_image_captions: dict,
) -> list:
    """Flatten everything into ``prepare_dataset``-compatible coherent records."""
    records: list = []
    for iid, info in image_index.items():
        records.append({
            "image_id": int(iid),
            "scene_image_id": int(iid),        # mirror for a uniform key
            "url": info.get("url", ""),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "object_categories": per_image_objects.get(iid, []),
            "captions": per_image_captions.get(iid, []),
            "source": "coco",
            "label": 0,
        })
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(coco_dir: Path, split: str, output_dir: Path) -> None:
    """Build the COCO coherent-image index for ``split``."""
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inst_path = coco_dir / "annotations" / f"instances_{split}.json"
    cap_path = coco_dir / "annotations" / f"captions_{split}.json"
    if not inst_path.exists():
        raise FileNotFoundError(f"{inst_path} not found — run scripts/download_coco.py first.")

    logger.info("=" * 60)
    logger.info("STEP 1/4 — Loading COCO annotations (%s)", split)
    logger.info("=" * 60)
    instances = load_json(inst_path)
    captions_blob = load_json(cap_path) if cap_path.exists() else {"annotations": []}

    logger.info("=" * 60)
    logger.info("STEP 2/4 — Indexing categories / objects / captions")
    logger.info("=" * 60)
    cat_name_for_id = build_category_index(instances["categories"])
    per_image_objects = objects_per_image(instances["annotations"], cat_name_for_id)
    per_image_captions = captions_per_image(captions_blob.get("annotations", []))

    image_index = build_image_index(instances["images"])
    logger.info("Images: %d  |  Categories: %d  |  With ≥1 object: %d",
                len(image_index), len(cat_name_for_id), len(per_image_objects))

    logger.info("=" * 60)
    logger.info("STEP 3/4 — Assembling coherent records")
    logger.info("=" * 60)
    records = assemble_records(image_index, per_image_objects, per_image_captions)
    save_json(records, output_dir / f"coherent_records_{split}.json")

    logger.info("=" * 60)
    logger.info("STEP 4/4 — COCO co-occurrence stats")
    logger.info("=" * 60)
    pair_counts = build_pair_counts(per_image_objects)
    save_json(pair_counts, output_dir / f"coco_pair_counts_{split}.json")

    print("\n" + "=" * 60)
    print(f"  SceneIQ — MS-COCO Index Summary ({split})")
    print("=" * 60)
    print(f"  Records:             {len(records)}")
    print(f"  Pair counts:         {len(pair_counts)}")
    print(f"  Output dir:          {output_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the COCO coherent-image index for SceneIQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coco-dir", type=str, default=str(MSCOCO_DIR))
    parser.add_argument("--split", type=str, default=MSCOCO_DEFAULT_SPLIT,
                        help="Which COCO split to index (val2017 / train2017).")
    parser.add_argument("--output-dir", type=str, default=str(MSCOCO_INDEX_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="build_coco_index.log")
    main(coco_dir=args.coco_dir, split=args.split, output_dir=args.output_dir)
