"""
SceneIQ — Assemble train/val/test splits.

Combines two sources into a single labelled dataset manifest:

* **Coherent (label=0)** — real Visual Genome photographs, sampled from
  ``image_data.json``. Images used as the *destination* scene for any
  synthetic incoherent sample are excluded to avoid train/label leakage.
* **Incoherent (label=1)** — the synthetic images produced by
  ``generate_synthetic.py`` (read from ``data/synthetic/metadata.json``).

Incoherent records additionally carry ``paste_bbox`` (the destination
``(x, y, w, h)`` where the alien object was pasted) and
``scene_image_id`` so downstream localization supervision and the
scene-graph branch have what they need.

The resulting manifest is split 70/15/15 (train/val/test) with a reproducible
seed, and written as three JSON files to ``data/processed/splits/``. Coherent
images are downloaded on demand into ``data/raw/visual_genome/images/`` so
every record in every split has a usable local path.

Downstream training code (``train.py``) just loads these JSONs.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MSCOCO_DEFAULT_SPLIT,
    MSCOCO_INDEX_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SYNTHETIC_DIR,
    TARGET_COHERENT_SIZE,
    TARGET_INCOHERENT_SIZE,
    VISUAL_GENOME_DIR,
)
from utils import load_json, save_json, set_seed, setup_logging, timer

logger = logging.getLogger("sceneiq")


# ── Coherent sampling ────────────────────────────────────────────────────────

def sample_coherent_records(
    image_data_path: Path,
    excluded_ids: set,
    n_samples: int,
    rng: random.Random,
) -> list:
    """Sample ``n_samples`` VG images (excluding leaked ids) as coherent records.

    Args:
        image_data_path: Path to VG ``image_data.json``.
        excluded_ids: VG image_ids that must not appear (scene ids used in
            synthetic generation).
        n_samples: How many coherent samples to draw.
        rng: Random generator.

    Returns:
        List of records ``{"image_id": int, "url": str, "label": 0}``. Fewer
        than ``n_samples`` may be returned if VG has fewer eligible images.
    """
    data = load_json(image_data_path)
    eligible = [e for e in data if e["image_id"] not in excluded_ids and e.get("url")]
    logger.info(
        "Coherent pool: %d eligible / %d total VG images (%d excluded for leakage)",
        len(eligible), len(data), len(excluded_ids),
    )
    rng.shuffle(eligible)
    chosen = eligible[:n_samples]
    return [
        {
            "image_id": e["image_id"],
            "scene_image_id": e["image_id"],   # mirror so the GAT branch has one field to key on
            "url": e["url"],
            "label": 0,
        }
        for e in chosen
    ]


def load_coco_records(coco_index_path: Path, n_samples: int, rng: random.Random) -> list:
    """Load COCO coherent records and subsample to ``n_samples``.

    COCO records already follow the coherent-record schema
    (``image_id`` / ``url`` / ``label=0``) so they plug into the same pipeline
    as VG samples without transformation.

    Args:
        coco_index_path: Path to ``coherent_records_<split>.json`` produced by
            ``scripts/build_coco_index.py``.
        n_samples: Max number of COCO records to draw (<=0 keeps all).
        rng: Random generator.

    Returns:
        List of COCO coherent records (each with ``"source": "coco"``).
    """
    if not coco_index_path.exists():
        logger.warning("COCO index not found at %s — skipping COCO coherent pool.",
                       coco_index_path)
        return []
    data = load_json(coco_index_path)
    rng.shuffle(data)
    if n_samples > 0:
        data = data[:n_samples]
    for rec in data:
        rec.setdefault("label", 0)
        rec.setdefault("source", "coco")
    logger.info("COCO coherent records loaded: %d", len(data))
    return data


def fetch_coherent_images(records: list, cache_dir: Path) -> list:
    """Download (or verify cached) coherent images and attach a local path.

    Args:
        records: Coherent records from ``sample_coherent_records``.
        cache_dir: Local directory used as download cache.

    Returns:
        Subset of ``records`` whose images are available locally, each with an
        added ``"image_path"`` field (absolute path as a string).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok: list = []
    for rec in tqdm(records, desc="fetch coherent", unit="img"):
        dest = cache_dir / f"{rec['image_id']}.jpg"
        if not dest.exists():
            try:
                resp = requests.get(rec["url"], timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            except Exception as exc:
                logger.warning("Skipping image %d: %s", rec["image_id"], exc)
                continue
        rec["image_path"] = str(dest)
        ok.append(rec)
    logger.info("Coherent images usable: %d / %d", len(ok), len(records))
    return ok


# ── Incoherent loading ───────────────────────────────────────────────────────

def load_incoherent_records(
    synthetic_dir: Path,
    n_samples: int,
    rng: random.Random,
) -> tuple[list, set]:
    """Load synthetic metadata and turn it into labelled records.

    Propagates the ``paste_bbox`` written by ``generate_synthetic.py`` so that
    downstream training has pixel-space ground truth for localization.

    Args:
        synthetic_dir: Directory containing ``metadata.json`` and ``images/``.
        n_samples: Max incoherent records to keep (``<=0`` keeps all).
        rng: Random generator.

    Returns:
        Tuple of:
          * List of records ``{"image_path", "alien_object", "scene_image_id",
            "paste_bbox", "alien_source_image_id", "label": 1}``.
          * Set of scene image_ids used as destinations (for leakage exclusion).
    """
    meta_path = synthetic_dir / "metadata.json"
    metadata = load_json(meta_path)

    records: list = []
    scene_ids: set = set()
    missing_bbox = 0
    for m in metadata:
        img_path = synthetic_dir / m["output_path"]
        if not img_path.exists():
            continue
        paste_bbox = m.get("paste_bbox")
        if paste_bbox is None:
            missing_bbox += 1
            continue
        records.append({
            "image_path": str(img_path),
            "alien_object": m["alien_object"],
            "scene_image_id": m["scene_image_id"],
            "alien_source_image_id": m.get("alien_source_image_id"),
            "paste_bbox": list(paste_bbox),     # [x, y, w, h] in pixel coords of the composited image
            "label": 1,
        })
        scene_ids.add(m["scene_image_id"])

    if missing_bbox:
        logger.warning(
            "Skipped %d synthetic records without paste_bbox. "
            "Re-run generate_synthetic.py so paste_bbox is recorded.",
            missing_bbox,
        )

    rng.shuffle(records)
    if n_samples > 0:
        records = records[:n_samples]
    logger.info("Incoherent records loaded: %d (from %d in metadata)",
                len(records), len(metadata))
    return records, scene_ids


# ── Splitting ────────────────────────────────────────────────────────────────

def split_records(
    records: list,
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> dict:
    """Split records into train/val/test by label-stratified fractions.

    Args:
        records: Full labelled record list.
        train_frac: Fraction assigned to train (e.g. 0.70).
        val_frac: Fraction assigned to val (e.g. 0.15). Test = remainder.
        rng: Random generator.

    Returns:
        Dict ``{"train": [...], "val": [...], "test": [...]}``.
    """
    by_label: dict = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    splits = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])
        logger.info(
            "Label=%d split: train=%d val=%d test=%d",
            label, n_train, n_val, n - n_train - n_val,
        )

    for split in splits.values():
        rng.shuffle(split)
    return splits


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    vg_dir: Path,
    synthetic_dir: Path,
    output_dir: Path,
    n_coherent: int,
    n_incoherent: int,
    train_frac: float,
    val_frac: float,
    seed: int,
    skip_download: bool,
    coco_index_path: Path | None = None,
    coco_fraction: float = 0.0,
) -> None:
    """Assemble and write train/val/test splits.

    If ``coco_index_path`` is given, a fraction ``coco_fraction`` of the
    ``n_coherent`` target comes from MS-COCO instead of Visual Genome — this
    diversifies the coherent pool with cleaner category-labelled photos.
    """
    vg_dir = Path(vg_dir)
    synthetic_dir = Path(synthetic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    set_seed(seed)

    # Step 1 — Incoherent (also tells us which scenes to exclude from coherent)
    logger.info("=" * 60)
    logger.info("STEP 1/4 — Loading incoherent synthetic records")
    logger.info("=" * 60)
    incoherent, excluded_ids = load_incoherent_records(synthetic_dir, n_incoherent, rng)
    if not incoherent:
        raise RuntimeError(
            f"No incoherent records found in {synthetic_dir / 'metadata.json'}. "
            "Run scripts/generate_synthetic.py first."
        )

    # Step 2 — Coherent sampling (VG + optional COCO)
    logger.info("=" * 60)
    logger.info("STEP 2/4 — Sampling coherent records")
    logger.info("=" * 60)
    n_coco = int(round(n_coherent * max(0.0, min(1.0, coco_fraction))))
    n_vg = n_coherent - n_coco

    coco_records: list = []
    if coco_index_path is not None and n_coco > 0:
        coco_records = load_coco_records(Path(coco_index_path), n_coco, rng)

    coherent = sample_coherent_records(
        vg_dir / "image_data.json", excluded_ids, n_vg, rng
    )
    logger.info("Coherent composition: VG=%d  COCO=%d  (target=%d)",
                len(coherent), len(coco_records), n_coherent)

    # Step 3 — Fetch (or verify) coherent images
    logger.info("=" * 60)
    logger.info("STEP 3/4 — Ensuring coherent images are on disk")
    logger.info("=" * 60)
    cache_dir = vg_dir / "images"
    coco_cache_dir = Path(vg_dir).parent / "mscoco" / "images"
    if skip_download:
        cache_dir.mkdir(parents=True, exist_ok=True)
        coco_cache_dir.mkdir(parents=True, exist_ok=True)
        coherent = [
            {**r, "image_path": str(cache_dir / f"{r['image_id']}.jpg")}
            for r in coherent
            if (cache_dir / f"{r['image_id']}.jpg").exists()
        ]
        coco_records = [
            {**r, "image_path": str(coco_cache_dir / f"{r['image_id']}.jpg")}
            for r in coco_records
            if (coco_cache_dir / f"{r['image_id']}.jpg").exists()
        ]
        logger.info("Skip-download: %d VG + %d COCO coherent images cached",
                    len(coherent), len(coco_records))
    else:
        coherent = fetch_coherent_images(coherent, cache_dir)
        if coco_records:
            coco_records = fetch_coherent_images(coco_records, coco_cache_dir)

    coherent = coherent + coco_records
    if not coherent:
        raise RuntimeError("No coherent images available on disk — aborting.")

    # Step 4 — Split
    logger.info("=" * 60)
    logger.info("STEP 4/4 — Splitting into train/val/test")
    logger.info("=" * 60)
    all_records = coherent + incoherent
    rng.shuffle(all_records)
    splits = split_records(all_records, train_frac, val_frac, rng)

    for name, recs in splits.items():
        save_json(recs, output_dir / f"{name}.json")

    # Summary
    def _class_balance(recs: list) -> str:
        n0 = sum(1 for r in recs if r["label"] == 0)
        n1 = sum(1 for r in recs if r["label"] == 1)
        return f"coherent={n0}, incoherent={n1}"

    with_bbox = sum(1 for r in incoherent if r.get("paste_bbox"))
    print("\n" + "=" * 60)
    print("  SceneIQ — Dataset Split Summary")
    print("=" * 60)
    print(f"  Coherent samples used:     {len(coherent)} (target {n_coherent})")
    print(f"  Incoherent samples used:   {len(incoherent)} (with paste_bbox: {with_bbox})")
    print(f"  Total:                     {len(all_records)}")
    print(f"  Train: {len(splits['train']):>5}  ({_class_balance(splits['train'])})")
    print(f"  Val:   {len(splits['val']):>5}  ({_class_balance(splits['val'])})")
    print(f"  Test:  {len(splits['test']):>5}  ({_class_balance(splits['test'])})")
    print(f"  Output directory:          {output_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble SceneIQ train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vg-dir", type=str, default=str(VISUAL_GENOME_DIR))
    parser.add_argument("--synthetic-dir", type=str, default=str(SYNTHETIC_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--n-coherent", type=int, default=TARGET_COHERENT_SIZE)
    parser.add_argument("--n-incoherent", type=int, default=TARGET_INCOHERENT_SIZE)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--coco-index",
        type=str,
        default=None,
        help="Path to coherent_records_<split>.json from build_coco_index.py. "
             "When supplied, a fraction of the coherent pool is drawn from COCO.",
    )
    parser.add_argument(
        "--coco-fraction",
        type=float,
        default=0.0,
        help="Fraction of n_coherent to draw from COCO (0.0 = VG only, "
             "1.0 = COCO only). Ignored unless --coco-index is set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="prepare_dataset.log")
    main(
        vg_dir=args.vg_dir,
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        n_coherent=args.n_coherent,
        n_incoherent=args.n_incoherent,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        skip_download=args.skip_download,
        coco_index_path=args.coco_index,
        coco_fraction=args.coco_fraction,
    )
