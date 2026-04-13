"""
SceneIQ — Assemble train/val/test splits.

Combines two sources into a single labelled dataset manifest:

* **Coherent (label=0)** — real Visual Genome photographs, sampled from
  ``image_data.json``. Images used as the *destination* scene for any
  synthetic incoherent sample are excluded to avoid train/label leakage.
* **Incoherent (label=1)** — the synthetic images produced by
  ``generate_synthetic.py`` (read from ``data/synthetic/metadata.json``).

The resulting manifest is split 70/15/15 (train/val/test) with a reproducible
seed, and written as three JSON files to ``data/processed/splits/``. Coherent
images are downloaded on demand into ``data/raw/visual_genome/images/`` so
every record in every split has a usable local path.

Downstream training code (``train.py``) just loads these JSONs and doesn't need
to know anything about VG's structure.
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
        {"image_id": e["image_id"], "url": e["url"], "label": 0}
        for e in chosen
    ]


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

    Args:
        synthetic_dir: Directory containing ``metadata.json`` and ``images/``.
        n_samples: Max incoherent records to keep (``<=0`` keeps all).
        rng: Random generator.

    Returns:
        Tuple of:
          * List of records ``{"image_path": str, "alien_object": str,
            "scene_image_id": int, "label": 1}``.
          * Set of scene image_ids used as destinations (for leakage exclusion).
    """
    meta_path = synthetic_dir / "metadata.json"
    metadata = load_json(meta_path)

    records: list = []
    scene_ids: set = set()
    for m in metadata:
        img_path = synthetic_dir / m["output_path"]
        if not img_path.exists():
            continue
        records.append({
            "image_path": str(img_path),
            "alien_object": m["alien_object"],
            "scene_image_id": m["scene_image_id"],
            "label": 1,
        })
        scene_ids.add(m["scene_image_id"])

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
) -> None:
    """Assemble and write train/val/test splits.

    Args:
        vg_dir: Directory with VG JSON files + ``images/`` cache.
        synthetic_dir: Directory with ``metadata.json`` + ``images/``.
        output_dir: Where to write the split JSONs.
        n_coherent: Target coherent count.
        n_incoherent: Target incoherent count (``<=0`` for all).
        train_frac: Fraction in train split.
        val_frac: Fraction in val split (test = 1 - train - val).
        seed: Random seed.
        skip_download: If True, don't try to download VG coherent images; use
            only those already cached.
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

    # Step 2 — Coherent sampling
    logger.info("=" * 60)
    logger.info("STEP 2/4 — Sampling coherent VG records")
    logger.info("=" * 60)
    coherent = sample_coherent_records(
        vg_dir / "image_data.json", excluded_ids, n_coherent, rng
    )

    # Step 3 — Fetch (or verify) coherent images
    logger.info("=" * 60)
    logger.info("STEP 3/4 — Ensuring coherent images are on disk")
    logger.info("=" * 60)
    cache_dir = vg_dir / "images"
    if skip_download:
        cache_dir.mkdir(parents=True, exist_ok=True)
        coherent = [
            {**r, "image_path": str(cache_dir / f"{r['image_id']}.jpg")}
            for r in coherent
            if (cache_dir / f"{r['image_id']}.jpg").exists()
        ]
        logger.info("Skip-download: %d coherent images available from cache", len(coherent))
    else:
        coherent = fetch_coherent_images(coherent, cache_dir)

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

    print("\n" + "=" * 60)
    print("  SceneIQ — Dataset Split Summary")
    print("=" * 60)
    print(f"  Coherent samples used:     {len(coherent)} (target {n_coherent})")
    print(f"  Incoherent samples used:   {len(incoherent)}")
    print(f"  Total:                     {len(all_records)}")
    print(f"  Train: {len(splits['train']):>5}  ({_class_balance(splits['train'])})")
    print(f"  Val:   {len(splits['val']):>5}  ({_class_balance(splits['val'])})")
    print(f"  Test:  {len(splits['test']):>5}  ({_class_balance(splits['test'])})")
    print(f"  Output directory:          {output_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Assemble SceneIQ train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vg-dir", type=str, default=str(VISUAL_GENOME_DIR),
        help="Directory with VG JSON files and image cache.",
    )
    parser.add_argument(
        "--synthetic-dir", type=str, default=str(SYNTHETIC_DIR),
        help="Directory with synthetic metadata.json and images/.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PROCESSED_DIR / "splits"),
        help="Where to write train.json, val.json, test.json.",
    )
    parser.add_argument(
        "--n-coherent", type=int, default=TARGET_COHERENT_SIZE,
        help="Target number of coherent samples.",
    )
    parser.add_argument(
        "--n-incoherent", type=int, default=TARGET_INCOHERENT_SIZE,
        help="Target number of incoherent samples (<=0 keeps all).",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.70,
        help="Fraction of records assigned to train.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Fraction assigned to val (test = 1 - train - val).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use only cached VG images; don't download missing ones.",
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
    )
