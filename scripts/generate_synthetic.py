"""
SceneIQ — Generate synthetic incoherent images.

Takes a plausible Visual Genome scene, identifies an object category that has
*never* co-occurred with any of the scene's existing objects (according to the
``pair_counts.json`` table produced by ``build_co_occurrence.py``), crops an
instance of that "alien" object from a different VG image, and pastes it into
the scene. The resulting image is labelled **incoherent** and saved to
``data/synthetic/``.

Outputs:
    data/synthetic/images/<sample_id>.jpg   — the generated incoherent images
    data/synthetic/metadata.json            — one record per sample with the
                                              source image, the alien object,
                                              its origin image, and where on
                                              the destination it was pasted.

This is a deliberately simple first-pass generator: random placement, no
semantic blending. It's enough to bootstrap the "incoherent" class for the
classifier; higher-fidelity synthesis is a future step.
"""

import argparse
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    SYNTHETIC_DIR,
    VISUAL_GENOME_DIR,
)
from utils import (
    extract_object_name,
    load_json,
    save_json,
    set_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("sceneiq")


# ── Loading & indexing ───────────────────────────────────────────────────────

def load_image_index(image_data_path: Path) -> dict:
    """Return ``{image_id: {"url": ..., "width": ..., "height": ...}}``.

    Args:
        image_data_path: Path to VG ``image_data.json``.

    Returns:
        Dictionary mapping VG image_id to its URL and dimensions.
    """
    data = load_json(image_data_path)
    index = {
        entry["image_id"]: {
            "url": entry.get("url", ""),
            "width": entry.get("width", 0),
            "height": entry.get("height", 0),
        }
        for entry in data
    }
    logger.info("Image index: %d images", len(index))
    return index


def load_objects_index(objects_path: Path) -> tuple[dict, dict]:
    """Build two indices over VG ``objects.json``.

    Args:
        objects_path: Path to VG ``objects.json``.

    Returns:
        Tuple of:
          * ``image_to_objects``: image_id → set of object names present.
          * ``object_to_crops``: object_name → list of ``(image_id, bbox)``
            where ``bbox = (x, y, w, h)``. Used to source a crop of an "alien"
            object from a different image.
    """
    data = load_json(objects_path)

    image_to_objects: dict = {}
    object_to_crops: dict = defaultdict(list)

    for entry in data:
        image_id = entry["image_id"]
        names_in_image: set = set()
        for obj in entry.get("objects", []):
            name = extract_object_name(obj)
            if not name:
                continue
            names_in_image.add(name)
            bbox = (obj.get("x", 0), obj.get("y", 0),
                    obj.get("w", 0), obj.get("h", 0))
            if bbox[2] > 0 and bbox[3] > 0:
                object_to_crops[name].append((image_id, bbox))
        image_to_objects[image_id] = names_in_image

    logger.info(
        "Indexed %d images, %d distinct object categories with usable crops",
        len(image_to_objects), len(object_to_crops),
    )
    return image_to_objects, dict(object_to_crops)


# ── Implausibility search ────────────────────────────────────────────────────

def is_implausible(alien: str, scene_objects: set, plausible_pairs: set) -> bool:
    """Return True iff ``alien`` never co-occurred with any object in the scene.

    Args:
        alien: Candidate alien object category.
        scene_objects: Object categories already present in the destination scene.
        plausible_pairs: Set of ``"a|b"`` strings (``a <= b``) from ``pair_counts.json``.

    Returns:
        True if no pair ``(alien, scene_object)`` appears in ``plausible_pairs``.
    """
    if alien in scene_objects:
        return False
    for obj in scene_objects:
        a, b = sorted([alien, obj])
        if f"{a}|{b}" in plausible_pairs:
            return False
    return True


def pick_alien_object(
    scene_objects: set,
    candidates: list,
    plausible_pairs: set,
    rng: random.Random,
    max_tries: int = 50,
) -> str | None:
    """Randomly sample object categories until one is implausible with the scene.

    Args:
        scene_objects: Objects already in the destination scene.
        candidates: Pre-filtered list of alien category names to sample from.
        plausible_pairs: Known plausible pairs from co-occurrence table.
        rng: Random generator.
        max_tries: Give up after this many sampling attempts.

    Returns:
        Chosen alien object name, or ``None`` if no match was found.
    """
    for _ in range(max_tries):
        alien = rng.choice(candidates)
        if is_implausible(alien, scene_objects, plausible_pairs):
            return alien
    return None


# ── Image fetching & compositing ─────────────────────────────────────────────

def fetch_image(image_id: int, url: str, cache_dir: Path) -> Image.Image | None:
    """Download a VG image (or load from cache) and return it as a PIL Image.

    Args:
        image_id: VG image_id (used as cache filename).
        url: Source URL from VG ``image_data.json``.
        cache_dir: Local directory used to cache downloads.

    Returns:
        PIL Image in RGB mode, or ``None`` on failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{image_id}.jpg"
    if not cached.exists():
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            cached.write_bytes(resp.content)
        except Exception as exc:
            logger.warning("Failed to fetch image %d (%s): %s", image_id, url, exc)
            return None
    try:
        return Image.open(cached).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to open cached image %s: %s", cached, exc)
        return None


def paste_alien(
    scene: Image.Image,
    alien_source: Image.Image,
    alien_bbox: tuple,
    rng: random.Random,
    min_crop_size: int = 32,
) -> tuple[Image.Image, tuple] | None:
    """Crop the alien object and paste it at a random location in the scene.

    Args:
        scene: Destination (coherent) image.
        alien_source: Source image containing the alien object.
        alien_bbox: ``(x, y, w, h)`` of the alien object in ``alien_source``.
        rng: Random generator for paste location.
        min_crop_size: Reject crops whose post-resize min side is below this.

    Returns:
        Tuple of ``(composited_image, paste_bbox)`` where ``paste_bbox`` is the
        destination ``(x, y, w, h)``. Returns ``None`` if the crop is too small
        to form a useful paste (degenerate bbox or sub-threshold size).
    """
    x, y, w, h = alien_bbox
    x2, y2 = x + w, y + h
    x = max(0, x); y = max(0, y)
    x2 = min(alien_source.width, x2); y2 = min(alien_source.height, y2)
    if x2 - x < min_crop_size or y2 - y < min_crop_size:
        return None
    crop = alien_source.crop((x, y, x2, y2))

    target = min(scene.width, scene.height) // 3
    if max(crop.width, crop.height) > target and max(crop.width, crop.height) > 0:
        scale = target / max(crop.width, crop.height)
        new_size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
        crop = crop.resize(new_size, Image.LANCZOS)

    if min(crop.width, crop.height) < min_crop_size:
        return None

    max_x = max(0, scene.width - crop.width)
    max_y = max(0, scene.height - crop.height)
    px = rng.randint(0, max_x)
    py = rng.randint(0, max_y)

    composite = scene.copy()
    composite.paste(crop, (px, py))
    return composite, (px, py, crop.width, crop.height)


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    vg_dir: Path,
    co_occurrence_dir: Path,
    output_dir: Path,
    n_samples: int,
    seed: int,
    min_alien_count: int,
    min_crop_size: int,
) -> None:
    """Generate ``n_samples`` synthetic incoherent images.

    Args:
        vg_dir: Directory with extracted VG JSON files.
        co_occurrence_dir: Directory containing ``pair_counts.json``.
        output_dir: Where to write generated images + metadata.
        n_samples: How many synthetic samples to create.
        seed: Random seed for reproducibility.
    """
    vg_dir = Path(vg_dir)
    output_dir = Path(output_dir)
    co_occurrence_dir = Path(co_occurrence_dir)
    image_out_dir = output_dir / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = vg_dir / "images"

    rng = random.Random(seed)
    set_seed(seed)

    # Load inputs
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading indices")
    logger.info("=" * 60)
    image_index = load_image_index(vg_dir / "image_data.json")
    image_to_objects, object_to_crops = load_objects_index(vg_dir / "objects.json")
    pair_counts = load_json(co_occurrence_dir / "pair_counts.json")
    plausible_pairs = set(pair_counts.keys())
    logger.info("Plausible pairs: %d", len(plausible_pairs))

    # Filter alien candidates: only categories frequent enough to be "real"
    # (VG free-text annotations pollute the long tail — "there is a man",
    # "bed sentence", etc.).
    object_counts = load_json(co_occurrence_dir / "object_counts.json")
    alien_candidates = [
        name for name in object_to_crops
        if object_counts.get(name, 0) >= min_alien_count
    ]
    logger.info(
        "Alien candidates after count filter (>=%d): %d / %d",
        min_alien_count, len(alien_candidates), len(object_to_crops),
    )
    if not alien_candidates:
        raise RuntimeError("No alien candidates survived the count filter.")

    # Need ≥2 objects so the implausibility check isn't trivially satisfied.
    candidate_scene_ids = [
        iid for iid, objs in image_to_objects.items()
        if len(objs) >= 2 and iid in image_index
    ]
    logger.info("Candidate scenes (≥2 objects): %d", len(candidate_scene_ids))
    rng.shuffle(candidate_scene_ids)

    # Generate
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Generating %d samples", n_samples)
    logger.info("=" * 60)
    metadata: list = []
    attempts = 0
    max_attempts = n_samples * 20

    pbar = tqdm(total=n_samples, desc="synthetic", unit="img")
    for scene_id in candidate_scene_ids:
        if len(metadata) >= n_samples or attempts >= max_attempts:
            break
        attempts += 1

        scene_objects = image_to_objects[scene_id]
        alien = pick_alien_object(scene_objects, alien_candidates, plausible_pairs, rng)
        if alien is None:
            continue

        alien_src_id, alien_bbox = rng.choice(object_to_crops[alien])
        if alien_src_id == scene_id or alien_src_id not in image_index:
            continue
        if alien_bbox[2] < min_crop_size or alien_bbox[3] < min_crop_size:
            continue

        scene_img = fetch_image(scene_id, image_index[scene_id]["url"], image_cache_dir)
        if scene_img is None:
            continue
        alien_img = fetch_image(
            alien_src_id, image_index[alien_src_id]["url"], image_cache_dir
        )
        if alien_img is None:
            continue

        paste_result = paste_alien(scene_img, alien_img, alien_bbox, rng, min_crop_size)
        if paste_result is None:
            continue
        composite, paste_bbox = paste_result

        alien_slug = re.sub(r"[^a-z0-9]+", "_", alien).strip("_") or "alien"
        sample_id = f"vg{scene_id}_alien_{alien_slug}_{len(metadata):05d}"
        out_path = image_out_dir / f"{sample_id}.jpg"
        composite.save(out_path, "JPEG", quality=92)

        metadata.append({
            "sample_id": sample_id,
            "output_path": str(out_path.relative_to(output_dir)),
            "scene_image_id": scene_id,
            "scene_objects": sorted(scene_objects),
            "alien_object": alien,
            "alien_source_image_id": alien_src_id,
            "alien_source_bbox": list(alien_bbox),
            "paste_bbox": list(paste_bbox),
        })
        pbar.update(1)

        if len(metadata) % 25 == 0:
            save_json(metadata, output_dir / "metadata.json")
    pbar.close()

    logger.info("=" * 60)
    logger.info("STEP 3/3 — Saving metadata")
    logger.info("=" * 60)
    save_json(metadata, output_dir / "metadata.json")

    print("\n" + "=" * 60)
    print("  SceneIQ — Synthetic Incoherent Image Generation")
    print("=" * 60)
    print(f"  Requested samples:        {n_samples}")
    print(f"  Generated samples:        {len(metadata)}")
    print(f"  Attempts made:            {attempts}")
    print(f"  Output images:            {image_out_dir}")
    print(f"  Metadata:                 {output_dir / 'metadata.json'}")
    if metadata:
        top_aliens = Counter(m["alien_object"] for m in metadata).most_common(5)
        print(f"  Top alien objects:        {top_aliens}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic incoherent images for SceneIQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vg-dir",
        type=str,
        default=str(VISUAL_GENOME_DIR),
        help="Directory containing extracted Visual Genome JSON files.",
    )
    parser.add_argument(
        "--co-occurrence-dir",
        type=str,
        default=str(PROCESSED_DIR / "co_occurrence"),
        help="Directory containing pair_counts.json from build_co_occurrence.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SYNTHETIC_DIR),
        help="Where to write generated images and metadata.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of synthetic incoherent images to generate (start small).",
    )
    parser.add_argument(
        "--min-alien-count",
        type=int,
        default=50,
        help="Drop alien categories appearing in fewer than this many VG images "
             "(filters out noisy free-text labels).",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=32,
        help="Minimum crop side (px) required before and after resize; smaller "
             "crops are rejected to avoid invisible pastes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="generate_synthetic.log")
    main(
        vg_dir=args.vg_dir,
        co_occurrence_dir=args.co_occurrence_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        min_alien_count=args.min_alien_count,
        min_crop_size=args.min_crop_size,
    )
