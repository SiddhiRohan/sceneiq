"""
SceneIQ — Download the Visual Genome dataset.

Downloads objects, relationships, image metadata, and scene graphs from the
official Visual Genome API, unzips them, validates the JSON, and prints
summary statistics.
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VISUAL_GENOME_DIR
from utils import setup_logging, timer

# ── Constants ────────────────────────────────────────────────────────────────

VG_FILES = {
    "objects.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
    "relationships.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "image_data.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    "scene_graphs.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip",
}

logger = logging.getLogger("sceneiq")


# ── Download ─────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a file from a URL with a tqdm progress bar.

    Args:
        url: Source URL.
        dest: Destination file path.
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded file.
    """
    if dest.exists() and not force:
        logger.info("Skipping (already exists): %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(dest, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        disable=False,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# ── Unzip ────────────────────────────────────────────────────────────────────

def unzip_file(zip_path: Path, output_dir: Path) -> Path:
    """Extract a zip archive and return the path to the extracted JSON file.

    Args:
        zip_path: Path to the zip file.
        output_dir: Directory to extract into.

    Returns:
        Path to the extracted JSON file.
    """
    json_name = zip_path.stem  # e.g., "objects.json"
    json_path = output_dir / json_name

    if json_path.exists():
        logger.info("Skipping unzip (already extracted): %s", json_name)
        return json_path

    logger.info("Unzipping %s", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    if not json_path.exists():
        # Some zips may contain differently named files; find the JSON
        extracted = [f for f in output_dir.glob("*.json")]
        logger.warning("Expected %s not found. Extracted files: %s", json_name, extracted)
        if extracted:
            json_path = extracted[-1]

    logger.info("Extracted → %s", json_path)
    return json_path


# ── Validation ───────────────────────────────────────────────────────────────

def validate_json(json_path: Path) -> list | dict:
    """Load and validate a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
    """
    logger.info("Validating JSON: %s", json_path.name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Valid JSON: %s (%d top-level entries)", json_path.name,
                len(data) if isinstance(data, list) else 1)
    return data


# ── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(output_dir: Path) -> dict:
    """Compute and return summary statistics for the downloaded Visual Genome data.

    Args:
        output_dir: Directory containing the extracted JSON files.

    Returns:
        Dictionary of summary statistics.
    """
    stats = {}

    # Image data
    image_data_path = output_dir / "image_data.json"
    if image_data_path.exists():
        logger.info("Loading image_data.json for stats...")
        with open(image_data_path, "r", encoding="utf-8") as f:
            image_data = json.load(f)
        stats["total_images"] = len(image_data)
        logger.info("Total images: %d", stats["total_images"])

    # Relationships
    rel_path = output_dir / "relationships.json"
    if rel_path.exists():
        logger.info("Loading relationships.json for stats...")
        with open(rel_path, "r", encoding="utf-8") as f:
            rel_data = json.load(f)

        total_rels = 0
        predicates = set()
        for entry in rel_data:
            rels = entry.get("relationships", [])
            total_rels += len(rels)
            for r in rels:
                pred = r.get("predicate", "").strip().lower()
                if pred:
                    predicates.add(pred)

        stats["total_relationships"] = total_rels
        stats["unique_predicates"] = len(predicates)
        stats["avg_relationships_per_image"] = (
            round(total_rels / len(rel_data), 2) if rel_data else 0
        )
        logger.info("Total relationships: %d", stats["total_relationships"])
        logger.info("Unique predicates: %d", stats["unique_predicates"])
        logger.info("Avg relationships/image: %.2f", stats["avg_relationships_per_image"])

    # Objects
    obj_path = output_dir / "objects.json"
    if obj_path.exists():
        logger.info("Loading objects.json for stats...")
        with open(obj_path, "r", encoding="utf-8") as f:
            obj_data = json.load(f)

        categories = set()
        for entry in obj_data:
            for obj in entry.get("objects", []):
                names = obj.get("names", [])
                if names:
                    categories.add(names[0].strip().lower())

        stats["unique_object_categories"] = len(categories)
        logger.info("Unique object categories: %d", stats["unique_object_categories"])

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(output_dir: Path, force: bool = False) -> None:
    """Download, extract, validate, and summarise the Visual Genome dataset.

    Args:
        output_dir: Directory to save downloaded files.
        force: If True, re-download files even if they already exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Step 1 — Download
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Downloading Visual Genome files")
    logger.info("=" * 60)
    zip_paths = {}
    for filename, url in VG_FILES.items():
        zip_paths[filename] = download_file(url, output_dir / filename, force=force)

    # Step 2 — Unzip
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Extracting zip files")
    logger.info("=" * 60)
    json_paths = {}
    for filename, zip_path in zip_paths.items():
        json_paths[filename] = unzip_file(zip_path, output_dir)

    # Step 3 — Validate
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Validating JSON files")
    logger.info("=" * 60)
    for filename, json_path in json_paths.items():
        validate_json(json_path)

    # Stats
    logger.info("=" * 60)
    logger.info("SUMMARY — Visual Genome Statistics")
    logger.info("=" * 60)
    stats = compute_stats(output_dir)

    print("\n" + "=" * 50)
    print("  Visual Genome — Download Summary")
    print("=" * 50)
    print(f"  Output directory:          {output_dir}")
    print(f"  Total images:              {stats.get('total_images', 'N/A'):,}")
    print(f"  Total relationships:       {stats.get('total_relationships', 'N/A'):,}")
    print(f"  Unique predicates:         {stats.get('unique_predicates', 'N/A'):,}")
    print(f"  Unique object categories:  {stats.get('unique_object_categories', 'N/A'):,}")
    print(f"  Avg relationships/image:   {stats.get('avg_relationships_per_image', 'N/A')}")
    print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download the Visual Genome dataset for SceneIQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VISUAL_GENOME_DIR),
        help="Directory to save downloaded Visual Genome files.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        default=False,
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="download_vg.log")
    main(output_dir=args.output_dir, force=args.force_redownload)
