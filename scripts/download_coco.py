"""
SceneIQ — Download MS-COCO 2017 annotations.

Pulls ``annotations_trainval2017.zip`` (instances + captions, ~240 MB) and
extracts the JSON files into ``data/raw/mscoco/annotations/``. COCO images
themselves are lazy-downloaded at ``prepare_dataset.py`` time — we only need
the annotations up front to build a coherent-image index.

Outputs:
    data/raw/mscoco/annotations/instances_train2017.json
    data/raw/mscoco/annotations/instances_val2017.json
    data/raw/mscoco/annotations/captions_train2017.json
    data/raw/mscoco/annotations/captions_val2017.json
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MSCOCO_ANNOTATIONS_URL, MSCOCO_DIR
from utils import setup_logging, timer

logger = logging.getLogger("sceneiq")


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already exists: %s (%d bytes)", dest, dest.stat().st_size)
        return
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    logger.info("Downloaded %s (%d bytes)", dest, dest.stat().st_size)


def unzip_file(zip_path: Path, extract_to: Path) -> list:
    """Extract ``zip_path`` under ``extract_to`` and return the extracted paths."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        zf.extractall(extract_to)
    logger.info("Extracted %d members to %s", len(members), extract_to)
    return [extract_to / m for m in members]


@timer
def main(output_dir: Path, skip_download: bool) -> None:
    """Download and extract COCO annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "annotations_trainval2017.zip"

    logger.info("=" * 60)
    logger.info("STEP 1/2 — Downloading annotations zip")
    logger.info("=" * 60)
    if not skip_download:
        download_file(MSCOCO_ANNOTATIONS_URL, zip_path)
    else:
        logger.info("--skip-download set; expecting %s to already exist", zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} not found and --skip-download given.")

    logger.info("=" * 60)
    logger.info("STEP 2/2 — Extracting")
    logger.info("=" * 60)
    unzip_file(zip_path, output_dir)

    anno_dir = output_dir / "annotations"
    expected = [
        "instances_val2017.json", "instances_train2017.json",
        "captions_val2017.json", "captions_train2017.json",
    ]
    missing = [n for n in expected if not (anno_dir / n).exists()]
    if missing:
        logger.warning("Expected COCO annotation files missing: %s", missing)

    print("\n" + "=" * 60)
    print("  SceneIQ — MS-COCO Download Summary")
    print("=" * 60)
    print(f"  Zip:          {zip_path}")
    print(f"  Annotations:  {anno_dir}")
    for n in expected:
        p = anno_dir / n
        print(f"    [{'OK' if p.exists() else ' - '}] {p.name}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MS-COCO 2017 annotations for SceneIQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=str(MSCOCO_DIR))
    parser.add_argument("--skip-download", action="store_true",
                        help="Use an existing zip on disk instead of re-downloading.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="download_coco.log")
    main(output_dir=args.output_dir, skip_download=args.skip_download)
