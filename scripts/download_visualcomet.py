"""
SceneIQ — Download VisualCOMET annotations.

Pulls the VisualCOMET annotation bundle (``{train,val,test}_annots.json``) and
saves it under ``data/raw/visualcomet/``. VisualCOMET sits on top of the VCR
dataset, so the image fields point at VCR image filenames — we only consume
the per-image commonsense inferences (before/after/intent) here.

If the primary URL is unreachable (the VisualCOMET site has moved in the past),
pass ``--url`` to point at the current host.

Outputs:
    data/raw/visualcomet/train_annots.json
    data/raw/visualcomet/val_annots.json
    data/raw/visualcomet/test_annots.json
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VISUALCOMET_ANNOTATIONS_URL, VISUALCOMET_DIR
from utils import setup_logging, timer

logger = logging.getLogger("sceneiq")


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` with tqdm progress."""
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
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path) -> list:
    """Extract a zip and return the list of extracted member paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        zf.extractall(out_dir)
    logger.info("Extracted %d members from %s to %s",
                len(members), zip_path.name, out_dir)
    return [out_dir / m for m in members]


@timer
def main(url: str, output_dir: Path) -> None:
    """Download + extract VisualCOMET annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "visualcomet.zip"

    logger.info("=" * 60)
    logger.info("STEP 1/2 — Downloading VisualCOMET bundle")
    logger.info("=" * 60)
    try:
        download_file(url, zip_path)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.error(
            "The VisualCOMET dataset has changed hosts in the past. "
            "Check https://visualcomet.xyz and re-run with --url <current>. "
            "If you already have the zip, place it at %s and re-run.", zip_path,
        )
        raise

    logger.info("=" * 60)
    logger.info("STEP 2/2 — Extracting")
    logger.info("=" * 60)
    extract_zip(zip_path, output_dir)

    present = [p.name for p in output_dir.glob("*_annots.json")]
    logger.info("VisualCOMET annotation files present: %s", present)

    print("\n" + "=" * 60)
    print("  SceneIQ — VisualCOMET Download Summary")
    print("=" * 60)
    print(f"  URL:                  {url}")
    print(f"  Output directory:     {output_dir}")
    print(f"  Annotation files:     {present}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VisualCOMET commonsense annotations for SceneIQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default=VISUALCOMET_ANNOTATIONS_URL)
    parser.add_argument("--output-dir", type=str, default=str(VISUALCOMET_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="download_visualcomet.log")
    main(url=args.url, output_dir=args.output_dir)
