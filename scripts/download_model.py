"""
SceneIQ — Download the trained fusion model from HuggingFace.

Downloads the final model checkpoint and scene-graph vocabulary
required for inference. Skips files that already exist locally.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, SCENE_GRAPHS_DIR
from utils import setup_logging

logger = logging.getLogger("sceneiq")

HUGGINGFACE_REPO = "SiddhiRohan/sceneiq"
FILES = [
    ("models/fusion/best.pt", MODELS_DIR / "fusion" / "best.pt"),
    ("data/processed/scene_graphs/vocab.json", SCENE_GRAPHS_DIR / "vocab.json"),
]


def main(repo_id: str, force: bool) -> None:
    """Download model checkpoint and vocabulary from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID.
        force: If True, re-download even if files exist locally.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    for remote_path, local_path in FILES:
        if local_path.exists() and not force:
            logger.info("Already exists, skipping: %s", local_path)
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", remote_path, local_path)
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=str(local_path.parent.parent),
            local_dir_use_symlinks=False,
        )
        logger.info("Downloaded: %s", local_path)

    print("\n" + "=" * 50)
    print("  SceneIQ — Model Download Complete")
    print("=" * 50)
    print(f"  Checkpoint: {MODELS_DIR / 'fusion' / 'best.pt'}")
    print(f"  Vocabulary: {SCENE_GRAPHS_DIR / 'vocab.json'}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the trained SceneIQ model from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=str, default=HUGGINGFACE_REPO,
                        help="HuggingFace repository ID.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist.")
    args = parser.parse_args()
    setup_logging()
    main(repo_id=args.repo, force=args.force)
