"""
SceneIQ — Ablation orchestrator.

Runs train.py + evaluate.py for each of the three modes (``vit``, ``gat``,
``fusion``) and emits a single comparison JSON + bar chart. Each sub-run
writes its own logs/checkpoints under the usual per-mode directories, so
nothing here is fused into a single "giant" checkpoint.

Usage:
    python scripts/run_ablation.py --num-epochs 10
    python scripts/run_ablation.py --modes vit fusion   # skip gat
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    EVALUATION_DIR,
    LEARNING_RATE,
    MODEL_MODES,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SCENE_GRAPH_DIR,
    VIT_MODEL_NAME,
)
from utils import load_json, save_json, setup_logging, timer

logger = logging.getLogger("sceneiq")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list) -> None:
    """Execute a subprocess command, streaming output."""
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def train_and_eval(mode: str, args: argparse.Namespace) -> dict:
    """Run train.py then evaluate.py for ``mode``; return eval metrics dict."""
    python = sys.executable
    train_cmd = [
        python, str(PROJECT_ROOT / "scripts" / "train.py"),
        "--mode", mode,
        "--splits-dir", args.splits_dir,
        "--checkpoints-dir", args.checkpoints_dir,
        "--scene-graphs-dir", args.scene_graphs_dir,
        "--model-name", args.model_name,
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-epochs", str(args.num_epochs),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]
    eval_cmd = [
        python, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--mode", mode,
        "--splits-dir", args.splits_dir,
        "--checkpoints-dir", args.checkpoints_dir,
        "--eval-dir", args.eval_dir,
        "--scene-graphs-dir", args.scene_graphs_dir,
        "--model-name", args.model_name,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]
    if not args.skip_train:
        _run(train_cmd)
    _run(eval_cmd)

    metrics_path = Path(args.eval_dir) / mode / "metrics.json"
    return load_json(metrics_path)


@timer
def main(args: argparse.Namespace) -> None:
    """Run the full ablation and write the comparison artifacts."""
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for mode in args.modes:
        logger.info("=" * 70)
        logger.info(" Ablation | mode=%s", mode)
        logger.info("=" * 70)
        results[mode] = train_and_eval(mode, args)

    # Flatten to a small comparison table
    summary: dict = {}
    for mode, metrics in results.items():
        summary[mode] = {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "pointing_accuracy": metrics.get("localization", {}).get("pointing_accuracy"),
            "mean_soft_iou": metrics.get("localization", {}).get("mean_soft_iou"),
        }

    save_json(summary, eval_dir / "ablation_summary.json")

    # ── Bar chart of F1 across modes ──
    modes = list(summary.keys())
    f1s = [summary[m]["f1"] or 0.0 for m in modes]
    accs = [summary[m]["accuracy"] or 0.0 for m in modes]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(modes)))
    ax.bar([i - 0.2 for i in x], f1s, width=0.4, label="F1", color="steelblue")
    ax.bar([i + 0.2 for i in x], accs, width=0.4, label="Accuracy", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SceneIQ — Ablation (test F1 / Accuracy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eval_dir / "ablation_f1_accuracy.png", dpi=150)
    plt.close(fig)

    # ── Text summary ──
    print("\n" + "=" * 60)
    print("  SceneIQ — Ablation Summary")
    print("=" * 60)
    print(f"{'Mode':<10} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Point':>7} {'sIoU':>7}")
    for mode in modes:
        s = summary[mode]
        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
        print(f"{mode:<10} {_fmt(s['accuracy']):>7} {_fmt(s['f1']):>7} "
              f"{_fmt(s['roc_auc']):>7} {_fmt(s['pointing_accuracy']):>7} {_fmt(s['mean_soft_iou']):>7}")
    print(f"\n  Summary:    {eval_dir / 'ablation_summary.json'}")
    print(f"  Bar chart:  {eval_dir / 'ablation_f1_accuracy.png'}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SceneIQ ablation across vit / gat / fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modes", nargs="+", default=list(MODEL_MODES), choices=list(MODEL_MODES))
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--checkpoints-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(EVALUATION_DIR))
    parser.add_argument("--scene-graphs-dir", type=str, default=str(SCENE_GRAPH_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-train", action="store_true",
                        help="Only re-run evaluate.py; assumes checkpoints exist.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="run_ablation.log")
    main(args)
