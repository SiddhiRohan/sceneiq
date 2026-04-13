"""
SceneIQ — Evaluate the trained ViT coherence classifier.

Loads ``models/best.pt`` and runs it against ``data/processed/splits/test.json``
to produce:

* ``evaluation/metrics.json``          — accuracy, precision, recall, F1,
                                         ROC-AUC, confusion-matrix counts.
* ``evaluation/per_alien_breakdown.json`` — recall on the incoherent class
                                            sliced by alien-object category
                                            (e.g. which implausible objects is
                                            the model best/worst at catching).
* ``evaluation/predictions.json``      — per-sample record with prediction,
                                         true label, and probability of the
                                         incoherent class.
* ``evaluation/confusion_matrix.png``  — heatmap of the 2×2 confusion matrix.
* ``evaluation/roc_curve.png``         — ROC curve on the incoherent class.
* ``evaluation/per_alien_recall.png``  — bar chart of recall by alien object
                                         (top-N categories by frequency).

Run after ``train.py`` has finished and written ``models/best.pt``.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    EVALUATION_DIR,
    MODELS_DIR,
    NUM_WORKERS,
    PROCESSED_DIR,
    VIT_MODEL_NAME,
)
from utils import load_json, save_json, setup_logging, timer

# Allow the Dataset class and collate fn to be reused from train.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SceneIQDataset, collate  # noqa: E402

logger = logging.getLogger("sceneiq")


# ── Inference ────────────────────────────────────────────────────────────────

def predict(
    model: ViTForImageClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return labels, preds, incoherent-probs.

    Args:
        model: Trained ViT classifier.
        loader: DataLoader over the test split.
        device: Target device.

    Returns:
        Tuple of ``(labels, preds, probs_incoherent)`` as 1-D numpy arrays.
    """
    model.eval()
    all_labels: list = []
    all_preds: list = []
    all_probs: list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"]
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Compute headline classification metrics.

    Args:
        labels: Ground-truth labels (0/1).
        preds: Predicted labels.
        probs: Probability assigned to the incoherent class.

    Returns:
        Dict of scalar metrics plus the 2×2 confusion matrix.
    """
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_samples": int(len(labels)),
    }


def compute_per_alien_recall(
    records: list,
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict:
    """Recall on the incoherent class, broken down by alien-object category.

    Args:
        records: Test-split records (same order as ``labels``/``preds``).
        labels: Ground-truth labels.
        preds: Predicted labels.

    Returns:
        Dict ``{alien_object: {"n": int, "recall": float}}``.
    """
    hits: dict = defaultdict(lambda: {"n": 0, "tp": 0})
    for rec, y, p in zip(records, labels, preds):
        if y != 1:
            continue
        alien = rec.get("alien_object", "unknown")
        hits[alien]["n"] += 1
        if p == 1:
            hits[alien]["tp"] += 1
    return {
        alien: {"n": v["n"], "recall": v["tp"] / v["n"] if v["n"] else 0.0}
        for alien, v in hits.items()
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_counts: dict, out_path: Path) -> None:
    """Write a 2×2 confusion-matrix heatmap."""
    cm = np.array([
        [cm_counts["true_negative"], cm_counts["false_positive"]],
        [cm_counts["false_negative"], cm_counts["true_positive"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["coherent", "incoherent"],
        yticklabels=["coherent", "incoherent"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_roc(labels: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    """Write the ROC curve for the incoherent class."""
    if len(set(labels)) < 2:
        logger.warning("Skipping ROC plot — only one class present.")
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (incoherent class)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_per_alien_recall(breakdown: dict, out_path: Path, top_n: int = 20) -> None:
    """Bar chart of per-alien recall for the top-N most frequent alien objects."""
    if not breakdown:
        logger.warning("Skipping per-alien plot — no incoherent samples in test.")
        return
    items = sorted(breakdown.items(), key=lambda kv: kv[1]["n"], reverse=True)[:top_n]
    aliens = [k for k, _ in items]
    recalls = [v["recall"] for _, v in items]
    counts = [v["n"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(aliens))))
    ax.barh(aliens[::-1], recalls[::-1], color="steelblue")
    for i, (r, n) in enumerate(zip(recalls[::-1], counts[::-1])):
        ax.text(min(r + 0.02, 0.98), i, f"n={n}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_title(f"Per-alien recall (top {len(aliens)} by frequency)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    splits_dir: Path,
    models_dir: Path,
    eval_dir: Path,
    model_name: str,
    checkpoint: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Run full evaluation and write metrics + plots.

    Args:
        splits_dir: Directory with ``test.json``.
        models_dir: Directory with trained checkpoints.
        eval_dir: Directory to write metrics and plots into.
        model_name: HF ViT model id used during training.
        checkpoint: Checkpoint filename (e.g. ``best.pt``).
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
    """
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run scripts/train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Data
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading test split and model")
    logger.info("=" * 60)
    processor = ViTImageProcessor.from_pretrained(model_name)
    test_ds = SceneIQDataset(splits_dir / "test.json", processor)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    logger.info("Loaded checkpoint %s", ckpt_path)

    # Predict
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Running inference on test split")
    logger.info("=" * 60)
    labels, preds, probs = predict(model, test_loader, device)

    # Metrics
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Computing metrics and writing plots")
    logger.info("=" * 60)
    metrics = compute_metrics(labels, preds, probs)
    save_json(metrics, eval_dir / "metrics.json")

    breakdown = compute_per_alien_recall(test_ds.records, labels, preds)
    save_json(breakdown, eval_dir / "per_alien_breakdown.json")

    predictions = [
        {
            "image_path": rec["image_path"],
            "label": int(y),
            "pred": int(p),
            "prob_incoherent": float(pr),
            "alien_object": rec.get("alien_object"),
        }
        for rec, y, p, pr in zip(test_ds.records, labels, preds, probs)
    ]
    save_json(predictions, eval_dir / "predictions.json")

    plot_confusion_matrix(metrics["confusion_matrix"], eval_dir / "confusion_matrix.png")
    plot_roc(labels, probs, eval_dir / "roc_curve.png")
    plot_per_alien_recall(breakdown, eval_dir / "per_alien_recall.png")

    # Summary
    top_missed = sorted(
        [(k, v) for k, v in breakdown.items() if v["n"] >= 3],
        key=lambda kv: kv[1]["recall"],
    )[:5]

    print("\n" + "=" * 60)
    print("  SceneIQ — Evaluation Summary")
    print("=" * 60)
    print(f"  Checkpoint:         {ckpt_path}")
    print(f"  Test samples:       {metrics['n_samples']}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']}")
    print(f"  Confusion matrix:   {metrics['confusion_matrix']}")
    if top_missed:
        print(f"  Hardest aliens:     {[(k, round(v['recall'], 2), v['n']) for k, v in top_missed]}")
    print(f"  Outputs in:         {eval_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the trained SceneIQ ViT classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(EVALUATION_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="evaluate.log")
    main(
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        eval_dir=args.eval_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
