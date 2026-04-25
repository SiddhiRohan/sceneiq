"""
SceneIQ — Evaluation for all three modes (vit / gat / fusion).

Runs the best checkpoint from ``checkpoints/<mode>/best.pt`` against
``data/processed/splits/test.json`` and emits:

* ``metrics.json``           — accuracy, precision, recall, F1, ROC-AUC,
                               confusion matrix, and (when applicable)
                               localization metrics (pointing accuracy,
                               mean soft-IoU).
* ``per_alien_breakdown.json`` — per-alien recall.
* ``predictions.json``       — per-sample SCS, pred, and bbox metrics.
* ``confusion_matrix.png``   — 2×2 heatmap.
* ``roc_curve.png``          — ROC curve.
* ``per_alien_recall.png``   — bar chart.
* ``heatmaps/``              — overlay visualizations of top-N incoherent
                               samples with predicted heatmap + ground-truth
                               bbox (only for modes with a real heatmap).

Written per-mode under ``evaluation/<mode>/``.
"""

import argparse
import logging
import sys
from collections import defaultdict
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
from transformers import ViTImageProcessor

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    EVALUATION_DIR,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    MODE_VIT,
    MODEL_MODES,
    MODELS_DIR,
    NUM_WORKERS,
    PATCH_GRID,
    PROCESSED_DIR,
    SCENE_GRAPH_DIR,
    SG_EMBED_DIM,
    SG_MAX_OBJECTS,
    VIT_MODEL_NAME,
)
from utils import load_json, save_json, setup_logging, timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SceneIQDataset, make_collate  # noqa: E402

from models import build_model                # noqa: E402

logger = logging.getLogger("sceneiq")


# ── Inference ────────────────────────────────────────────────────────────────

def predict(model, loader, device) -> dict:
    """Run inference; return per-sample tensors + raw metadata.

    Returns:
        Dict of parallel lists / numpy arrays:
          labels, preds, probs, heatmaps (B, H, W), target_masks (B, H, W),
          image_paths, alien_objects.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_heatmaps, all_targets = [], []
    all_paths, all_aliens = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"]
            target_mask = batch["target_mask"]
            graph_batch = batch.get("graph_batch")
            if isinstance(graph_batch, dict):
                graph_batch = {k: v.to(device, non_blocking=True) for k, v in graph_batch.items()}

            output = model(pixel_values, graph_batch)
            probs = torch.sigmoid(output.scs_logit)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_heatmaps.append(torch.sigmoid(output.heatmap_logits).cpu().numpy())
            all_targets.append(target_mask.numpy())
            all_paths.extend(batch["image_paths"])
            all_aliens.extend(batch["alien_objects"])

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "heatmaps": np.concatenate(all_heatmaps, axis=0) if all_heatmaps else np.zeros((0, PATCH_GRID, PATCH_GRID)),
        "target_masks": np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, PATCH_GRID, PATCH_GRID)),
        "image_paths": all_paths,
        "alien_objects": all_aliens,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_classification_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Headline binary classification metrics + confusion matrix counts."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "n_samples": int(len(labels)),
    }


def compute_localization_metrics(
    heatmaps: np.ndarray,
    target_masks: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Pointing accuracy and mean soft-IoU, computed on incoherent samples only.

    Args:
        heatmaps: (B, H, W) predicted probabilities in [0, 1].
        target_masks: (B, H, W) ground-truth patch coverage in [0, 1].
        labels: (B,) binary labels.

    Returns:
        Dict with ``n_localization_samples``, ``pointing_accuracy``,
        ``mean_soft_iou``. Returns ``None`` for metrics with zero eligible
        samples.
    """
    incoherent = labels == 1
    n = int(incoherent.sum())
    if n == 0:
        return {"n_localization_samples": 0, "pointing_accuracy": None, "mean_soft_iou": None}

    hm = heatmaps[incoherent]
    tm = target_masks[incoherent]

    # Pointing accuracy: argmax patch must lie under the ground-truth bbox (mask > 0).
    B, H, W = hm.shape
    flat_hm = hm.reshape(B, -1)
    argmax_idx = flat_hm.argmax(axis=1)
    flat_tm = tm.reshape(B, -1)
    hit = flat_tm[np.arange(B), argmax_idx] > 0
    pointing = float(hit.mean())

    # Soft IoU on each sample, then average.
    eps = 1e-6
    inter = (hm * tm).sum(axis=(1, 2))
    union = hm.sum(axis=(1, 2)) + tm.sum(axis=(1, 2)) - inter
    iou = (inter + eps) / (union + eps)
    return {
        "n_localization_samples": n,
        "pointing_accuracy": pointing,
        "mean_soft_iou": float(iou.mean()),
    }


def compute_per_alien_recall(records: list, labels: np.ndarray, preds: np.ndarray) -> dict:
    """Recall on the incoherent class broken down by alien category."""
    hits = defaultdict(lambda: {"n": 0, "tp": 0})
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

def plot_confusion_matrix(cm: dict, out_path: Path) -> None:
    arr = np.array([[cm["true_negative"], cm["false_positive"]],
                    [cm["false_negative"], cm["true_positive"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["coherent", "incoherent"],
                yticklabels=["coherent", "incoherent"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (test)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    if len(set(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_per_alien_recall(breakdown: dict, out_path: Path, top_n: int = 20) -> None:
    if not breakdown:
        return
    items = sorted(breakdown.items(), key=lambda kv: kv[1]["n"], reverse=True)[:top_n]
    aliens = [k for k, _ in items]; recalls = [v["recall"] for _, v in items]
    counts = [v["n"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(aliens))))
    ax.barh(aliens[::-1], recalls[::-1], color="steelblue")
    for i, (r, n) in enumerate(zip(recalls[::-1], counts[::-1])):
        ax.text(min(r + 0.02, 0.98), i, f"n={n}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Recall"); ax.set_title(f"Per-alien recall (top {len(aliens)})")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_heatmap_overlays(
    image_paths: list,
    heatmaps: np.ndarray,
    target_masks: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    out_dir: Path,
    max_samples: int = 12,
) -> None:
    """For a handful of incoherent samples, render: image / heatmap / ground truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pick the highest-confidence incoherent predictions
    order = np.argsort(-probs)
    chosen = [i for i in order if labels[i] == 1][:max_samples]
    for rank, i in enumerate(chosen):
        try:
            img = Image.open(image_paths[i]).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping heatmap for %s: %s", image_paths[i], exc)
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(img); axes[0].set_title(f"input (SCS={1 - probs[i]:.2f})"); axes[0].axis("off")
        axes[1].imshow(heatmaps[i], cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("predicted heatmap"); axes[1].axis("off")
        axes[2].imshow(target_masks[i], cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("ground-truth patch mask"); axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{rank:02d}.png", dpi=120)
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    mode: str,
    splits_dir: Path,
    checkpoints_dir: Path,
    eval_dir: Path,
    scene_graphs_dir: Path,
    model_name: str,
    checkpoint: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Run evaluation on a single trained model."""
    if mode not in MODEL_MODES:
        raise ValueError(f"--mode must be one of {MODEL_MODES}, got {mode!r}")

    splits_dir = Path(splits_dir)
    checkpoints_dir = Path(checkpoints_dir) / mode
    eval_dir = Path(eval_dir) / mode
    eval_dir.mkdir(parents=True, exist_ok=True)
    scene_graphs_dir = Path(scene_graphs_dir)

    ckpt_path = checkpoints_dir / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py --mode {mode} first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Scene-graph data (only when needed) ──
    need_graphs = mode in ("gat", "fusion")
    if need_graphs:
        obj_vocab = load_json(scene_graphs_dir / "object_vocab.json")
        pred_vocab = load_json(scene_graphs_dir / "predicate_vocab.json")
        graphs = load_json(scene_graphs_dir / "graphs.json")
        n_objects, n_predicates = len(obj_vocab), len(pred_vocab)
    else:
        n_objects = n_predicates = 2
        graphs = {}

    model = build_model(
        mode=mode, vit_model_name=model_name,
        n_objects=n_objects, n_predicates=n_predicates,
        patch_grid=PATCH_GRID, sg_embed_dim=SG_EMBED_DIM,
        gat_hidden_dim=GAT_HIDDEN_DIM, gat_num_heads=GAT_NUM_HEADS,
        gat_num_layers=GAT_NUM_LAYERS,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    logger.info("Loaded checkpoint %s", ckpt_path)

    processor = ViTImageProcessor.from_pretrained(model_name)
    test_records = load_json(splits_dir / "test.json")
    test_ds = SceneIQDataset(test_records, processor, patch_grid=PATCH_GRID)
    collate = make_collate(graphs, getattr(model, "gat", None), device=None, sg_max_objects=SG_MAX_OBJECTS)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
    )

    # ── Inference ──
    out = predict(model, test_loader, device)

    # ── Metrics ──
    clf = compute_classification_metrics(out["labels"], out["preds"], out["probs"])
    loc = compute_localization_metrics(out["heatmaps"], out["target_masks"], out["labels"])
    breakdown = compute_per_alien_recall(test_records, out["labels"], out["preds"])
    metrics = {"mode": mode, **clf, "localization": loc}

    save_json(metrics, eval_dir / "metrics.json")
    save_json(breakdown, eval_dir / "per_alien_breakdown.json")
    predictions = [
        {
            "image_path": rec["image_path"],
            "label": int(y), "pred": int(p), "prob_incoherent": float(pr),
            "alien_object": rec.get("alien_object"),
        }
        for rec, y, p, pr in zip(test_records, out["labels"], out["preds"], out["probs"])
    ]
    save_json(predictions, eval_dir / "predictions.json")

    # ── Plots ──
    plot_confusion_matrix(clf["confusion_matrix"], eval_dir / "confusion_matrix.png")
    plot_roc(out["labels"], out["probs"], eval_dir / "roc_curve.png")
    plot_per_alien_recall(breakdown, eval_dir / "per_alien_recall.png")
    if mode in ("vit", "fusion"):
        plot_heatmap_overlays(
            image_paths=out["image_paths"],
            heatmaps=out["heatmaps"],
            target_masks=out["target_masks"],
            labels=out["labels"],
            probs=out["probs"],
            out_dir=eval_dir / "heatmaps",
        )

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  SceneIQ — Evaluation Summary  (mode={mode})")
    print("=" * 60)
    print(f"  Test samples:           {clf['n_samples']}")
    print(f"  Accuracy:               {clf['accuracy']:.4f}")
    print(f"  Precision/Recall/F1:    {clf['precision']:.4f} / {clf['recall']:.4f} / {clf['f1']:.4f}")
    print(f"  ROC-AUC:                {clf['roc_auc']}")
    if loc["n_localization_samples"] > 0 and loc["pointing_accuracy"] is not None:
        print(f"  Pointing accuracy:      {loc['pointing_accuracy']:.4f}  (n={loc['n_localization_samples']})")
        print(f"  Mean soft-IoU:          {loc['mean_soft_iou']:.4f}")
    else:
        print("  Localization:           n/a (mode has no heatmap or no incoherent test samples)")
    print(f"  Outputs in:             {eval_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SceneIQ model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default=MODE_VIT, choices=list(MODEL_MODES))
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--checkpoints-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(EVALUATION_DIR))
    parser.add_argument("--scene-graphs-dir", type=str, default=str(SCENE_GRAPH_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=f"evaluate_{args.mode}.log")
    main(
        mode=args.mode,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        eval_dir=args.eval_dir,
        scene_graphs_dir=args.scene_graphs_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
