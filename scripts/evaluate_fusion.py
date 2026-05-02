"""
SceneIQ — Phase 2 evaluation: classification metrics, localization, ablation.

Evaluates the fusion model (or any Phase 2 variant) on the test split,
computing:

* Standard classification metrics (accuracy, precision, recall, F1, AUC).
* Per-patch localization: pointing accuracy against ground-truth alien
  bounding boxes (``paste_bbox`` from ``data/synthetic/metadata.json``).
* Heatmap visualizations overlaid on sample images.
* Ablation comparison table across all three model variants.

Run after all three ``train_fusion.py`` runs have completed.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    CROSS_ATTN_DIM,
    EVALUATION_DIR,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    MODELS_DIR,
    NUM_WORKERS,
    PROCESSED_DIR,
    SCENE_GRAPHS_DIR,
    SYNTHETIC_DIR,
    VIT_MODEL_NAME,
)
from utils import load_json, save_json, setup_logging, timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FusionModel, GATClassifier
from train_fusion import SceneIQFusionDataset, collate_fusion

logger = logging.getLogger("sceneiq")


# ── Prediction helpers ──────────────────────────────────────────────────────

def predict_vit(model, loader, device):
    """Run ViT-only inference.

    Args:
        model: ViTForImageClassification.
        loader: DataLoader.
        device: Target device.

    Returns:
        Tuple of (labels, preds, probs) as numpy arrays.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict-vit", unit="batch"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"]
            logits = model(pixel_values=pv).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def predict_gat(model, loader, device):
    """Run GAT-only inference.

    Args:
        model: GATClassifier.
        loader: DataLoader.
        device: Target device.

    Returns:
        Tuple of (labels, preds, probs) as numpy arrays.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict-gat", unit="batch"):
            labels = batch["labels"]
            gb = batch["graph_batch"].to(device)
            logits = model(gb.x, gb.edge_index, gb.batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def predict_fusion(model, loader, device):
    """Run fusion model inference, returning patch scores for localization.

    Args:
        model: FusionModel.
        loader: DataLoader.
        device: Target device.

    Returns:
        Tuple of (labels, preds, probs, all_patch_scores).
        all_patch_scores is a list of (196,) numpy arrays.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_patch_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict-fusion", unit="batch"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"]
            gb = batch["graph_batch"].to(device)
            node_counts = batch["node_counts"]

            out = model(pv, gb.x, gb.edge_index, gb.batch, node_counts)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            patch_scores = out["patch_scores"]

            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            for i in range(patch_scores.size(0)):
                all_patch_scores.append(patch_scores[i].cpu().numpy())

    return (
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), all_patch_scores,
    )


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs):
    """Compute classification metrics.

    Args:
        labels: Ground-truth labels.
        preds: Predicted labels.
        probs: Incoherent-class probabilities.

    Returns:
        Dict of metric values.
    """
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
        "n_samples": int(len(labels)),
    }


# ── Localization ────────────────────────────────────────────────────────────

def build_bbox_lookup(metadata_path: Path) -> dict:
    """Build a lookup from image path to paste bounding box.

    Args:
        metadata_path: Path to ``data/synthetic/metadata.json``.

    Returns:
        Dict mapping ``image_path`` -> ``(x, y, w, h)``.
    """
    metadata = load_json(metadata_path)
    lookup = {}
    for entry in metadata:
        img_path = entry.get("output_path", "")
        bbox = entry.get("paste_bbox")
        if bbox:
            lookup[img_path] = tuple(bbox)
    return lookup


def compute_pointing_accuracy(
    records: list,
    labels: np.ndarray,
    preds: np.ndarray,
    patch_scores: list,
    bbox_lookup: dict,
    image_size: int = 224,
    grid_size: int = 14,
) -> dict:
    """Compute pointing accuracy: does the top patch hit the alien bbox?

    Args:
        records: Test-split records.
        labels: Ground-truth labels.
        preds: Model predictions.
        patch_scores: List of (196,) arrays with per-patch scores.
        bbox_lookup: Dict from image_path to (x, y, w, h).
        image_size: Input image resolution.
        grid_size: ViT patch grid size (14 for 224/16).

    Returns:
        Dict with pointing accuracy, hit count, and total incoherent.
    """
    patch_size = image_size // grid_size
    hits = 0
    total = 0

    for i, rec in enumerate(records):
        if labels[i] != 1:
            continue

        img_path = rec.get("image_path", "")
        bbox = bbox_lookup.get(img_path)
        if bbox is None:
            # Try relative path match
            for key in bbox_lookup:
                if key in img_path or img_path in key:
                    bbox = bbox_lookup[key]
                    break
        if bbox is None:
            continue

        total += 1
        scores_2d = patch_scores[i].reshape(grid_size, grid_size)
        top_patch = np.unravel_index(scores_2d.argmax(), scores_2d.shape)
        patch_center_x = top_patch[1] * patch_size + patch_size // 2
        patch_center_y = top_patch[0] * patch_size + patch_size // 2

        bx, by, bw, bh = bbox
        # Scale bbox to 224x224 (bbox is in original image coords)
        # We need the original image size to scale properly
        # For simplicity, check if bbox is already in normalized or pixel coords
        # The synthetic pipeline saves pixel coords in the original image
        # Since ViT resizes to 224x224, we need to load the original to get scale
        # Approximation: use bbox directly if it seems reasonable
        try:
            orig_img = Image.open(img_path)
            ow, oh = orig_img.size
            sx = image_size / ow
            sy = image_size / oh
            bx_s = bx * sx
            by_s = by * sy
            bw_s = bw * sx
            bh_s = bh * sy
        except Exception:
            bx_s, by_s, bw_s, bh_s = bx, by, bw, bh

        if (bx_s <= patch_center_x <= bx_s + bw_s and
                by_s <= patch_center_y <= by_s + bh_s):
            hits += 1

    return {
        "pointing_accuracy": hits / max(total, 1),
        "hits": hits,
        "total_incoherent_with_bbox": total,
    }


# ── Heatmap visualization ──────────────────────────────────────────────────

def generate_heatmaps(
    records: list,
    labels: np.ndarray,
    preds: np.ndarray,
    patch_scores: list,
    out_dir: Path,
    n_samples: int = 20,
    grid_size: int = 14,
) -> None:
    """Generate heatmap overlays for a sample of test images.

    Saves correct detections and missed detections separately.

    Args:
        records: Test-split records.
        labels: Ground-truth labels.
        preds: Model predictions.
        patch_scores: List of (196,) arrays.
        out_dir: Directory to save heatmap images.
        n_samples: Number of heatmaps per category to generate.
        grid_size: ViT patch grid size.
    """
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Separate into correct detections and misses (incoherent only)
    correct = []
    missed = []
    for i, rec in enumerate(records):
        if labels[i] != 1:
            continue
        if preds[i] == 1:
            correct.append(i)
        else:
            missed.append(i)

    def _save_heatmaps(indices, prefix):
        for j, idx in enumerate(indices[:n_samples]):
            try:
                img = Image.open(records[idx]["image_path"]).convert("RGB")
            except Exception:
                continue

            scores_2d = patch_scores[idx].reshape(grid_size, grid_size)
            scale = 224 / grid_size
            heatmap = ndimage_zoom(scores_2d, scale, order=1)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img.resize((224, 224)))
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(img.resize((224, 224)))
            axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[1].set_title(f"Inconsistency heatmap ({prefix})")
            axes[1].axis("off")

            fig.tight_layout()
            fig.savefig(heatmap_dir / f"{prefix}_{j:03d}.png", dpi=100)
            plt.close(fig)

    _save_heatmaps(correct, "hit")
    _save_heatmaps(missed, "miss")
    logger.info(
        "Saved %d hit + %d miss heatmaps to %s",
        min(len(correct), n_samples), min(len(missed), n_samples), heatmap_dir,
    )


# ── Ablation comparison ────────────────────────────────────────────────────

def ablation_table(models_dir: Path, eval_dir: Path) -> dict:
    """Build an ablation comparison table from saved test metrics.

    Args:
        models_dir: Base models directory.
        eval_dir: Evaluation output directory.

    Returns:
        Dict of ``{model_type: metrics}``.
    """
    table = {}
    for variant in ["vit", "gat", "fusion"]:
        metrics_path = models_dir / variant / "test_metrics.json"
        if metrics_path.exists():
            table[variant] = load_json(metrics_path)

    # Also include Phase 1 baseline if available
    phase1_path = models_dir / "test_metrics.json"
    if phase1_path.exists():
        table["vit_phase1"] = load_json(phase1_path)

    save_json(table, eval_dir / "ablation_comparison.json")

    print("\n" + "=" * 70)
    print("  Ablation Comparison")
    print("=" * 70)
    print(f"  {'Model':<15} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("  " + "-" * 55)
    for name, m in table.items():
        print(
            f"  {name:<15} {m.get('accuracy', 0):>10.4f} "
            f"{m.get('f1', 0):>10.4f} "
            f"{m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>10.4f}"
        )
    print("=" * 70 + "\n")
    return table


# ── Main ────────────────────────────────────────────────────────────────────

@timer
def main(
    model_type: str,
    splits_dir: Path,
    models_dir: Path,
    graphs_dir: Path,
    eval_dir: Path,
    model_name: str,
    checkpoint: str,
    batch_size: int,
    num_workers: int,
    n_heatmaps: int,
    compare: bool,
) -> None:
    """Evaluate a Phase 2 model variant.

    Args:
        model_type: One of ``vit``, ``gat``, ``fusion``.
        splits_dir: Directory with test.json.
        models_dir: Base models directory.
        graphs_dir: Scene graph index directory.
        eval_dir: Output directory for results.
        model_name: HF ViT model ID.
        checkpoint: Checkpoint filename.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        n_heatmaps: Number of heatmap samples to generate.
        compare: If True, also print ablation comparison table.
    """
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    out_dir = Path(eval_dir) / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Evaluating: %s", device, model_type)

    # Load data
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading test data")
    logger.info("=" * 60)

    need_graphs = model_type in ("gat", "fusion")
    need_images = model_type in ("vit", "fusion")

    graph_index, name_to_idx, num_categories, num_predicates = {}, {}, 1, 2
    if need_graphs:
        vocab = load_json(Path(graphs_dir) / "vocab.json")
        name_to_idx = vocab["name_to_idx"]
        num_categories = len(name_to_idx)
        num_predicates = len(vocab["predicate_to_idx"])
        graph_index = load_json(Path(graphs_dir) / "graph_index.json")

    processor = ViTImageProcessor.from_pretrained(model_name) if need_images else None

    test_ds = SceneIQFusionDataset(
        splits_dir / "test.json", processor=processor,
        graph_index=graph_index, name_to_idx=name_to_idx,
        need_images=need_images,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fusion,
        pin_memory=device.type == "cuda",
    )

    # Load model
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Loading model")
    logger.info("=" * 60)

    ckpt_path = models_dir / model_type / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, preds, probs = predict_vit(model, test_loader, device)
        patch_scores = None
    elif model_type == "gat":
        model = GATClassifier(
            num_categories=num_categories, num_predicates=num_predicates,
            hidden_dim=GAT_HIDDEN_DIM, num_heads=GAT_NUM_HEADS,
            num_layers=GAT_NUM_LAYERS,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, preds, probs = predict_gat(model, test_loader, device)
        patch_scores = None
    else:
        model = FusionModel(
            vit_model_name=model_name,
            num_categories=num_categories, num_predicates=num_predicates,
            gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_NUM_HEADS,
            gat_layers=GAT_NUM_LAYERS, fusion_dim=CROSS_ATTN_DIM,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, preds, probs, patch_scores = predict_fusion(model, test_loader, device)

    # Metrics
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Computing metrics")
    logger.info("=" * 60)

    metrics = compute_metrics(labels, preds, probs)
    save_json(metrics, out_dir / "metrics.json")

    # Localization (fusion only)
    if patch_scores is not None:
        metadata_path = SYNTHETIC_DIR / "metadata.json"
        if metadata_path.exists():
            bbox_lookup = build_bbox_lookup(metadata_path)
            pointing = compute_pointing_accuracy(
                test_ds.records, labels, preds, patch_scores, bbox_lookup,
            )
            metrics["pointing_accuracy"] = pointing["pointing_accuracy"]
            metrics["pointing_hits"] = pointing["hits"]
            metrics["pointing_total"] = pointing["total_incoherent_with_bbox"]
            save_json(metrics, out_dir / "metrics.json")
            logger.info("Pointing accuracy: %.4f (%d/%d)",
                        pointing["pointing_accuracy"], pointing["hits"],
                        pointing["total_incoherent_with_bbox"])

        generate_heatmaps(
            test_ds.records, labels, preds, patch_scores,
            out_dir, n_samples=n_heatmaps,
        )

    # Summary
    print("\n" + "=" * 60)
    print(f"  SceneIQ — Evaluation Summary ({model_type})")
    print("=" * 60)
    print(f"  Checkpoint:         {ckpt_path}")
    print(f"  Test samples:       {metrics['n_samples']}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics.get('precision', 'N/A')}")
    print(f"  Recall:             {metrics.get('recall', 'N/A')}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"  ROC-AUC:            {metrics.get('roc_auc', 'N/A')}")
    if "pointing_accuracy" in metrics:
        print(f"  Pointing accuracy:  {metrics['pointing_accuracy']:.4f}")
    print(f"  Outputs in:         {out_dir}")
    print("=" * 60 + "\n")

    if compare:
        ablation_table(models_dir, Path(eval_dir))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a Phase 2 SceneIQ model variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="fusion",
                        choices=["vit", "gat", "fusion"])
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--graphs-dir", type=str, default=str(SCENE_GRAPHS_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(EVALUATION_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--n-heatmaps", type=int, default=20)
    parser.add_argument("--compare", action="store_true",
                        help="Print ablation comparison table.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=f"evaluate_{args.model}.log")
    main(
        model_type=args.model,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        graphs_dir=args.graphs_dir,
        eval_dir=args.eval_dir,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_heatmaps=args.n_heatmaps,
        compare=args.compare,
    )
