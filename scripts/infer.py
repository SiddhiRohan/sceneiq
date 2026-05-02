"""
SceneIQ — Run inference on a single image.

Supports two model types:

    --model-type vit     ViT-only classifier (Phase 1 or regularised Phase 2).
    --model-type fusion  ViT + GAT fusion model with optional heatmap overlay.

For fusion mode, a dummy (empty) scene graph is used when no external graph
is provided, so the model runs on arbitrary images.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom
from transformers import ViTForImageClassification, ViTImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CROSS_ATTN_DIM,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    MODELS_DIR,
    SCENE_GRAPHS_DIR,
    VIT_MODEL_NAME,
)
from utils import load_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FusionModel


def infer_vit(image_path: str, checkpoint: str, model_name: str) -> dict:
    """Run ViT-only coherence prediction.

    Args:
        image_path: Path to the input image.
        checkpoint: Path to the trained checkpoint file.
        model_name: HuggingFace model ID.

    Returns:
        Dict with prediction, probabilities, and the loaded image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return {
        "image": img,
        "coherent_prob": probs[0].item(),
        "incoherent_prob": probs[1].item(),
        "patch_scores": None,
    }


def infer_fusion(image_path: str, checkpoint: str, model_name: str,
                 graphs_dir: str) -> dict:
    """Run fusion model coherence prediction with localization.

    Uses a dummy scene graph (single unknown node, no edges) for
    arbitrary images without a pre-extracted scene graph.

    Args:
        image_path: Path to the input image.
        checkpoint: Path to the trained checkpoint file.
        model_name: HuggingFace model ID.
        graphs_dir: Path to scene graph vocabulary directory.

    Returns:
        Dict with prediction, probabilities, patch scores, and image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(model_name)

    vocab = load_json(Path(graphs_dir) / "vocab.json")
    num_categories = len(vocab["name_to_idx"])
    num_predicates = len(vocab["predicate_to_idx"])

    model = FusionModel(
        vit_model_name=model_name,
        num_categories=num_categories, num_predicates=num_predicates,
        gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_NUM_HEADS,
        gat_layers=GAT_NUM_LAYERS, fusion_dim=CROSS_ATTN_DIM,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    # Dummy graph: single unknown node, no edges
    node_labels = torch.tensor([0], dtype=torch.long, device=device)
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    graph_batch = torch.tensor([0], dtype=torch.long, device=device)
    node_counts = [1]

    with torch.no_grad():
        out = model(pixel_values, node_labels, edge_index, graph_batch, node_counts)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)[0]
        patch_scores = out["patch_scores"][0].cpu().numpy()

    return {
        "image": img,
        "coherent_prob": probs[0].item(),
        "incoherent_prob": probs[1].item(),
        "patch_scores": patch_scores,
    }


def display_result(result: dict, image_path: str, show_heatmap: bool,
                   save_path: str | None) -> None:
    """Display the prediction result with optional heatmap overlay.

    Args:
        result: Inference result dict.
        image_path: Original image path (for display).
        show_heatmap: Whether to show patch-level heatmap.
        save_path: If provided, save the figure instead of showing.
    """
    coherent_prob = result["coherent_prob"]
    incoherent_prob = result["incoherent_prob"]
    pred_label = "COHERENT" if coherent_prob > incoherent_prob else "INCOHERENT"
    scs = coherent_prob

    print()
    print("=" * 50)
    print("  SceneIQ — Inference Result")
    print("=" * 50)
    print(f"  Image:      {image_path}")
    print(f"  Prediction: {pred_label}")
    print(f"  SCS:        {scs:.4f}")
    print(f"  P(coherent):   {coherent_prob:.4f}")
    print(f"  P(incoherent): {incoherent_prob:.4f}")
    print("=" * 50)
    print()

    img = result["image"]
    color = "green" if pred_label == "COHERENT" else "red"
    patch_scores = result["patch_scores"]

    if show_heatmap and patch_scores is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title(
            f"{pred_label}  |  SCS: {scs:.3f}",
            fontsize=13, fontweight="bold", color=color,
        )
        axes[0].axis("off")

        scores_2d = patch_scores.reshape(14, 14)
        heatmap = ndimage_zoom(scores_2d, 224 / 14, order=1)
        axes[1].imshow(img.resize((224, 224)))
        axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title("Inconsistency heatmap", fontsize=13)
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(img)
        ax.set_title(
            f"{pred_label}  |  Scene Coherence Score: {scs:.3f}",
            fontsize=14, fontweight="bold", color=color,
        )
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved to: {save_path}\n")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SceneIQ coherence inference on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument(
        "--model-type", type=str, default="vit", choices=["vit", "fusion"],
        help="Model variant to use.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the model checkpoint. Auto-detected if not set.",
    )
    parser.add_argument(
        "--model-name", type=str, default=VIT_MODEL_NAME,
        help="HuggingFace model ID used during training.",
    )
    parser.add_argument(
        "--graphs-dir", type=str, default=str(SCENE_GRAPHS_DIR),
        help="Scene graph vocabulary directory (fusion mode only).",
    )
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Show inconsistency heatmap overlay (fusion mode only).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save the output figure to this path instead of displaying.",
    )
    args = parser.parse_args()

    # Auto-detect checkpoint
    if args.checkpoint is None:
        if args.model_type == "fusion":
            args.checkpoint = str(MODELS_DIR / "fusion" / "best.pt")
        else:
            args.checkpoint = str(MODELS_DIR / "vit" / "best.pt")
            if not Path(args.checkpoint).exists():
                args.checkpoint = str(MODELS_DIR / "best.pt")

    if args.model_type == "fusion":
        result = infer_fusion(args.image, args.checkpoint, args.model_name,
                              args.graphs_dir)
    else:
        result = infer_vit(args.image, args.checkpoint, args.model_name)

    display_result(result, args.image, args.heatmap, args.save)
