"""
SceneIQ — Qualitative demo on a single image.

Loads a trained checkpoint, runs it on an image, and writes a side-by-side
visualization (original + heatmap overlay + predicted SCS). Also prints a
short verdict ("COHERENT" / "INCOHERENT") so this script is useful from a
notebook or the terminal alone.

Usage:
    python scripts/demo.py --mode fusion --image path/to/photo.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EVALUATION_DIR,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    MODE_VIT,
    MODEL_MODES,
    MODELS_DIR,
    PATCH_GRID,
    SCENE_GRAPH_DIR,
    SG_EMBED_DIM,
    SG_MAX_OBJECTS,
    VIT_MODEL_NAME,
)
from utils import load_json, setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model  # noqa: E402

logger = logging.getLogger("sceneiq")


def load_model(mode: str, model_name: str, checkpoints_dir: Path,
               scene_graphs_dir: Path, device: torch.device):
    """Instantiate the architecture and load its best checkpoint."""
    if mode in ("gat", "fusion"):
        obj_vocab = load_json(scene_graphs_dir / "object_vocab.json")
        pred_vocab = load_json(scene_graphs_dir / "predicate_vocab.json")
        n_objects, n_predicates = len(obj_vocab), len(pred_vocab)
        graphs = load_json(scene_graphs_dir / "graphs.json")
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
    ckpt_path = Path(checkpoints_dir) / mode / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()
    return model, graphs


def make_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Upsample the heatmap to the image size and return an RGBA overlay."""
    img_np = np.array(image.convert("RGB")) / 255.0
    heat = np.array(Image.fromarray((heatmap * 255).astype(np.uint8))
                    .resize(image.size, Image.BILINEAR)) / 255.0
    cmap = plt.get_cmap("hot")
    heat_rgb = cmap(heat)[..., :3]
    overlay = (1 - alpha) * img_np + alpha * heat_rgb
    return np.clip(overlay, 0, 1)


def main(args: argparse.Namespace) -> None:
    """Run the demo and save a visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, graphs = load_model(
        mode=args.mode,
        model_name=args.model_name,
        checkpoints_dir=Path(args.checkpoints_dir),
        scene_graphs_dir=Path(args.scene_graphs_dir),
        device=device,
    )
    processor = ViTImageProcessor.from_pretrained(args.model_name)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # If we have a scene_image_id supplied, pull its graph; else a blank 1-node graph.
    graph_batch = None
    if args.mode in ("gat", "fusion"):
        gat = getattr(model, "gat", None)
        if gat is None:
            raise RuntimeError("GAT submodule missing from model.")
        if args.scene_image_id and str(args.scene_image_id) in graphs:
            graph_list = [graphs[str(args.scene_image_id)]]
        else:
            graph_list = [{"objects": [1], "edges": []}]
        graph_batch = gat.collate_graphs(graph_list, n_max=SG_MAX_OBJECTS, device=device)

    with torch.no_grad():
        out = model(pixel_values, graph_batch)
    scs = torch.sigmoid(out.scs_logit)[0].item()          # 1 = incoherent prob (BCE target was 1 for incoherent)
    scs_coherent = 1.0 - scs
    heatmap = torch.sigmoid(out.heatmap_logits)[0].cpu().numpy()   # (grid, grid) in [0, 1]
    verdict = "INCOHERENT" if scs > 0.5 else "COHERENT"

    # ── Visualization ──
    overlay = make_heatmap_overlay(image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Patch heatmap (14×14)"); axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"{verdict}  |  SCS(coherent)={scs_coherent:.2f}")
    axes[2].axis("off")
    fig.tight_layout()
    out_path = Path(args.out) if args.out else Path(args.eval_dir) / f"demo_{image_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"  SceneIQ — Demo  (mode={args.mode})")
    print("=" * 60)
    print(f"  Image:                 {image_path}")
    print(f"  Verdict:               {verdict}")
    print(f"  SCS (coherent):        {scs_coherent:.4f}")
    print(f"  SCS (incoherent):      {scs:.4f}")
    print(f"  Visualization:         {out_path}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SceneIQ on a single image for qualitative inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--mode", type=str, default=MODE_VIT, choices=list(MODEL_MODES))
    parser.add_argument("--scene-image-id", type=str, default=None,
                        help="Optional VG scene_image_id to look up a scene graph "
                             "(only used in gat/fusion modes).")
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--checkpoints-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--scene-graphs-dir", type=str, default=str(SCENE_GRAPH_DIR))
    parser.add_argument("--eval-dir", type=str, default=str(EVALUATION_DIR))
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: eval_dir/demo_<stem>.png)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="demo.log")
    main(args)
