"""
SceneIQ — Run inference on a single image.

Loads the trained ViT checkpoint and predicts whether a photograph is
coherent or incoherent, displaying the image with the prediction overlaid.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, VIT_MODEL_NAME


def main(image_path: str, checkpoint: str, model_name: str) -> None:
    """Run coherence prediction on a single image.

    Args:
        image_path: Path to the input image.
        checkpoint: Path to the trained checkpoint file.
        model_name: HuggingFace model ID used during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Predict
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0]

    coherent_prob = probs[0].item()
    incoherent_prob = probs[1].item()
    pred_label = "COHERENT" if coherent_prob > incoherent_prob else "INCOHERENT"
    scs = coherent_prob  # Scene Coherence Score: 1 = coherent, 0 = incoherent

    # Print results
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

    # Display image with prediction
    color = "green" if pred_label == "COHERENT" else "red"
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_title(
        f"{pred_label}  |  Scene Coherence Score: {scs:.3f}",
        fontsize=14, fontweight="bold", color=color,
    )
    ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SceneIQ coherence inference on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument(
        "--checkpoint", type=str, default=str(MODELS_DIR / "best.pt"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--model-name", type=str, default=VIT_MODEL_NAME,
        help="HuggingFace model ID used during training.",
    )
    args = parser.parse_args()
    main(args.image, args.checkpoint, args.model_name)
