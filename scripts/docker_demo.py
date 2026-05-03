"""
SceneIQ — Docker demo script.

Runs inference on the bundled sample images (one coherent, one incoherent)
and prints results to the terminal. Saves heatmap outputs to /app/output/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.infer import infer_fusion, display_result


def main():
    """Run inference on sample images and print results."""
    samples = [
        ("sample_coherent.jpg", "Coherent sample (real VG photograph)"),
        ("sample_incoherent.jpg", "Incoherent sample (synthetic composite)"),
    ]

    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  SceneIQ — Docker Inference Demo")
    print("=" * 60)
    print()

    for image_path, description in samples:
        if not Path(image_path).exists():
            print(f"  Skipping {image_path} (not found)")
            continue

        print(f"  [{description}]")
        result = infer_fusion(
            image_path=image_path,
            checkpoint="models/fusion/best.pt",
            model_name="google/vit-base-patch16-224",
            graphs_dir="models/fusion",
        )

        pred = "COHERENT" if result["coherent_prob"] > result["incoherent_prob"] else "INCOHERENT"
        scs = result["coherent_prob"]

        print(f"  Image:         {image_path}")
        print(f"  Prediction:    {pred}")
        print(f"  Coherence:     {scs:.4f}")
        print(f"  P(coherent):   {result['coherent_prob']:.4f}")
        print(f"  P(incoherent): {result['incoherent_prob']:.4f}")

        # Save heatmap
        save_path = str(output_dir / f"{Path(image_path).stem}_heatmap.png")
        display_result(result, image_path, show_heatmap=True, save_path=save_path)
        print(f"  Heatmap saved: {save_path}")
        print()

    print("=" * 60)
    print("  Demo complete. Heatmaps saved to /app/output/")
    print()
    print("  To run on your own image:")
    print("    docker run -v /path/to/image.jpg:/app/input.jpg sceneiq \\")
    print("      python scripts/infer.py /app/input.jpg --model-type fusion --heatmap")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
