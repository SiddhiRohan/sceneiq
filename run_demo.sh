#!/usr/bin/env bash
# SceneIQ — Demo script
# Runs the trained fusion model on a sample image to demonstrate inference.
#
# Usage:
#   bash run_demo.sh                        # runs on a sample image
#   bash run_demo.sh path/to/your/image.jpg # runs on your image

set -e

echo "=============================================="
echo "  SceneIQ — Inference Demo"
echo "=============================================="
echo ""

# Verify model checkpoint exists
if [ ! -f "models/fusion/best.pt" ]; then
    echo "ERROR: Model checkpoint not found at models/fusion/best.pt"
    echo "Make sure you cloned with Git LFS or unzipped the full archive."
    exit 1
fi

# Run inference
if [ -n "$1" ]; then
    echo "Running inference on: $1"
    python scripts/infer.py "$1" --model-type fusion --heatmap
else
    # Find a sample image
    SAMPLE=$(find data/raw/visual_genome/images -name "*.jpg" 2>/dev/null | head -1)
    if [ -z "$SAMPLE" ]; then
        SAMPLE=$(find data/synthetic/images -name "*.jpg" 2>/dev/null | head -1)
    fi

    if [ -n "$SAMPLE" ]; then
        echo "Running inference on sample image: $SAMPLE"
        python scripts/infer.py "$SAMPLE" --model-type fusion --heatmap
    else
        echo "No sample images found in data/. Please provide an image path:"
        echo "  bash run_demo.sh path/to/your/image.jpg"
        exit 1
    fi
fi
