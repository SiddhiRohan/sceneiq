#!/usr/bin/env bash
# SceneIQ — One-command Docker build and run
#
# Usage:
#   bash docker_run.sh                     # build + run demo on sample images
#   bash docker_run.sh path/to/image.jpg   # build + run on your image

set -e

IMAGE_NAME="sceneiq"

echo "=============================================="
echo "  SceneIQ — Docker Build & Run"
echo "=============================================="
echo ""

# Build the image if it doesn't exist (or rebuild if Dockerfile changed)
echo "Building Docker image (this may take a few minutes the first time)..."
docker build -t "$IMAGE_NAME" .
echo ""

if [ -n "$1" ]; then
    # Run on user-provided image
    INPUT_PATH=$(realpath "$1")
    echo "Running inference on: $1"
    docker run --rm \
        -v "$INPUT_PATH:/app/input.jpg" \
        "$IMAGE_NAME" \
        python scripts/infer.py /app/input.jpg --model-type fusion --heatmap --save /app/output/result.png
else
    # Run demo on bundled sample images
    echo "Running demo on sample images..."
    docker run --rm "$IMAGE_NAME"
fi
