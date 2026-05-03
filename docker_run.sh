#!/usr/bin/env bash
# SceneIQ — One-command Docker build and run
#
# Usage:
#   bash docker_run.sh                     # build + run demo on sample images
#   bash docker_run.sh path/to/image.jpg   # build + run on your image
#
# Outputs (heatmaps) are saved to ./docker_output/ on your machine.

set -e

# Prevent Git Bash on Windows from mangling /app paths in docker commands
export MSYS_NO_PATHCONV=1

IMAGE_NAME="sceneiq"
OUTPUT_DIR="$(pwd)/docker_output"
mkdir -p "$OUTPUT_DIR"

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
        -v "$OUTPUT_DIR:/app/output" \
        "$IMAGE_NAME" \
        python scripts/infer.py /app/input.jpg --model-type fusion --heatmap --save /app/output/result.png
else
    # Run demo on bundled sample images
    echo "Running demo on sample images..."
    docker run --rm \
        -v "$OUTPUT_DIR:/app/output" \
        "$IMAGE_NAME"
fi

echo ""
echo "Outputs saved to: $OUTPUT_DIR"

# Open output heatmaps (with labels and scores) so the user sees the results
if [ -z "$1" ]; then
    echo "Opening result images..."
    case "$(uname -s)" in
        Linux*)  xdg-open "$OUTPUT_DIR/sample_coherent_heatmap.png" 2>/dev/null; xdg-open "$OUTPUT_DIR/sample_incoherent_heatmap.png" 2>/dev/null ;;
        Darwin*) open "$OUTPUT_DIR/sample_coherent_heatmap.png"; open "$OUTPUT_DIR/sample_incoherent_heatmap.png" ;;
        MINGW*|MSYS*|CYGWIN*) start "$OUTPUT_DIR/sample_coherent_heatmap.png"; start "$OUTPUT_DIR/sample_incoherent_heatmap.png" ;;
    esac
fi
