# SceneIQ — Docker container for inference
# Build:  docker build -t sceneiq .
# Run:    docker run sceneiq
# Run on your own image:
#   docker run -v /path/to/image.jpg:/app/input.jpg sceneiq python scripts/infer.py /app/input.jpg --model-type fusion --heatmap --save /app/output.png

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ViT weights so the container is fully self-contained
RUN python -c "from transformers import ViTModel, ViTImageProcessor; \
    ViTModel.from_pretrained('google/vit-base-patch16-224'); \
    ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')"

# Copy project code
COPY config.py utils.py ./
COPY scripts/infer.py scripts/models.py scripts/__init__.py* ./scripts/

# Copy trained model and vocabulary
COPY models/fusion/best.pt ./models/fusion/best.pt
COPY data/processed/scene_graphs/vocab.json ./data/processed/scene_graphs/vocab.json

# Copy sample images for demo
COPY sample_coherent.jpg sample_incoherent.jpg ./

# Default: run demo on both sample images
CMD ["python", "scripts/docker_demo.py"]
