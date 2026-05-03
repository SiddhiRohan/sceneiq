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

# Install PyTorch CPU-only first (smaller image, no CUDA)
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric with CPU-only wheels
RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Install remaining Python dependencies (no torch/torchvision/pyg — already installed)
COPY requirements.txt .
RUN grep -v -E '^#|^$|^torch==|^torchvision==|^torch-geometric==' requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt

# Pre-download ViT weights so the container is fully self-contained
RUN python -c "from transformers import ViTModel, ViTImageProcessor; \
    ViTModel.from_pretrained('google/vit-base-patch16-224'); \
    ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')"

# Copy project code
COPY config.py utils.py ./
COPY scripts/infer.py scripts/models.py scripts/docker_demo.py scripts/__init__.py ./scripts/

# Copy trained model and vocabulary
COPY models/fusion/best.pt ./models/fusion/best.pt
COPY data/processed/scene_graphs/vocab.json ./data/processed/scene_graphs/vocab.json

# Copy sample images for demo
COPY sample_coherent.jpg sample_incoherent.jpg ./

# Default: run demo on both sample images
CMD ["python", "scripts/docker_demo.py"]
