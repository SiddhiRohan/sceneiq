# SceneIQ

**Detecting Logical Inconsistencies in Real-World Photographs Using Vision Transformers and Scene-Graph Fusion**

MSML640 — Computer Vision, Spring 2026  
University of Maryland

**Group 12:** Siddhi Rohan Chakka, Krishna Kishore Buddi, Dhanush Vasa

---

## Overview

SceneIQ classifies whether a photograph is **coherent** (the scene makes sense)
or **incoherent** (an out-of-place object has been inserted). It combines a
Vision Transformer (ViT) backbone with a Graph Attention Network (GAT) that
encodes scene-graph relationships, fused via cross-attention. The model also
produces per-patch inconsistency heatmaps that highlight where the anomaly is.

**Final model performance (test set, n=2,250):**

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.6%  |
| F1        | 91.8%  |
| ROC-AUC   | 98.0%  |

## Quick Start — Run Inference

### Step 1: Clone or unzip the repository

```bash
git lfs install          # required once — installs Git LFS
git clone https://github.com/SiddhiRohan/sceneiq.git
cd sceneiq
```

> **Note:** This repo uses [Git LFS](https://git-lfs.github.com/) for the model
> checkpoint (`models/fusion/best.pt`). Install Git LFS before cloning, otherwise
> you'll get a broken pointer file instead of the actual model. Alternatively,
> use the zip submission which includes the checkpoint directly.

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run inference

```bash
# Basic prediction
python scripts/infer.py path/to/your/image.jpg --model-type fusion

# With inconsistency heatmap overlay
python scripts/infer.py path/to/your/image.jpg --model-type fusion --heatmap

# Save output to file instead of displaying
python scripts/infer.py path/to/your/image.jpg --model-type fusion --heatmap --save output.png
```

**Or run the demo script to test on a sample image:**

```bash
bash run_demo.sh
```

That's it — 3 steps, no retraining required.

## Docker (single command)

Run the entire project with one command — no local Python or dependencies needed.
Only requires [Docker](https://www.docker.com/products/docker-desktop/) to be installed and running.

```bash
# Run demo on bundled sample images (builds automatically on first run)
bash docker_run.sh

# Run on your own image
bash docker_run.sh path/to/your/image.jpg
```

The script builds the Docker image on first run (~5 minutes, requires
internet to pull dependencies) and then runs inference. After the
initial build, the image is fully self-contained: all dependencies,
the trained model checkpoint, and pretrained ViT weights are baked in.

**What happens:**
- Runs the fusion model on two bundled sample images (one coherent, one incoherent)
- Prints predictions and confidence scores to the terminal
- Saves heatmap result images to `docker_output/` on your machine
- Automatically opens the result images so you can see the predictions visually

You can also build and run manually:

```bash
docker build -t sceneiq .
docker run --rm -v "$(pwd)/docker_output:/app/output" sceneiq
```

## Project Structure

```
sceneiq/
├── config.py                  # Central configuration (paths, hyperparameters)
├── utils.py                   # Shared utilities (logging, JSON I/O, seeding)
├── requirements.txt           # Python dependencies
├── run_demo.sh                # One-command local demo script
├── Dockerfile                 # Docker containerization for inference
├── docker_run.sh              # One-command Docker build + run script
├── sample_coherent.jpg        # Sample coherent image for demo
├── sample_incoherent.jpg      # Sample incoherent image for demo
│
├── scripts/
│   ├── infer.py               # Single-image inference (ViT or fusion)
│   ├── docker_demo.py         # Docker demo entrypoint
│   ├── models.py              # Model architectures (GAT, Fusion, etc.)
│   ├── train.py               # Phase 1 ViT-only training
│   ├── train_fusion.py        # Phase 2 fusion training (ViT+GAT)
│   ├── evaluate.py            # Phase 1 evaluation
│   ├── evaluate_fusion.py     # Phase 2 evaluation with localization
│   ├── extract_scene_graphs.py # Extract scene graphs from Visual Genome
│   ├── download_vg.py         # Download Visual Genome dataset
│   ├── build_co_occurrence.py # Build object co-occurrence statistics
│   ├── generate_synthetic.py  # Generate incoherent composite images
│   └── prepare_dataset.py     # Assemble train/val/test splits
│
├── models/                    # Model checkpoints
│   └── fusion/best.pt         # Final trained model (included in repo)
│
├── evaluation/                # Evaluation outputs (metrics, plots, heatmaps)
├── data/                      # Dataset files (not included in repo)
│   ├── raw/visual_genome/     # Visual Genome images and annotations
│   ├── processed/             # Train/val/test splits, scene graphs
│   └── synthetic/             # Generated incoherent images
│
└── report/                    # LaTeX source for the project report
```

## Model Architecture

The fusion model has three branches:

1. **ViT backbone** (`google/vit-base-patch16-224`): extracts 196 patch token
   embeddings (14x14 grid) from the input image.

2. **GAT encoder**: embeds scene-graph nodes (object categories) and applies
   multi-head graph attention over relationship edges to produce per-node
   embeddings.

3. **Cross-attention fusion**: ViT patch tokens (queries) attend to GAT node
   embeddings (keys/values), producing fused representations that combine
   visual and structural information.

Two output heads:
- **Classification head**: mean-pooled fused features through a linear layer
  for binary coherent/incoherent prediction.
- **Localization head**: per-patch sigmoid scores producing a 14x14
  inconsistency heatmap.

## Ablation Study

| Model           | Accuracy | F1    | Description                     |
|-----------------|----------|-------|---------------------------------|
| ViT + GAT Fusion| 94.6%    | 91.8% | Full model with cross-attention |
| ViT-only (Ph. 1)| 94.4%    | 91.5% | ViT classifier, no scene graph  |
| ViT regularized | 90.5%    | 84.1% | ViT with augmentation + dropout |
| GAT-only        | 66.7%    | 0.0%  | Scene graph alone (majority cls) |

The fusion model outperforms all baselines. The GAT-only result confirms that
visual features are essential — scene-graph structure alone cannot distinguish
coherent from incoherent images. However, combining both modalities via
cross-attention yields the best performance.

## Reproducing the Full Pipeline

If you want to retrain from scratch (not required for grading):

```bash
# 1. Download Visual Genome
python scripts/download_vg.py

# 2. Build co-occurrence statistics
python scripts/build_co_occurrence.py

# 3. Generate synthetic incoherent images
python scripts/generate_synthetic.py --n-samples 5000

# 4. Prepare train/val/test splits
python scripts/prepare_dataset.py

# 5. Extract scene graphs
python scripts/extract_scene_graphs.py

# 6. Train the fusion model
python scripts/train_fusion.py --model fusion --augment --patience 3

# 7. Evaluate
python scripts/evaluate_fusion.py --model fusion --compare
```

## Configuration

All project constants live in `config.py`. Key settings:

- **ViT model**: `google/vit-base-patch16-224` (ImageNet-21k pretrained)
- **GAT**: 128-dim hidden, 4 heads, 2 layers, 0.3 dropout
- **Training**: AdamW, lr=2e-5, batch size 32, cosine LR schedule
- **Early stopping**: patience=3 on validation accuracy
- **Data**: 10K coherent + 5K incoherent, 70/15/15 train/val/test split

Override defaults via CLI arguments on any script (`--help` for options).

## Requirements

**For Docker (recommended):**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- No other dependencies needed

**For local setup:**
- Python 3.10+
- PyTorch 2.x (GPU recommended for inference speed)
- See `requirements.txt` for full dependency list

## License

This project was developed for MSML640 at the University of Maryland.
