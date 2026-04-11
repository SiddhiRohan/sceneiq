# SceneIQ

**Detecting Logical Inconsistencies in Real-World Photographs Using Vision Transformers**

MSML640 — University of Maryland

---

## Overview

SceneIQ identifies logically incoherent elements in photographs (e.g., a
fireplace on a beach, snow in a desert) by combining Vision Transformer (ViT)
features with scene-graph reasoning via Graph Neural Networks.

**Datasets:** Visual Genome (108K), MS-COCO (330K), VisualCOMET (59K)

## Project Structure

```
sceneiq/
├── config.py         # Central constants (paths, thresholds, seeds)
├── utils.py          # Shared helpers (logging, seeding, JSON I/O, timer)
├── requirements.txt  # Pinned Python dependencies
├── data/
│   ├── raw/          # Downloaded datasets
│   ├── processed/    # Cleaned and labeled data
│   ├── synthetic/    # Augmented incoherent images
│   └── thumbnails/   # For manual verification
├── scripts/          # Pipeline scripts (each with argparse + __main__)
├── models/           # Model definitions and checkpoints
├── evaluation/       # Metrics and visualization code
├── notebooks/        # EDA and result analysis
└── configs/          # Hyperparameter YAML/JSON files
```

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd sceneiq

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Pipeline Scripts

Each script in `scripts/` follows these conventions:
- **argparse** CLI arguments for all configurable parameters
- **Logging** with timestamps (not bare `print`)
- **Saves intermediate outputs** so nothing needs recomputation
- Run any script with `--help` to see its options

| Script | Purpose |
|--------|---------|
| `scripts/download_data.py` | Download Visual Genome, MS-COCO, VisualCOMET |
| `scripts/build_co_occurrence.py` | Build object co-occurrence matrices |
| `scripts/generate_synthetic.py` | Create incoherent images via augmentation |
| `scripts/prepare_dataset.py` | Assemble final train/val/test splits |
| `scripts/train.py` | Train ViT-based inconsistency classifier |
| `scripts/evaluate.py` | Run evaluation metrics and generate plots |

Example:

```bash
python scripts/download_data.py --dataset visual_genome --output data/raw/visual_genome
python scripts/train.py --epochs 10 --batch-size 32 --lr 2e-5
```

## Configuration

All project constants live in `config.py`:
- Dataset paths and sizes
- Co-occurrence frequency threshold (default: 10)
- Target dataset sizes (10K coherent, 5K incoherent)
- Random seed (42)
- ViT image size (224x224)
- Training hyperparameters

Override at runtime via script CLI arguments; `config.py` provides defaults.

## Experiment Tracking

Experiments are logged to [Weights & Biases](https://wandb.ai). Set your entity
in `config.py` or pass `--wandb-entity` to training scripts.

```bash
wandb login
python scripts/train.py --wandb
```
