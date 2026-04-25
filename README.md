# SceneIQ

**Detecting Logical Inconsistencies in Real-World Photographs Using Vision Transformers**

MSML640 — University of Maryland · Spring 2026

---

## Overview

SceneIQ identifies logically incoherent photographs (e.g. a boat parked in a
bedroom, a penguin in a kitchen) and localizes the offending region. The
model combines a **Vision Transformer (ViT)** backbone with a **Graph
Attention Network (GAT)** over per-image scene graphs, fused via
**cross-attention**.

The main output is:

1. A **Scene Coherence Score (SCS)** — a scalar in `[0, 1]` where 1 = fully
   coherent.
2. A **14×14 per-patch inconsistency heatmap** localizing the incoherent
   region.

## Architecture variants (ablation)

| mode     | ViT | GAT | Cross-attention | Heatmap |
|----------|:---:|:---:|:---------------:|:-------:|
| `vit`    |  ✓  |     |                 |   ✓     |
| `gat`    |     |  ✓  |                 |         |
| `fusion` |  ✓  |  ✓  |       ✓         |   ✓     |

## Project structure

```
sceneiq/
├── config.py               # Central constants (paths, thresholds, seeds, hyperparams)
├── utils.py                # Shared helpers (logging, seeding, JSON I/O, timer)
├── requirements.txt
├── models/                 # Python package — architectures
│   ├── vit_branch.py       # ViT + SCS head + per-patch localization head
│   ├── gat_branch.py       # GAT encoder over VG scene graphs
│   └── fusion.py           # Cross-attention fusion + build_model(mode=...) factory
├── scripts/
│   ├── download_vg.py      # Download Visual Genome (objects, relationships, image_data, scene_graphs)
│   ├── download_coco.py    # Download MS-COCO 2017 annotations
│   ├── download_visualcomet.py # Download VisualCOMET commonsense annotations
│   ├── build_co_occurrence.py  # VG object co-occurrence matrices
│   ├── build_scene_graphs.py   # Per-image scene-graph preprocessing for the GAT branch
│   ├── build_coco_index.py     # Per-image COCO coherent-record + category pair-count index
│   ├── build_visualcomet_index.py # Per-image VisualCOMET commonsense (events/before/after/intent)
│   ├── generate_synthetic.py   # Copy-paste augmentation → incoherent images + paste_bbox
│   ├── prepare_dataset.py  # Train/val/test splits; --coco-index mixes COCO into the coherent pool
│   ├── data.py             # Shared Dataset + collation (bbox → patch mask, graph batching)
│   ├── train.py            # Train a single mode (vit | gat | fusion)
│   ├── evaluate.py         # Classification + localization metrics + heatmap overlays
│   ├── run_ablation.py     # Train+evaluate all three modes and write a comparison
│   └── demo.py             # Qualitative single-image SCS + heatmap
├── data/                   # (generated)
│   ├── raw/                # Downloaded datasets
│   ├── processed/          # Splits, co-occurrence tables, scene graphs
│   └── synthetic/          # Generated incoherent images + metadata
├── checkpoints/            # Trained model weights (one subdir per mode)
├── evaluation/             # Metrics + plots (per mode + ablation comparison)
└── logs/                   # Rolling log files, one per script
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric` wheels can be fiddly on some platforms. The GAT branch has
an automatic fallback to plain multi-head self-attention if `torch_geometric`
is not importable, so you can still train/eval all three modes — just with a
slightly weaker graph encoder.

## End-to-end pipeline

```bash
# 1. Core data (Visual Genome)
python scripts/download_vg.py
python scripts/build_co_occurrence.py
python scripts/build_scene_graphs.py
python scripts/generate_synthetic.py --n-samples 5000
python scripts/prepare_dataset.py

# 1b. Optional: mix MS-COCO into the coherent pool
python scripts/download_coco.py
python scripts/build_coco_index.py --split val2017
python scripts/prepare_dataset.py \
    --coco-index data/processed/mscoco/coherent_records_val2017.json \
    --coco-fraction 0.3

# 1c. Optional: build the VisualCOMET commonsense index
python scripts/download_visualcomet.py
python scripts/build_visualcomet_index.py

# 2. Train one mode
python scripts/train.py --mode fusion --num-epochs 10

# 3. Evaluate
python scripts/evaluate.py --mode fusion

# 4. Full ablation (trains + evaluates all three)
python scripts/run_ablation.py --num-epochs 10

# 5. Qualitative demo
python scripts/demo.py --mode fusion --image path/to/photo.jpg

# 6. Tests
python tests/run_all.py    # 276/276 should pass in a few seconds, offline
```

## Outputs

### Per-mode evaluation (`evaluation/<mode>/`)

- `metrics.json` — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- `metrics.json[.localization]` — pointing accuracy + mean soft-IoU (vit/fusion)
- `per_alien_breakdown.json` — recall sliced by alien-object category
- `predictions.json` — per-sample SCS, pred, label
- `confusion_matrix.png`, `roc_curve.png`, `per_alien_recall.png`
- `heatmaps/00.png … 11.png` — heatmap overlays on top-confidence incoherent predictions

### Ablation comparison (`evaluation/`)

- `ablation_summary.json` — flat table of `{mode: {accuracy, f1, pointing_accuracy, …}}`
- `ablation_f1_accuracy.png` — grouped bar chart

## Key design decisions

**Why VG ground-truth scene graphs, not RelTR?** We're not studying scene
graph generation — we're using scene graphs as *input*. VG already ships
ground-truth graphs for 108K images, which are bit-for-bit reproducible and
free. Swapping in RelTR/EGTR later is a drop-in replacement for the data
loaded by `build_scene_graphs.py`.

**Why BCE + soft-IoU, not BCE alone?** The proposal defines a per-patch
heatmap as a first-class output. Soft-IoU against the bbox-derived patch
mask is differentiable, only kicks in on incoherent samples, and is small
enough not to destabilize the coherence loss.

**Why the paste_bbox lives in `generate_synthetic.py` metadata?** Free
supervision. The moment we paste an alien object we know exactly where the
inconsistency is — writing it to JSON lets every downstream script (training
loss, evaluation, heatmap visualization) use it without re-deriving.

## Experiment tracking

Experiments optionally log to [Weights & Biases](https://wandb.ai):

```bash
wandb login
python scripts/train.py --mode fusion --wandb
```

Set `WANDB_ENTITY` in `config.py` if you don't want to pass it per-run.
