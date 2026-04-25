"""
SceneIQ — Central configuration for all project constants.

All paths, thresholds, dataset sizes, and hyperparameters live here
so every script imports from a single source of truth.
"""

import os
from pathlib import Path

# =============================================================================
# Project root
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Directory paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
MODELS_DIR = PROJECT_ROOT / "checkpoints"   # trained-weight checkpoints (models/ is now the Python package)
CONFIGS_DIR = PROJECT_ROOT / "configs"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Dataset-specific paths
# =============================================================================
VISUAL_GENOME_DIR = RAW_DIR / "visual_genome"
MSCOCO_DIR = RAW_DIR / "mscoco"
VISUALCOMET_DIR = RAW_DIR / "visualcomet"

# =============================================================================
# Dataset sizes (approximate)
# =============================================================================
DATASET_SIZES = {
    "visual_genome": 108_000,
    "mscoco": 330_000,
    "visualcomet": 59_000,
}

# =============================================================================
# Co-occurrence & sampling
# =============================================================================
CO_OCCURRENCE_FREQ_THRESHOLD = 10  # minimum co-occurrence count to keep a pair
TARGET_COHERENT_SIZE = 10_000
TARGET_INCOHERENT_SIZE = 5_000
STRATIFIED_SAMPLE_PER_TYPE = 1_000  # per inconsistency type

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Model / image settings
# =============================================================================
IMAGE_SIZE = (224, 224)  # ViT input resolution
VIT_MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10

# Model-mode constants used by scripts/train.py --mode
MODE_VIT = "vit"
MODE_GAT = "gat"
MODE_FUSION = "fusion"
MODEL_MODES = (MODE_VIT, MODE_GAT, MODE_FUSION)

# ViT patch grid size (224 / 16 = 14) — localization heatmap resolution.
PATCH_GRID = 14

# =============================================================================
# MS-COCO (coherent image source + category/caption annotations)
# =============================================================================
MSCOCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
# URL template — plug split ∈ {train2017, val2017, test2017} and image_id.
MSCOCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{image_id:012d}.jpg"
MSCOCO_INDEX_DIR = PROCESSED_DIR / "mscoco"
MSCOCO_DEFAULT_SPLIT = "val2017"   # val2017 is 5K images — enough for a coherent pool

# =============================================================================
# VisualCOMET (commonsense inferences: before/after/intent)
# =============================================================================
VISUALCOMET_ANNOTATIONS_URL = "https://visualcomet.xyz/data/visualcomet.zip"
VISUALCOMET_INDEX_DIR = PROCESSED_DIR / "visualcomet"

# =============================================================================
# Scene-graph module (GAT branch)
# =============================================================================
SCENE_GRAPH_DIR = PROCESSED_DIR / "scene_graphs"
SG_OBJECT_VOCAB_SIZE = 2000      # Truncate to the N most frequent object labels
SG_PREDICATE_VOCAB_SIZE = 500    # Truncate to the N most frequent predicates
SG_EMBED_DIM = 128               # Per-node/edge embedding dimension
GAT_HIDDEN_DIM = 256             # Hidden size inside the GAT branch
GAT_NUM_HEADS = 4                # Attention heads inside the GAT
GAT_NUM_LAYERS = 2               # Stacked GAT layers
SG_MAX_OBJECTS = 36              # Cap per-image scene graph size for batching

# =============================================================================
# Losses
# =============================================================================
LOC_LOSS_WEIGHT = 1.0            # Weight on soft-IoU localization loss
BCE_LOSS_WEIGHT = 1.0            # Weight on binary coherence BCE loss

# =============================================================================
# WandB
# =============================================================================
WANDB_PROJECT = "sceneiq"
WANDB_ENTITY = None  # set to your wandb username/team or leave None

# =============================================================================
# Ensure critical directories exist at import time
# =============================================================================
for _dir in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, THUMBNAILS_DIR,
             MODELS_DIR, CONFIGS_DIR, EVALUATION_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
