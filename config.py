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
MODELS_DIR = PROJECT_ROOT / "models"
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

# =============================================================================
# Phase 2 — Scene-graph fusion
# =============================================================================
SCENE_GRAPHS_DIR = PROCESSED_DIR / "scene_graphs"
GAT_HIDDEN_DIM = 128
GAT_NUM_HEADS = 4
GAT_NUM_LAYERS = 2
GAT_DROPOUT = 0.3
CROSS_ATTN_DIM = 256
CLASSIFIER_DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 3

# =============================================================================
# WandB
# =============================================================================
WANDB_PROJECT = "sceneiq"
WANDB_ENTITY = None  # set to your wandb username/team or leave None

# =============================================================================
# Ensure critical directories exist at import time
# =============================================================================
for _dir in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, THUMBNAILS_DIR,
             MODELS_DIR, CONFIGS_DIR, EVALUATION_DIR, LOGS_DIR,
             SCENE_GRAPHS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
