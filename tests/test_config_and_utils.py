"""
Tests for config.py and utils.py.

Covers: constant presence and types, path objects, loss weights, random-seed
determinism, JSON round-trip, normalization helpers, and the @timer decorator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import torch

import config
import utils


# ── Config: directories & types ──────────────────────────────────────────────

def test_config_project_root_is_path() -> None:
    assert isinstance(config.PROJECT_ROOT, Path)


def test_config_data_dir_under_project_root() -> None:
    assert config.DATA_DIR.parent == config.PROJECT_ROOT


def test_config_raw_dir_under_data_dir() -> None:
    assert config.RAW_DIR.parent == config.DATA_DIR


def test_config_models_dir_is_checkpoints() -> None:
    """MODELS_DIR must point at checkpoints/, not the Python package dir."""
    assert config.MODELS_DIR.name == "checkpoints"


def test_config_evaluation_dir_is_path() -> None:
    assert isinstance(config.EVALUATION_DIR, Path)


def test_config_scene_graph_dir_under_processed() -> None:
    assert config.SCENE_GRAPH_DIR.parent == config.PROCESSED_DIR


# ── Config: mode constants ───────────────────────────────────────────────────

def test_mode_vit_string() -> None:
    assert config.MODE_VIT == "vit"


def test_mode_gat_string() -> None:
    assert config.MODE_GAT == "gat"


def test_mode_fusion_string() -> None:
    assert config.MODE_FUSION == "fusion"


def test_model_modes_tuple_matches_constants() -> None:
    assert config.MODEL_MODES == (config.MODE_VIT, config.MODE_GAT, config.MODE_FUSION)


def test_model_modes_unique() -> None:
    assert len(set(config.MODEL_MODES)) == 3


# ── Config: numeric sanity ───────────────────────────────────────────────────

def test_patch_grid_matches_vit_b16() -> None:
    """ViT-B/16 at 224 has a 14×14 patch grid. If config drifts, training breaks."""
    assert config.PATCH_GRID == 14
    assert config.IMAGE_SIZE == (224, 224)
    assert 224 // 16 == config.PATCH_GRID


def test_loss_weights_are_non_negative() -> None:
    assert config.BCE_LOSS_WEIGHT >= 0
    assert config.LOC_LOSS_WEIGHT >= 0


def test_loss_weights_sum_nonzero() -> None:
    """At least one loss term must be active, otherwise training is a no-op."""
    assert (config.BCE_LOSS_WEIGHT + config.LOC_LOSS_WEIGHT) > 0


def test_gat_hidden_dim_divisible_by_num_heads() -> None:
    """Pytorch MultiheadAttention requires hidden_dim % num_heads == 0."""
    assert config.GAT_HIDDEN_DIM % config.GAT_NUM_HEADS == 0


def test_gat_num_layers_positive() -> None:
    assert config.GAT_NUM_LAYERS >= 1


def test_sg_vocab_sizes_positive() -> None:
    assert config.SG_OBJECT_VOCAB_SIZE > 2   # needs room beyond PAD + UNK
    assert config.SG_PREDICATE_VOCAB_SIZE > 2


def test_sg_max_objects_positive() -> None:
    assert config.SG_MAX_OBJECTS >= 1


def test_batch_size_positive() -> None:
    assert config.BATCH_SIZE > 0


def test_learning_rate_positive_and_small() -> None:
    assert 0 < config.LEARNING_RATE < 1.0


def test_num_epochs_positive() -> None:
    assert config.NUM_EPOCHS >= 1


def test_random_seed_is_int() -> None:
    assert isinstance(config.RANDOM_SEED, int)


def test_vit_model_name_is_hf_style() -> None:
    """google/vit-* is the HF org/model convention."""
    assert "/" in config.VIT_MODEL_NAME
    assert config.VIT_MODEL_NAME.startswith("google/vit")


def test_dataset_sizes_dict_has_expected_keys() -> None:
    assert set(config.DATASET_SIZES.keys()) == {"visual_genome", "mscoco", "visualcomet"}


# ── utils.set_seed — determinism ─────────────────────────────────────────────

def test_set_seed_numpy_deterministic() -> None:
    utils.set_seed(123)
    a = np.random.rand(3)
    utils.set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_set_seed_torch_deterministic() -> None:
    utils.set_seed(7)
    a = torch.randn(4)
    utils.set_seed(7)
    b = torch.randn(4)
    assert torch.allclose(a, b)


def test_set_seed_different_seeds_differ() -> None:
    utils.set_seed(1)
    a = torch.randn(4)
    utils.set_seed(2)
    b = torch.randn(4)
    assert not torch.allclose(a, b)


# ── utils.save_json / load_json ──────────────────────────────────────────────

def test_json_roundtrip_primitive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.json"
        utils.save_json({"a": 1, "b": [1, 2, 3]}, p)
        back = utils.load_json(p)
        assert back == {"a": 1, "b": [1, 2, 3]}


def test_json_roundtrip_nested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.json"
        data = {"outer": {"inner": [{"k": "v"}]}}
        utils.save_json(data, p)
        assert utils.load_json(p) == data


def test_json_roundtrip_unicode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.json"
        utils.save_json({"name": "café"}, p)
        assert utils.load_json(p) == {"name": "café"}


def test_save_json_creates_parent_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "deeply" / "nested" / "x.json"
        utils.save_json([1, 2], nested)
        assert nested.exists()


def test_load_json_missing_raises() -> None:
    try:
        utils.load_json("/does/not/exist.json")
    except FileNotFoundError:
        return
    raise AssertionError("load_json should raise FileNotFoundError for missing path")


def test_load_json_malformed_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.json"
        p.write_text("not valid json {")
        try:
            utils.load_json(p)
        except json.JSONDecodeError:
            return
        raise AssertionError("load_json should raise JSONDecodeError for malformed JSON")


# ── utils.normalise_name ─────────────────────────────────────────────────────

def test_normalise_name_lowercases() -> None:
    assert utils.normalise_name("DOG") == "dog"


def test_normalise_name_strips_whitespace() -> None:
    assert utils.normalise_name("  cat  ") == "cat"


def test_normalise_name_empty_string() -> None:
    assert utils.normalise_name("") == ""


def test_normalise_name_none() -> None:
    assert utils.normalise_name(None) == ""


def test_normalise_name_mixed_case_and_whitespace() -> None:
    assert utils.normalise_name("  Traffic Light ") == "traffic light"


# ── utils.extract_object_name ────────────────────────────────────────────────

def test_extract_object_name_prefers_names_list() -> None:
    assert utils.extract_object_name({"names": ["Dog", "K9"], "name": "Ignored"}) == "dog"


def test_extract_object_name_falls_back_to_name() -> None:
    assert utils.extract_object_name({"name": "Cat"}) == "cat"


def test_extract_object_name_empty_dict_returns_empty() -> None:
    assert utils.extract_object_name({}) == ""


def test_extract_object_name_empty_names_list_falls_back() -> None:
    assert utils.extract_object_name({"names": [], "name": "ball"}) == "ball"


def test_extract_object_name_strips_and_lowercases() -> None:
    assert utils.extract_object_name({"names": ["  Tree  "]}) == "tree"


# ── utils.timer decorator ────────────────────────────────────────────────────

def test_timer_returns_original_value() -> None:
    @utils.timer
    def add(a, b):
        return a + b
    assert add(2, 3) == 5


def test_timer_preserves_name_via_functools_wraps() -> None:
    @utils.timer
    def my_function():
        return 1
    assert my_function.__name__ == "my_function"


def test_timer_passes_kwargs() -> None:
    @utils.timer
    def greet(name="world"):
        return f"hi {name}"
    assert greet(name="test") == "hi test"


# ── utils.setup_logging ──────────────────────────────────────────────────────

def test_setup_logging_returns_logger() -> None:
    logger = utils.setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "sceneiq"
