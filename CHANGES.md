# CHANGES — SceneIQ scaffold → full fusion + localization pipeline

This document lists every modification against the original
`sceneiq-main.zip` upload. Numbers are lines of code (LoC) counted by
`wc -l`; "±N" indicates net change against the original.

---

## 1. Proposal coverage — before vs. after

Mapping the proposal's "What We Will Build" list to code:

| Proposal item                                         | Original          | Now       |
|-------------------------------------------------------|-------------------|-----------|
| Inconsistency mining pipeline (co-occurrence)         | Done              | Done      |
| Copy-paste augmentation pipeline                      | Done              | Done      |
| Dataset splits with leakage exclusion                 | Done              | Done      |
| ViT feature extraction                                | Done (HF default) | **Enhanced** — replaced HF head with task-specific heads |
| Scene-graph generation / integration (GAT)            | **Missing**       | **Done**  (VG ground-truth graphs + PyG GAT with attention fallback) |
| Cross-attention fusion (ViT + GAT)                    | **Missing**       | **Done**  |
| Coherence head (SCS scalar)                           | Partial (2-class) | **Done**  (sigmoid scalar, proposal Eq. 1) |
| Localization heatmap                                  | **Missing**       | **Done**  (14×14 per-patch, soft-IoU loss) |
| Evaluation framework + SCS metric                     | Classification only | **Done** (+ pointing accuracy, mean soft-IoU) |
| Experiment 1 — binary coherence classification        | Done              | Done      |
| Experiment 2 — inconsistency localization             | **Missing**       | **Done**  |
| Experiment 3 — ablation (vit / gat / fusion)          | **Missing**       | **Done**  |
| Qualitative SCS demonstration                         | **Missing**       | **Done**  (scripts/demo.py) |
| MS-COCO data source                                   | Missing           | **Done**  (downloader, per-image index, optional coherent pool via `--coco-index`) |
| VisualCOMET data source                               | Missing           | **Done**  (downloader, per-image commonsense index) |

**Summary.** Every proposal item — core and auxiliary — now has working
code. MS-COCO and VisualCOMET integrations each ship a downloader, a
preprocessing script producing a per-image JSON index, and full test
coverage on the preprocessing.

---

## 2. Success criteria — code paths now in place

The proposal lists five success criteria; at least three must be met.

| Criterion                                                       | Code path? |
|-----------------------------------------------------------------|------------|
| Binary coherence F1 ≥ 0.75 on test set                          | Yes — `scripts/evaluate.py` computes it for every mode |
| Localization pointing accuracy ≥ 0.45                           | Yes — `compute_localization_metrics` in `scripts/evaluate.py` |
| Fusion model outperforms ViT-only and GAT-only by ≥ 3 % F1      | Yes — `scripts/run_ablation.py` produces the three-way comparison |
| Qualitative SCS demo on diverse real-world images               | Yes — `scripts/demo.py --image ... --mode fusion` |
| Complete ablation with clear analysis of failures               | Yes — per-alien breakdown JSON + heatmap overlays per mode |

All five are structurally possible now. Hitting the numeric thresholds
depends on training runs, not on code that still needs to be written.

---

## 3. File-by-file change log

### 3.1 Original files kept unchanged

| File                              | LoC | Notes |
|-----------------------------------|----:|-------|
| `scripts/download_vg.py`          | 285 | Visual Genome downloader |
| `scripts/build_co_occurrence.py`  | 263 | Object / pair / predicate / triplet counts |
| `scripts/generate_synthetic.py`   | 455 | Copy-paste augmentation — already recorded `paste_bbox` |
| `utils.py`                        | 167 | Logging, seeding, JSON I/O, timer, VG helpers |
| `requirements.txt`                |  26 | Unchanged |

### 3.2 Modified originals

| File                    | Before | After | Δ    | What changed |
|-------------------------|-------:|------:|-----:|--------------|
| `config.py`             |     79 |   121 | +42  | Added mode constants, patch grid, GAT hyperparams (hidden dim / num heads / num layers), loss weights, scene-graph paths. Added MS-COCO + VisualCOMET URL / index constants. Renamed `MODELS_DIR` from `models/` to `checkpoints/` so `models/` can be the Python package. |
| `.gitignore`            |     55 |    59 | +4   | Added `checkpoints/` and `evaluation/` to the ignored set. |
| `scripts/prepare_dataset.py` |  257 | 411 | +154 | Propagates `paste_bbox` and `scene_image_id` through every split record so the training loop can build the 14×14 target mask and look up scene graphs. Coherent records now carry `scene_image_id = image_id`. Added `--coco-index` / `--coco-fraction` flags so the coherent pool can be mixed from VG and MS-COCO. `load_coco_records` handles missing files gracefully (warn, return `[]`). |
| `scripts/train.py`      |    273 |   329 | +56  | Rewritten. Drops `ViTForImageClassification`. Supports `--mode vit|gat|fusion`. Uses `build_model` factory. Combined BCE + soft-IoU loss (gated per-sample for localization). Per-mode checkpoint directory. Logs `bce` / `loc` components separately. |
| `scripts/evaluate.py`   |    348 |   417 | +69  | Rewritten. Adds pointing-accuracy and mean soft-IoU (`compute_localization_metrics`). Emits `heatmaps/` overlays for the top-confidence incoherent predictions. Handles all three modes through the unified `SceneIQOutput`. |
| `README.md`             |     95 |   128 | +33  | Rewritten. Documents modes, ablation flow, output structure, and design decisions. |

### 3.3 New files

| File                            | LoC | Role |
|---------------------------------|----:|------|
| `models/__init__.py`            |  19 | Public exports (`build_model`, branches, `SceneIQOutput`) |
| `models/vit_branch.py`          |  91 | ViT + SCS head + per-patch localization head (`SceneIQViT`, `ViTOutput`) |
| `models/gat_branch.py`          | 230 | GAT encoder over scene-graph triplets (`SceneIQGAT`, `GATOutput`). PyTorch Geometric when available, multi-head self-attention fallback otherwise. `collate_graphs` classmethod packs per-image graphs into batched tensors with correct padding + edge offsetting. |
| `models/fusion.py`              | 300 | Cross-attention fusion (`SceneIQFusion`), mode-specific wrappers (`_ViTOnly`, `_GATOnly`) that share a single `SceneIQOutput` interface, `build_model` factory, `soft_iou_loss`, and `compute_loss`. |
| `scripts/build_scene_graphs.py` | 226 | Preprocesses VG `scene_graphs.json` into `object_vocab.json`, `predicate_vocab.json`, `graphs.json` for the GAT branch. Uses ground-truth VG graphs (bit-for-bit reproducible; RelTR/EGTR is a drop-in replacement later). |
| `scripts/download_coco.py`         | 120 | Downloads and extracts the ~240 MB `annotations_trainval2017.zip` into `data/raw/mscoco/`. |
| `scripts/build_coco_index.py`      | 175 | Parses COCO `instances` + `captions` annotations into a per-image coherent-record index + COCO-side pair counts. |
| `scripts/download_visualcomet.py`  | 115 | Downloads and extracts the VisualCOMET commonsense bundle into `data/raw/visualcomet/`. Supports `--url` override for host changes. |
| `scripts/build_visualcomet_index.py` | 170 | Aggregates VisualCOMET `{train,val,test}_annots.json` into a single per-image `{events, places, intent, before, after, num_records}` dict with dedup + normalisation. |
| `scripts/data.py`               | 151 | Shared `SceneIQDataset` + `make_collate`. Computes the 14×14 patch-coverage mask from `paste_bbox` via `bbox_to_patch_mask` (area-weighted — preserves bbox fractional area in the mask sum). |
| `scripts/run_ablation.py`       | 174 | Orchestrates train + evaluate across `{vit, gat, fusion}` and emits `ablation_summary.json` + a comparison bar chart. |
| `scripts/demo.py`               | 170 | Qualitative single-image inference. Prints SCS and writes an image / heatmap / overlay visualization. |

### 3.4 Test infrastructure (all new)

| File                                | Tests | Role |
|-------------------------------------|------:|------|
| `tests/__init__.py`                 |     – | Marks the dir as a package |
| `tests/conftest.py`                 |     – | Shared fakes: `FakeBackbone`, `FakeProcessor`, `install_fake_vit`, `make_small_gat`, `write_grey_jpeg` |
| `tests/test_smoke.py`               |    22 | End-to-end architecture smoke |
| `tests/test_config_and_utils.py`    |    47 | Constants, JSON round-trip, seed determinism, normalization helpers, @timer |
| `tests/test_bbox.py`                |    28 | Area conservation, clamping, grid sizes, aspect ratios, patch-boundary alignment |
| `tests/test_scene_graphs.py`        |    25 | Vocab builder, `_object_size`, `extract_graph` (happy path, variants, edge cases, SG_MAX_OBJECTS cap) |
| `tests/test_gat.py`                 |    21 | Collation batching, mask / edge-offset correctness, forward shape invariants, gradient flow |
| `tests/test_vit_fusion.py`          |    28 | ViT heads, `_ViTOnly` / `_GATOnly` wrappers, fusion shapes and backward, factory behaviour |
| `tests/test_losses.py`              |    21 | `soft_iou_loss` boundaries + monotonicity, `compute_loss` weight scaling / gating / gradient flow |
| `tests/test_dataset_collate.py`     |    25 | `SceneIQDataset` keys & shapes, bbox area preservation, `make_collate` with / without GAT |
| `tests/test_coco.py`                |    24 | COCO index building: category / image / caption helpers, pair counts, end-to-end `main()` on toy annotation blobs |
| `tests/test_visualcomet.py`         |    22 | VisualCOMET aggregation: dedup + strip + flatten, per-image grouping, `main()` on toy annotation files |
| `tests/test_prepare_dataset.py`     |    13 | `load_coco_records`, `load_incoherent_records` (bbox drop), `sample_coherent_records`, `split_records` (stratification + no overlap) |
| `tests/run_all.py`                  |     – | Pytest-free discovery + runner. Exits non-zero on any failure. |
| **Total tests**                     | **276** | |

`python tests/run_all.py` runs all 276 tests offline (no HuggingFace,
COCO, or VisualCOMET download required) and reports `276/276 passed` in
~5 seconds on CPU.

---

## 4. LoC totals

| Category          | Before | After |
|-------------------|-------:|------:|
| Production code   | 2,061  | 3,700 |
| Tests             |     0  | ~2,600 |
| **Total**         | 2,061  | ~6,300 |

(Production code count excludes `requirements.txt`, `README.md`, `CHANGES.md`.)

---

## 5. Architectural decisions worth highlighting

**Why VG ground-truth graphs, not RelTR/EGTR?** We're not studying scene
graph *generation* — we're using scene graphs as *input*. Visual Genome
already ships ground-truth graphs for ~108K images. Using them gives
bit-for-bit reproducibility and zero inference cost. Swapping in
RelTR/EGTR later is a drop-in replacement of `graphs.json`.

**Why soft-IoU, not per-pixel cross-entropy, on the heatmap?** IoU is
scale-invariant and differentiable (when computed on sigmoided
probabilities), and small bboxes don't drown under a pixel-weighted
BCE. Soft-IoU is also the metric of record for pointing-accuracy-style
tasks.

**Why record `paste_bbox` in synthetic metadata?** The moment we paste
the alien object, we already know exactly where the inconsistency is.
Propagating that into every incoherent split record turns a weakly
supervised localization problem into a supervised one for free.

**Why rename `MODELS_DIR` to `checkpoints/`?** The original used
`models/` as the checkpoint directory. Once we added a Python package
at `models/` the paths would collide. Renaming the runtime output dir
to `checkpoints/` is cleaner than renaming the package.

**Why a PyG-optional GAT?** `torch_geometric` wheels can be fiddly
(CUDA versions, source builds). The GAT branch falls back to a plain
multi-head self-attention encoder when PyG isn't importable — you lose
the explicit message-passing but training still works.

---

## 6. Migration guide (if you had data / weights from the original)

If you already ran scripts from the uploaded `sceneiq-main.zip`:

1. **Existing checkpoints** under `models/*.pt` should be moved to
   `checkpoints/vit/` to remain loadable by the rewritten
   `scripts/evaluate.py --mode vit --checkpoint <file>`. Note the saved
   format has changed — new checkpoints are dicts with `{"mode", "state_dict"}`.
2. **Existing synthetic metadata** already contains `paste_bbox` (the
   upload wrote it), so `scripts/prepare_dataset.py` works without
   regenerating images.
3. **Before running the fusion mode**, run
   `python scripts/build_scene_graphs.py` once to produce
   `data/processed/scene_graphs/{object_vocab,predicate_vocab,graphs}.json`.

---

## 7. One-command sanity check

```bash
pip install -r requirements.txt
python tests/run_all.py          # 276/276 should pass in a few seconds
```

If any test fails, that's the regression — fix it before proceeding.

---

## 8. MS-COCO / VisualCOMET quickstart

```bash
# MS-COCO — fetch annotations, build the index, mix 30% COCO into the coherent pool
python scripts/download_coco.py
python scripts/build_coco_index.py --split val2017
python scripts/prepare_dataset.py \
    --coco-index data/processed/mscoco/coherent_records_val2017.json \
    --coco-fraction 0.3

# VisualCOMET — fetch annotations and build the per-image commonsense index
python scripts/download_visualcomet.py
python scripts/build_visualcomet_index.py
# Downstream: load data/processed/visualcomet/commonsense.json keyed by img_fn
```

The VisualCOMET URL occasionally moves. If the default download fails, re-run
with `--url <current-url>` (see the VisualCOMET site for the latest).
