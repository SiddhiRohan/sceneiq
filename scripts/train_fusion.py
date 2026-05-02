"""
SceneIQ — Phase 2 unified training script.

Trains one of three model variants for the ablation study:

    --model vit     ViT-only classifier with regularization (augmentation,
                    dropout, early stopping, LR scheduler).
    --model gat     GAT-only classifier using scene-graph features.
    --model fusion  Full ViT + GAT cross-attention fusion model.

Reads the same splits as Phase 1 from ``data/processed/splits/``.
Scene-graph inputs are loaded from ``data/processed/scene_graphs/``.

Checkpoints are saved to ``models/{model_type}/best.pt`` to keep
Phase 1 checkpoints intact.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    CLASSIFIER_DROPOUT,
    CROSS_ATTN_DIM,
    EARLY_STOPPING_PATIENCE,
    GAT_DROPOUT,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SCENE_GRAPHS_DIR,
    VIT_MODEL_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from utils import load_json, normalise_name, save_json, set_seed, setup_logging, timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FusionModel, GATClassifier

logger = logging.getLogger("sceneiq")


# ── Augmentation ────────────────────────────────────────────────────────────

def get_augmentation_pipeline() -> A.Compose:
    """Return the Albumentations augmentation pipeline.

    Returns:
        An ``A.Compose`` pipeline for training-time augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16),
                        hole_width_range=(8, 16), p=0.2),
    ])


# ── Scene graph helpers ─────────────────────────────────────────────────────

def inject_alien_node(
    graph: dict,
    alien_name: str,
    name_to_idx: dict,
    rng: random.Random,
) -> dict:
    """Add an alien object node to a copy of the scene graph.

    Connects the alien node to a randomly chosen existing node with the
    special ``alien_in_scene`` edge (predicate index 1).

    Args:
        graph: Compact graph dict from the graph index.
        alien_name: Name of the alien object category.
        name_to_idx: Object name vocabulary.
        rng: Random number generator for reproducibility.

    Returns:
        New graph dict with the alien node appended.
    """
    node_labels = list(graph["node_labels"])
    src_list = list(graph["edge_index"][0])
    dst_list = list(graph["edge_index"][1])
    edge_labels = list(graph["edge_labels"])

    alien_idx = name_to_idx.get(normalise_name(alien_name), 0)
    new_node_pos = len(node_labels)
    node_labels.append(alien_idx)

    if new_node_pos > 0:
        anchor = rng.randint(0, new_node_pos - 1)
        src_list.extend([new_node_pos, anchor])
        dst_list.extend([anchor, new_node_pos])
        edge_labels.extend([1, 1])  # alien_in_scene predicate

    return {
        "node_labels": node_labels,
        "edge_index": [src_list, dst_list],
        "edge_labels": edge_labels,
        "num_nodes": len(node_labels),
    }


DUMMY_GRAPH = {
    "node_labels": [0],
    "edge_index": [[], []],
    "edge_labels": [],
    "num_nodes": 1,
}


# ── Dataset ─────────────────────────────────────────────────────────────────

class SceneIQFusionDataset(Dataset):
    """Loads (image, scene graph, label) triples from a split manifest.

    For coherent images, looks up the VG scene graph by ``image_id``.
    For incoherent images, looks up the base scene graph via
    ``scene_image_id`` and injects the alien object node.

    Args:
        split_path: Path to a split JSON file.
        processor: HuggingFace ViTImageProcessor.
        graph_index: Dict mapping ``str(image_id)`` to compact graph dicts.
        name_to_idx: Object name vocabulary.
        augment: Whether to apply data augmentation.
        need_images: If False, skip image loading (for GAT-only mode).
    """

    def __init__(
        self,
        split_path: Path,
        processor: ViTImageProcessor | None,
        graph_index: dict,
        name_to_idx: dict,
        augment: bool = False,
        need_images: bool = True,
    ):
        self.records = load_json(split_path)
        self.processor = processor
        self.graph_index = graph_index
        self.name_to_idx = name_to_idx
        self.need_images = need_images
        self.aug = get_augmentation_pipeline() if augment else None
        self.rng = random.Random(42)
        logger.info(
            "FusionDataset %s: %d records (augment=%s, images=%s)",
            split_path.name, len(self.records), augment, need_images,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        label = rec["label"]
        result = {"label": torch.tensor(label, dtype=torch.long)}

        # Image
        if self.need_images:
            img = Image.open(rec["image_path"]).convert("RGB")
            if self.aug is not None:
                img_np = np.array(img)
                img_np = self.aug(image=img_np)["image"]
                img = Image.fromarray(img_np)
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
            result["pixel_values"] = pixel_values

        # Scene graph
        if label == 0:
            image_id = str(rec.get("image_id", ""))
            graph = self.graph_index.get(image_id, DUMMY_GRAPH)
        else:
            scene_id = str(rec.get("scene_image_id", ""))
            base_graph = self.graph_index.get(scene_id, DUMMY_GRAPH)
            alien_name = rec.get("alien_object", "")
            graph = inject_alien_node(base_graph, alien_name, self.name_to_idx, self.rng)

        result["node_labels"] = torch.tensor(graph["node_labels"], dtype=torch.long)
        edge_index = graph["edge_index"]
        if edge_index[0]:
            result["edge_index"] = torch.tensor(edge_index, dtype=torch.long)
        else:
            result["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        result["num_nodes"] = graph["num_nodes"]

        return result


def collate_fusion(batch: list) -> dict:
    """Collate function that batches images and graphs together.

    Args:
        batch: List of sample dicts from SceneIQFusionDataset.

    Returns:
        Batched dict with stacked tensors and a PyG Batch for graphs.
    """
    has_images = "pixel_values" in batch[0]
    labels = torch.stack([b["label"] for b in batch])

    result = {"labels": labels}

    if has_images:
        result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

    # Build PyG batch
    data_list = []
    node_counts = []
    for b in batch:
        data_list.append(Data(
            x=b["node_labels"],
            edge_index=b["edge_index"],
        ))
        node_counts.append(b["num_nodes"])

    graph_batch = Batch.from_data_list(data_list)
    result["graph_batch"] = graph_batch
    result["node_counts"] = node_counts

    return result


# ── Train / eval loops ──────────────────────────────────────────────────────

def run_epoch_vit(model, loader, device, optimizer, desc):
    """Train/eval loop for the ViT-only model.

    Args:
        model: ViTForImageClassification.
        loader: DataLoader.
        device: Target device.
        optimizer: Optimizer (None for eval).
        desc: Progress bar label.

    Returns:
        Dict with loss, accuracy, f1.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    n = len(all_labels)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds) if n else 0.0,
        "f1": f1_score(all_labels, all_preds, average="binary") if n else 0.0,
    }


def run_epoch_gat(model, loader, device, optimizer, desc):
    """Train/eval loop for the GAT-only model.

    Args:
        model: GATClassifier.
        loader: DataLoader.
        device: Target device.
        optimizer: Optimizer (None for eval).
        desc: Progress bar label.

    Returns:
        Dict with loss, accuracy, f1.
    """
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            labels = batch["labels"].to(device, non_blocking=True)
            gb = batch["graph_batch"].to(device)

            logits = model(gb.x, gb.edge_index, gb.batch)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    n = len(all_labels)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds) if n else 0.0,
        "f1": f1_score(all_labels, all_preds, average="binary") if n else 0.0,
    }


def run_epoch_fusion(model, loader, device, optimizer, desc):
    """Train/eval loop for the fusion model.

    Args:
        model: FusionModel.
        loader: DataLoader.
        device: Target device.
        optimizer: Optimizer (None for eval).
        desc: Progress bar label.

    Returns:
        Dict with loss, accuracy, f1.
    """
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            gb = batch["graph_batch"].to(device)
            node_counts = batch["node_counts"]

            out = model(pv, gb.x, gb.edge_index, gb.batch, node_counts)
            loss = criterion(out["logits"], labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = out["logits"].argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    n = len(all_labels)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds) if n else 0.0,
        "f1": f1_score(all_labels, all_preds, average="binary") if n else 0.0,
    }


# ── Main ────────────────────────────────────────────────────────────────────

@timer
def main(
    model_type: str,
    splits_dir: Path,
    models_dir: Path,
    graphs_dir: Path,
    model_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    num_workers: int,
    seed: int,
    augment: bool,
    patience: int,
    dropout: float,
    use_wandb: bool,
    wandb_run_name: str | None,
) -> None:
    """Train a model variant for the ablation study.

    Args:
        model_type: One of ``vit``, ``gat``, ``fusion``.
        splits_dir: Directory containing train/val/test JSON splits.
        models_dir: Base directory for checkpoints.
        graphs_dir: Directory with scene graph index and vocabulary.
        model_name: HuggingFace ViT model ID.
        batch_size: Batch size.
        learning_rate: AdamW learning rate.
        num_epochs: Maximum training epochs.
        num_workers: DataLoader workers.
        seed: Random seed.
        augment: Whether to apply augmentation.
        patience: Early stopping patience.
        dropout: Dropout probability.
        use_wandb: Log to Weights & Biases.
        wandb_run_name: Optional W&B run name.
    """
    splits_dir = Path(splits_dir)
    out_dir = Path(models_dir) / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Model type: %s", device, model_type)

    # WandB
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=wandb_run_name or f"phase2-{model_type}",
            config={
                "model_type": model_type, "model_name": model_name,
                "batch_size": batch_size, "learning_rate": learning_rate,
                "num_epochs": num_epochs, "augment": augment,
                "patience": patience, "dropout": dropout, "seed": seed,
            },
        )

    # Load scene graph data
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading data")
    logger.info("=" * 60)

    need_graphs = model_type in ("gat", "fusion")
    need_images = model_type in ("vit", "fusion")

    graph_index, name_to_idx, num_categories, num_predicates = {}, {}, 1, 2
    if need_graphs:
        vocab = load_json(Path(graphs_dir) / "vocab.json")
        name_to_idx = vocab["name_to_idx"]
        predicate_to_idx = vocab["predicate_to_idx"]
        num_categories = len(name_to_idx)
        num_predicates = len(predicate_to_idx)
        logger.info("Loading graph index...")
        graph_index = load_json(Path(graphs_dir) / "graph_index.json")
        logger.info("Graph index: %d entries", len(graph_index))

    processor = None
    if need_images:
        processor = ViTImageProcessor.from_pretrained(model_name)

    # Datasets
    ds_kwargs = dict(
        processor=processor, graph_index=graph_index,
        name_to_idx=name_to_idx, need_images=need_images,
    )
    train_ds = SceneIQFusionDataset(
        splits_dir / "train.json", augment=augment, **ds_kwargs,
    )
    val_ds = SceneIQFusionDataset(splits_dir / "val.json", **ds_kwargs)
    test_ds = SceneIQFusionDataset(splits_dir / "test.json", **ds_kwargs)

    loader_kw = dict(
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fusion, pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    # Model
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Building model: %s", model_type)
    logger.info("=" * 60)

    if model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=2,
            id2label={0: "coherent", 1: "incoherent"},
            label2id={"coherent": 0, "incoherent": 1},
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        ).to(device)
        run_fn = run_epoch_vit
    elif model_type == "gat":
        model = GATClassifier(
            num_categories=num_categories, num_predicates=num_predicates,
            hidden_dim=GAT_HIDDEN_DIM, num_heads=GAT_NUM_HEADS,
            num_layers=GAT_NUM_LAYERS, dropout=dropout,
        ).to(device)
        run_fn = run_epoch_gat
    else:
        model = FusionModel(
            vit_model_name=model_name,
            num_categories=num_categories, num_predicates=num_predicates,
            gat_hidden=GAT_HIDDEN_DIM, gat_heads=GAT_NUM_HEADS,
            gat_layers=GAT_NUM_LAYERS, fusion_dim=CROSS_ATTN_DIM,
            fusion_heads=4, dropout=dropout,
        ).to(device)
        run_fn = run_epoch_fusion

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Training for up to %d epochs (patience=%d)", num_epochs, patience)
    logger.info("=" * 60)
    best_val_acc = -1.0
    patience_counter = 0
    history: list = []

    for epoch in range(1, num_epochs + 1):
        logger.info("Epoch %d/%d (lr=%.2e)", epoch, num_epochs, scheduler.get_last_lr()[0])

        train_metrics = run_fn(model, train_loader, device, optimizer, f"train e{epoch}")
        val_metrics = run_fn(model, val_loader, device, None, f"val e{epoch}")
        scheduler.step()

        logger.info(
            "Epoch %d | train loss=%.4f acc=%.4f f1=%.4f | val loss=%.4f acc=%.4f f1=%.4f",
            epoch,
            train_metrics["loss"], train_metrics["accuracy"], train_metrics["f1"],
            val_metrics["loss"], val_metrics["accuracy"], val_metrics["f1"],
        )

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if run is not None:
            run.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info("New best checkpoint (val acc=%.4f)", best_val_acc)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        save_json(history, out_dir / "history.json")

    torch.save(model.state_dict(), out_dir / "last.pt")
    save_json(history, out_dir / "history.json")

    # Test evaluation with best checkpoint
    logger.info("Loading best checkpoint for test evaluation")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_metrics = run_fn(model, test_loader, device, None, "test")
    save_json(test_metrics, out_dir / "test_metrics.json")

    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()})
        run.finish()

    print("\n" + "=" * 60)
    print(f"  SceneIQ — Training Summary ({model_type})")
    print("=" * 60)
    print(f"  Model type:            {model_type}")
    print(f"  Epochs completed:      {len(history)}/{num_epochs}")
    print(f"  Best val accuracy:     {best_val_acc:.4f}")
    print(f"  Test accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:               {test_metrics['f1']:.4f}")
    print(f"  Augmentation:          {augment}")
    print(f"  Checkpoints:           {out_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train SceneIQ model variants for Phase 2 ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="fusion",
                        choices=["vit", "gat", "fusion"])
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--graphs-dir", type=str, default=str(SCENE_GRAPHS_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--dropout", type=float, default=CLASSIFIER_DROPOUT)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=f"train_{args.model}.log")
    main(
        model_type=args.model,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        graphs_dir=args.graphs_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
        patience=args.patience,
        dropout=args.dropout,
        use_wandb=args.wandb,
        wandb_run_name=args.wandb_run_name,
    )
