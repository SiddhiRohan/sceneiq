"""
SceneIQ — Train the ViT-based coherence classifier.

Fine-tunes a pretrained Vision Transformer (``google/vit-base-patch16-224`` by
default) on the binary coherence task:

    label 0 — coherent  (real photo)
    label 1 — incoherent (synthetic photo with an alien object pasted in)

Reads the splits written by ``prepare_dataset.py`` from
``data/processed/splits/{train,val,test}.json``. Each epoch logs train/val
loss and accuracy; the best checkpoint by val accuracy is saved to
``models/best.pt``, and the final epoch is saved to ``models/last.pt``.

Metrics are logged to the console, to a file, and (optionally) to
Weights & Biases.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    VIT_MODEL_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from utils import load_json, save_json, set_seed, setup_logging, timer

logger = logging.getLogger("sceneiq")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SceneIQDataset(Dataset):
    """Loads (image, label) pairs from a split manifest.

    Args:
        split_path: Path to a ``train.json`` / ``val.json`` / ``test.json``
            produced by ``prepare_dataset.py``.
        processor: HF ``ViTImageProcessor`` used to resize/normalise images.
    """

    def __init__(self, split_path: Path, processor: ViTImageProcessor):
        self.records = load_json(split_path)
        self.processor = processor
        logger.info("Dataset %s: %d records", split_path.name, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, image_path: str) -> Path:
        """Resolve a potentially relative image path against project root."""
        p = Path(image_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(self._resolve_path(rec["image_path"])).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }


def collate(batch: list) -> dict:
    """Stack a list of dataset items into a batched dict."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


# ── Train / eval loops ───────────────────────────────────────────────────────

def run_epoch(
    model: ViTForImageClassification,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict:
    """Run one pass over ``loader``; train when ``optimizer`` is given, else eval.

    Args:
        model: ViT classifier.
        loader: DataLoader for the current split.
        device: Target device.
        optimizer: Optimiser (``None`` for eval mode).
        desc: Label for the progress bar and logs.

    Returns:
        Dict with keys ``loss``, ``accuracy``, ``f1``.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values, labels=labels)
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


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    splits_dir: Path,
    models_dir: Path,
    model_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    num_workers: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
) -> None:
    """Fine-tune a ViT on the SceneIQ coherence task.

    Args:
        splits_dir: Directory containing ``train.json``/``val.json``/``test.json``.
        models_dir: Directory to write checkpoints into.
        model_name: HF model id of the pretrained ViT.
        batch_size: Batch size for all loaders.
        learning_rate: AdamW learning rate.
        num_epochs: Training epochs.
        num_workers: DataLoader worker processes.
        seed: Random seed.
        use_wandb: If True, log metrics to Weights & Biases.
        wandb_run_name: Optional explicit run name.
    """
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # WandB (optional)
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "seed": seed,
            },
        )

    # Data
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading data")
    logger.info("=" * 60)
    processor = ViTImageProcessor.from_pretrained(model_name)
    train_ds = SceneIQDataset(splits_dir / "train.json", processor)
    val_ds = SceneIQDataset(splits_dir / "val.json", processor)
    test_ds = SceneIQDataset(splits_dir / "test.json", processor)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Model
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Loading model %s", model_name)
    logger.info("=" * 60)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "coherent", 1: "incoherent"},
        label2id={"coherent": 0, "incoherent": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Training for %d epochs", num_epochs)
    logger.info("=" * 60)
    best_val_acc = -1.0
    history: list = []
    for epoch in range(1, num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, num_epochs)
        train_metrics = run_epoch(model, train_loader, device, optimizer, f"train e{epoch}")
        val_metrics = run_epoch(model, val_loader, device, None, f"val e{epoch}")

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
            torch.save(model.state_dict(), models_dir / "best.pt")
            logger.info("Saved new best checkpoint (val acc=%.4f)", best_val_acc)

        save_json(history, models_dir / "history.json")

    torch.save(model.state_dict(), models_dir / "last.pt")

    # Final test evaluation using the best checkpoint
    logger.info("Loading best.pt for final test evaluation")
    model.load_state_dict(torch.load(models_dir / "best.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, device, None, "test")
    save_json(test_metrics, models_dir / "test_metrics.json")
    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()})
        run.finish()

    print("\n" + "=" * 60)
    print("  SceneIQ — Training Summary")
    print("=" * 60)
    print(f"  Model:                 {model_name}")
    print(f"  Epochs:                {num_epochs}")
    print(f"  Best val accuracy:     {best_val_acc:.4f}")
    print(f"  Test accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"  Test F1:               {test_metrics['f1']:.4f}")
    print(f"  Best checkpoint:       {models_dir / 'best.pt'}")
    print(f"  Last checkpoint:       {models_dir / 'last.pt'}")
    print(f"  History:               {models_dir / 'history.json'}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the SceneIQ ViT coherence classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="train.log")
    main(
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_run_name=args.wandb_run_name,
    )
