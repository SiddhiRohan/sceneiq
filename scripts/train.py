"""
SceneIQ — Train the coherence model (ViT / GAT / Fusion).

Supports three architectures via ``--mode``:

  * ``vit``    — ViT backbone + SCS head + per-patch localization head.
  * ``gat``    — GAT over VG scene graphs + SCS head. No localization.
  * ``fusion`` — ViT + GAT + cross-attention fusion + SCS + localization.

Losses:
  * Binary cross-entropy on the SCS logit.
  * Soft-IoU on the per-patch heatmap, applied only to incoherent samples
    (coherent samples have an empty target bbox).

Checkpoints are written to ``checkpoints/<mode>/`` — ``best.pt`` is saved at
the best val accuracy. Metrics history is saved as JSON alongside.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    BCE_LOSS_WEIGHT,
    GAT_HIDDEN_DIM,
    GAT_NUM_HEADS,
    GAT_NUM_LAYERS,
    LEARNING_RATE,
    LOC_LOSS_WEIGHT,
    MODE_VIT,
    MODEL_MODES,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    PATCH_GRID,
    PROCESSED_DIR,
    RANDOM_SEED,
    SCENE_GRAPH_DIR,
    SG_EMBED_DIM,
    SG_MAX_OBJECTS,
    VIT_MODEL_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from utils import load_json, save_json, set_seed, setup_logging, timer

# Local imports (scripts/ sits next to models/ at project root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SceneIQDataset, make_collate  # noqa: E402

from models import build_model                                  # noqa: E402
from models.fusion import compute_loss                          # noqa: E402

logger = logging.getLogger("sceneiq")


# ── Epoch loop ───────────────────────────────────────────────────────────────

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
    bce_weight: float,
    loc_weight: float,
) -> dict:
    """Run one pass over ``loader``; train when ``optimizer`` given, else eval.

    Returns:
        Dict with keys ``loss``, ``bce``, ``loc``, ``accuracy``, ``f1``.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_bce = total_loc = 0.0
    all_preds: list = []
    all_labels: list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            graph_batch = batch.get("graph_batch")
            if isinstance(graph_batch, dict):
                graph_batch = {k: v.to(device, non_blocking=True) for k, v in graph_batch.items()}

            output = model(pixel_values, graph_batch)

            losses = compute_loss(
                scs_logit=output.scs_logit,
                heatmap_logits=output.heatmap_logits,
                labels=labels,
                target_mask=target_mask,
                has_heatmap=output.has_heatmap,
                bce_weight=bce_weight,
                loc_weight=loc_weight,
            )

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            n = labels.size(0)
            total_loss += losses["total"].item() * n
            total_bce += losses["bce"].item() * n
            total_loc += losses["loc"].item() * n

            preds = (torch.sigmoid(output.scs_logit) > 0.5).long()
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    n_total = max(1, len(all_labels))
    return {
        "loss": total_loss / n_total,
        "bce": total_bce / n_total,
        "loc": total_loc / n_total,
        "accuracy": accuracy_score(all_labels, all_preds) if all_labels else 0.0,
        "f1": f1_score(all_labels, all_preds, average="binary") if all_labels else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(
    mode: str,
    splits_dir: Path,
    checkpoints_dir: Path,
    scene_graphs_dir: Path,
    model_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    num_workers: int,
    seed: int,
    bce_weight: float,
    loc_weight: float,
    use_wandb: bool,
    wandb_run_name: str | None,
) -> None:
    """Fine-tune the SceneIQ model for one of three modes."""
    if mode not in MODEL_MODES:
        raise ValueError(f"--mode must be one of {MODEL_MODES}, got {mode!r}")

    splits_dir = Path(splits_dir)
    checkpoints_dir = Path(checkpoints_dir) / mode
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scene_graphs_dir = Path(scene_graphs_dir)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | mode: %s", device, mode)

    # ── Scene-graph vocabs / graphs (only loaded when needed) ──
    need_graphs = mode in ("gat", "fusion")
    if need_graphs:
        obj_vocab = load_json(scene_graphs_dir / "object_vocab.json")
        pred_vocab = load_json(scene_graphs_dir / "predicate_vocab.json")
        graphs = load_json(scene_graphs_dir / "graphs.json")
        n_objects = len(obj_vocab)
        n_predicates = len(pred_vocab)
        logger.info("Scene graph vocab: %d objects, %d predicates, %d graphs",
                    n_objects, n_predicates, len(graphs))
    else:
        n_objects = n_predicates = 2  # placeholder; branch won't be built
        graphs = {}

    # ── Model ──
    model = build_model(
        mode=mode,
        vit_model_name=model_name,
        n_objects=n_objects,
        n_predicates=n_predicates,
        patch_grid=PATCH_GRID,
        sg_embed_dim=SG_EMBED_DIM,
        gat_hidden_dim=GAT_HIDDEN_DIM,
        gat_num_heads=GAT_NUM_HEADS,
        gat_num_layers=GAT_NUM_LAYERS,
    ).to(device)
    logger.info("Model parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # ── WandB ──
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=wandb_run_name or f"sceneiq-{mode}",
            config={
                "mode": mode, "model_name": model_name, "batch_size": batch_size,
                "learning_rate": learning_rate, "num_epochs": num_epochs, "seed": seed,
                "bce_weight": bce_weight, "loc_weight": loc_weight,
            },
        )

    # ── Data ──
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading data")
    logger.info("=" * 60)
    processor = ViTImageProcessor.from_pretrained(model_name)
    train_records = load_json(splits_dir / "train.json")
    val_records = load_json(splits_dir / "val.json")
    test_records = load_json(splits_dir / "test.json")
    train_ds = SceneIQDataset(train_records, processor, patch_grid=PATCH_GRID)
    val_ds = SceneIQDataset(val_records, processor, patch_grid=PATCH_GRID)
    test_ds = SceneIQDataset(test_records, processor, patch_grid=PATCH_GRID)
    logger.info("train=%d  val=%d  test=%d", len(train_ds), len(val_ds), len(test_ds))

    # The GAT module (if any) is nested inside the wrapper; fish it out for collate_graphs.
    gat_module = getattr(model, "gat", None)

    collate = make_collate(graphs, gat_module, device=None, sg_max_objects=SG_MAX_OBJECTS)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
                         pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ── Train ──
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Training (%s, %d epochs)", mode, num_epochs)
    logger.info("=" * 60)
    best_val_acc = -1.0
    history: list = []
    for epoch in range(1, num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, num_epochs)
        tr = run_epoch(model, train_loader, device, optimizer, f"train e{epoch}", bce_weight, loc_weight)
        va = run_epoch(model, val_loader, device, None, f"val e{epoch}", bce_weight, loc_weight)
        logger.info(
            "Epoch %d | train loss=%.4f (bce=%.4f loc=%.4f) acc=%.4f f1=%.4f | val loss=%.4f acc=%.4f f1=%.4f",
            epoch, tr["loss"], tr["bce"], tr["loc"], tr["accuracy"], tr["f1"],
            va["loss"], va["accuracy"], va["f1"],
        )
        record = {"epoch": epoch, "train": tr, "val": va}
        history.append(record)
        if run is not None:
            run.log({"epoch": epoch,
                     **{f"train/{k}": v for k, v in tr.items()},
                     **{f"val/{k}": v for k, v in va.items()}})

        if va["accuracy"] > best_val_acc:
            best_val_acc = va["accuracy"]
            torch.save({"mode": mode, "state_dict": model.state_dict()}, checkpoints_dir / "best.pt")
            logger.info("Saved new best checkpoint (val acc=%.4f)", best_val_acc)
        save_json(history, checkpoints_dir / "history.json")

    torch.save({"mode": mode, "state_dict": model.state_dict()}, checkpoints_dir / "last.pt")

    # ── Test ──
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Final test evaluation from best.pt")
    logger.info("=" * 60)
    ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te = run_epoch(model, test_loader, device, None, "test", bce_weight, loc_weight)
    save_json(te, checkpoints_dir / "test_metrics.json")
    if run is not None:
        run.log({f"test/{k}": v for k, v in te.items()})
        run.finish()

    print("\n" + "=" * 60)
    print(f"  SceneIQ — Training Summary  (mode={mode})")
    print("=" * 60)
    print(f"  Epochs:                {num_epochs}")
    print(f"  Best val accuracy:     {best_val_acc:.4f}")
    print(f"  Test accuracy:         {te['accuracy']:.4f}")
    print(f"  Test F1:               {te['f1']:.4f}")
    print(f"  Best checkpoint:       {checkpoints_dir / 'best.pt'}")
    print(f"  History:               {checkpoints_dir / 'history.json'}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SceneIQ coherence model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default=MODE_VIT, choices=list(MODEL_MODES))
    parser.add_argument("--splits-dir", type=str, default=str(PROCESSED_DIR / "splits"))
    parser.add_argument("--checkpoints-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--scene-graphs-dir", type=str, default=str(SCENE_GRAPH_DIR))
    parser.add_argument("--model-name", type=str, default=VIT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bce-weight", type=float, default=BCE_LOSS_WEIGHT)
    parser.add_argument("--loc-weight", type=float, default=LOC_LOSS_WEIGHT)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=f"train_{args.mode}.log")
    main(
        mode=args.mode,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        scene_graphs_dir=args.scene_graphs_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        bce_weight=args.bce_weight,
        loc_weight=args.loc_weight,
        use_wandb=args.wandb,
        wandb_run_name=args.wandb_run_name,
    )
