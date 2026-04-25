"""
SceneIQ — Per-image scene-graph preprocessing.

Reads Visual Genome's ``scene_graphs.json`` (the ground-truth scene graphs) and
emits a compact, ID-encoded representation that the GAT branch consumes at
training time. We use the VG-provided scene graphs directly rather than running
RelTR/EGTR inference — this is intentional:

  * Faster and bit-for-bit reproducible.
  * All coherent test images have ground-truth graphs already.
  * Synthetic incoherent images reuse the *scene* image's graph (the alien
    object isn't reflected in the graph — which is exactly the signal the
    fusion model should learn to exploit).

Outputs:
    data/processed/scene_graphs/
      ├── object_vocab.json     — object_label -> object_id (0 is <PAD>, 1 is <UNK>)
      ├── predicate_vocab.json  — predicate_label -> predicate_id
      └── graphs.json           — { image_id: {"objects": [o_id, ...],
                                               "edges": [[src, dst, p_id], ...]} }

"objects" is capped at SG_MAX_OBJECTS (drops the smallest-bbox objects first
so crowded VG graphs still fit). Edges whose endpoints are dropped are
discarded.

Missing scene graphs (an image_id with no record) are handled gracefully in
train.py by falling back to a single <UNK> node with no edges.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RANDOM_SEED,
    SCENE_GRAPH_DIR,
    SG_MAX_OBJECTS,
    SG_OBJECT_VOCAB_SIZE,
    SG_PREDICATE_VOCAB_SIZE,
    VISUAL_GENOME_DIR,
)
from utils import (
    extract_object_name,
    load_json,
    normalise_name,
    save_json,
    set_seed,
    setup_logging,
    timer,
)

logger = logging.getLogger("sceneiq")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ── Vocab construction ───────────────────────────────────────────────────────

def build_vocab(counts: Counter, max_size: int) -> dict:
    """Return a ``{label: id}`` vocab of size ``max_size`` with PAD=0, UNK=1.

    Frequencies below the cutoff map to the UNK id.
    """
    most_common = counts.most_common(max_size - 2)  # reserve 2 ids for PAD/UNK
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for label, _ in most_common:
        vocab[label] = len(vocab)
    return vocab


def summarise_vocab(name: str, vocab: dict, total: int) -> None:
    """Log a short report on vocab coverage."""
    logger.info("%s vocab: %d entries (cap=%d, unique=%d)",
                name, len(vocab), len(vocab), total)


# ── Scene-graph extraction ───────────────────────────────────────────────────

def _object_size(obj: dict) -> int:
    """Return the bbox area of a VG object — used to rank by size."""
    return max(0, int(obj.get("w", 0))) * max(0, int(obj.get("h", 0)))


def extract_graph(entry: dict, obj_vocab: dict, pred_vocab: dict) -> dict:
    """Convert one VG scene_graphs entry to the compact training format.

    Keeps the ``SG_MAX_OBJECTS`` largest objects (by bbox area), maps labels to
    ids using ``obj_vocab`` / ``pred_vocab``, and rewrites edges onto the new
    (0..n-1) local indexing.

    Args:
        entry: A dict with keys ``objects`` and ``relationships`` as in VG's
            ``scene_graphs.json`` layout.
        obj_vocab: Object label → id.
        pred_vocab: Predicate label → id.

    Returns:
        ``{"objects": [ids], "edges": [[src, dst, pred_id], ...]}``.
    """
    objs = entry.get("objects", []) or []
    # Rank by area, keep top-K
    ranked = sorted(
        [(i, o) for i, o in enumerate(objs)],
        key=lambda io: _object_size(io[1]),
        reverse=True,
    )[:SG_MAX_OBJECTS]

    # Map VG object_id → local index (0..n-1)
    vg_id_to_local: dict = {}
    local_obj_ids: list = []
    for local_idx, (_, obj) in enumerate(ranked):
        vg_id_to_local[obj.get("object_id", id(obj))] = local_idx
        name = extract_object_name(obj)
        local_obj_ids.append(obj_vocab.get(name, obj_vocab[UNK_TOKEN]))

    edges: list = []
    for rel in entry.get("relationships", []) or []:
        s_id = rel.get("subject_id") or (rel.get("subject") or {}).get("object_id")
        o_id = rel.get("object_id") or (rel.get("object") or {}).get("object_id")
        if s_id is None or o_id is None:
            continue
        if s_id not in vg_id_to_local or o_id not in vg_id_to_local:
            continue
        pred_name = normalise_name(rel.get("predicate", ""))
        pred_id = pred_vocab.get(pred_name, pred_vocab[UNK_TOKEN])
        edges.append([vg_id_to_local[s_id], vg_id_to_local[o_id], pred_id])

    return {"objects": local_obj_ids, "edges": edges}


# ── Main ─────────────────────────────────────────────────────────────────────

@timer
def main(vg_dir: Path, output_dir: Path, seed: int) -> None:
    """Build object/predicate vocabs and emit per-image scene graphs."""
    vg_dir = Path(vg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    sg_path = vg_dir / "scene_graphs.json"
    if not sg_path.exists():
        raise FileNotFoundError(
            f"{sg_path} not found — run scripts/download_vg.py first."
        )

    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading scene_graphs.json")
    logger.info("=" * 60)
    data = load_json(sg_path)
    logger.info("Entries: %d", len(data))

    # First pass — build vocabs
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Building vocabularies")
    logger.info("=" * 60)
    obj_counter: Counter = Counter()
    pred_counter: Counter = Counter()
    for entry in tqdm(data, desc="vocab scan", unit="img"):
        for obj in entry.get("objects", []) or []:
            name = extract_object_name(obj)
            if name:
                obj_counter[name] += 1
        for rel in entry.get("relationships", []) or []:
            pred = normalise_name(rel.get("predicate", ""))
            if pred:
                pred_counter[pred] += 1

    object_vocab = build_vocab(obj_counter, SG_OBJECT_VOCAB_SIZE)
    predicate_vocab = build_vocab(pred_counter, SG_PREDICATE_VOCAB_SIZE)
    summarise_vocab("Object", object_vocab, len(obj_counter))
    summarise_vocab("Predicate", predicate_vocab, len(pred_counter))

    save_json(object_vocab, output_dir / "object_vocab.json")
    save_json(predicate_vocab, output_dir / "predicate_vocab.json")

    # Second pass — emit per-image graphs
    logger.info("=" * 60)
    logger.info("STEP 3/3 — Encoding per-image graphs")
    logger.info("=" * 60)
    graphs: dict = {}
    n_with_edges = 0
    for entry in tqdm(data, desc="encode", unit="img"):
        iid = entry.get("image_id")
        if iid is None:
            continue
        g = extract_graph(entry, object_vocab, predicate_vocab)
        graphs[str(iid)] = g   # JSON keys must be strings
        if g["edges"]:
            n_with_edges += 1

    save_json(graphs, output_dir / "graphs.json")

    print("\n" + "=" * 60)
    print("  SceneIQ — Scene-Graph Preprocessing Summary")
    print("=" * 60)
    print(f"  Images processed:        {len(graphs)}")
    print(f"  Images with ≥1 edge:     {n_with_edges}")
    print(f"  Object vocab size:       {len(object_vocab)}")
    print(f"  Predicate vocab size:    {len(predicate_vocab)}")
    print(f"  Output directory:        {output_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Visual Genome scene graphs for the GAT branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vg-dir", type=str, default=str(VISUAL_GENOME_DIR))
    parser.add_argument("--output-dir", type=str, default=str(SCENE_GRAPH_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="build_scene_graphs.log")
    main(vg_dir=args.vg_dir, output_dir=args.output_dir, seed=args.seed)
