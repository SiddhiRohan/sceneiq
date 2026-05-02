"""
SceneIQ — Extract compact scene graphs from Visual Genome.

Reads the full VG ``scene_graphs.json`` (108K entries) and produces:

* ``data/processed/scene_graphs/vocab.json``
    — ``name_to_idx`` and ``predicate_to_idx`` dictionaries.
* ``data/processed/scene_graphs/graph_index.json``
    — Per-image compact graph (node labels, edge index, edge labels)
      keyed by string image_id.

These are consumed by ``train_fusion.py`` to build graph inputs for the
GAT branch of the fusion model.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SCENE_GRAPHS_DIR, VISUAL_GENOME_DIR
from utils import extract_object_name, normalise_name, save_json, setup_logging, timer

logger = logging.getLogger("sceneiq")


def build_vocab(
    scene_graphs: list,
    min_object_count: int = 5,
    min_predicate_count: int = 5,
) -> tuple[dict, dict]:
    """Build vocabularies for object names and relationship predicates.

    Args:
        scene_graphs: Raw VG scene graph entries.
        min_object_count: Minimum frequency to include an object name.
        min_predicate_count: Minimum frequency to include a predicate.

    Returns:
        Tuple of (name_to_idx, predicate_to_idx) dictionaries.
    """
    obj_counter: Counter = Counter()
    pred_counter: Counter = Counter()

    for sg in scene_graphs:
        for obj in sg.get("objects", []):
            name = extract_object_name(obj)
            if name:
                obj_counter[name] += 1
        for rel in sg.get("relationships", []):
            pred = normalise_name(rel.get("predicate", ""))
            if pred:
                pred_counter[pred] += 1

    name_to_idx = {"<UNK>": 0}
    idx = 1
    for name, count in obj_counter.most_common():
        if count >= min_object_count:
            name_to_idx[name] = idx
            idx += 1

    predicate_to_idx = {"<UNK>": 0, "alien_in_scene": 1}
    idx = 2
    for pred, count in pred_counter.most_common():
        if count >= min_predicate_count:
            predicate_to_idx[pred] = idx
            idx += 1

    logger.info(
        "Vocabulary: %d object names, %d predicates (after min-count filter)",
        len(name_to_idx), len(predicate_to_idx),
    )
    return name_to_idx, predicate_to_idx


def extract_graph(
    sg_entry: dict,
    name_to_idx: dict,
    predicate_to_idx: dict,
) -> dict:
    """Convert one VG scene graph to a compact integer representation.

    Args:
        sg_entry: A single entry from ``scene_graphs.json``.
        name_to_idx: Object name vocabulary.
        predicate_to_idx: Predicate vocabulary.

    Returns:
        Dict with ``node_labels``, ``edge_index``, ``edge_labels``, ``num_nodes``.
    """
    objects = sg_entry.get("objects", [])
    relationships = sg_entry.get("relationships", [])

    if not objects:
        return {
            "node_labels": [0],
            "edge_index": [[], []],
            "edge_labels": [],
            "num_nodes": 1,
        }

    obj_id_to_node = {}
    node_labels = []
    for i, obj in enumerate(objects):
        obj_id_to_node[obj["object_id"]] = i
        name = extract_object_name(obj)
        node_labels.append(name_to_idx.get(name, 0))

    src_list = []
    dst_list = []
    edge_labels = []
    for rel in relationships:
        s = obj_id_to_node.get(rel["subject_id"])
        d = obj_id_to_node.get(rel["object_id"])
        if s is None or d is None:
            continue
        pred = normalise_name(rel.get("predicate", ""))
        pred_idx = predicate_to_idx.get(pred, 0)
        # Forward edge
        src_list.append(s)
        dst_list.append(d)
        edge_labels.append(pred_idx)
        # Reverse edge (make graph undirected)
        src_list.append(d)
        dst_list.append(s)
        edge_labels.append(pred_idx)

    return {
        "node_labels": node_labels,
        "edge_index": [src_list, dst_list],
        "edge_labels": edge_labels,
        "num_nodes": len(node_labels),
    }


@timer
def main(sg_path: Path, out_dir: Path, min_obj: int, min_pred: int) -> None:
    """Extract all scene graphs and build vocabulary.

    Args:
        sg_path: Path to VG ``scene_graphs.json``.
        out_dir: Output directory for vocab and graph index.
        min_obj: Minimum count to include an object name.
        min_pred: Minimum count to include a predicate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1/3 — Loading scene_graphs.json")
    logger.info("=" * 60)
    with open(sg_path, "r", encoding="utf-8") as f:
        scene_graphs = json.load(f)
    logger.info("Loaded %d scene graph entries", len(scene_graphs))

    logger.info("=" * 60)
    logger.info("STEP 2/3 — Building vocabulary")
    logger.info("=" * 60)
    name_to_idx, predicate_to_idx = build_vocab(
        scene_graphs, min_object_count=min_obj, min_predicate_count=min_pred,
    )
    save_json(
        {"name_to_idx": name_to_idx, "predicate_to_idx": predicate_to_idx},
        out_dir / "vocab.json",
    )

    logger.info("=" * 60)
    logger.info("STEP 3/3 — Extracting compact graphs")
    logger.info("=" * 60)
    graph_index = {}
    for sg in tqdm(scene_graphs, desc="extracting graphs", unit="img"):
        image_id = str(sg["image_id"])
        graph_index[image_id] = extract_graph(sg, name_to_idx, predicate_to_idx)

    save_json(graph_index, out_dir / "graph_index.json")

    print("\n" + "=" * 60)
    print("  SceneIQ — Scene Graph Extraction Summary")
    print("=" * 60)
    print(f"  Source:            {sg_path}")
    print(f"  Graphs extracted:  {len(graph_index)}")
    print(f"  Object vocab:      {len(name_to_idx)} names")
    print(f"  Predicate vocab:   {len(predicate_to_idx)} predicates")
    print(f"  Output dir:        {out_dir}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract compact scene graphs from Visual Genome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sg-path", type=str,
        default=str(VISUAL_GENOME_DIR / "scene_graphs.json"),
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(SCENE_GRAPHS_DIR),
    )
    parser.add_argument("--min-obj-count", type=int, default=5)
    parser.add_argument("--min-pred-count", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file="extract_scene_graphs.log")
    main(
        sg_path=args.sg_path,
        out_dir=args.out_dir,
        min_obj=args.min_obj_count,
        min_pred=args.min_pred_count,
    )
