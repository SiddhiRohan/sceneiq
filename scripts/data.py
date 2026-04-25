"""
SceneIQ — Dataset + collation shared across train.py / evaluate.py / demo.py.

Produces the three tensors every mode needs:
  * ``pixel_values``: (3, 224, 224) — ViT-ready
  * ``label``: () long — 0 / 1
  * ``target_mask``: (H, W) float in [0, 1] — fraction of each patch covered
    by the ground-truth paste bbox. All zeros for coherent samples.

Also carries the raw ``scene_image_id`` so the GAT branch can look up the
scene graph, and the record's original ``image_path`` for reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("sceneiq")


# ── Bbox → patch mask ────────────────────────────────────────────────────────

def bbox_to_patch_mask(
    bbox: list | tuple,
    image_width: int,
    image_height: int,
    grid: int,
) -> torch.Tensor:
    """Return a (grid, grid) tensor with per-patch coverage of ``bbox``.

    Args:
        bbox: ``(x, y, w, h)`` in pixel coordinates of the original image.
        image_width: Original image width (before the ViT processor resize).
        image_height: Original image height.
        grid: Target grid size (14 for ViT-B/16 @ 224).

    Returns:
        Float tensor in [0, 1] whose sum is the bbox's fractional area.
    """
    mask = torch.zeros(grid, grid, dtype=torch.float32)
    if bbox is None or image_width <= 0 or image_height <= 0:
        return mask
    x, y, w, h = bbox
    x0 = max(0.0, x) / image_width
    y0 = max(0.0, y) / image_height
    x1 = min(image_width, x + w) / image_width
    y1 = min(image_height, y + h) / image_height
    if x1 <= x0 or y1 <= y0:
        return mask

    # Patch index ranges (fractional)
    gx0 = x0 * grid
    gx1 = x1 * grid
    gy0 = y0 * grid
    gy1 = y1 * grid

    ix0 = int(gx0)
    ix1 = min(grid, int(gx1) + 1)
    iy0 = int(gy0)
    iy1 = min(grid, int(gy1) + 1)

    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            cover_x = max(0.0, min(ix + 1.0, gx1) - max(float(ix), gx0))
            cover_y = max(0.0, min(iy + 1.0, gy1) - max(float(iy), gy0))
            area = cover_x * cover_y   # in patch-side units
            if area > 0:
                mask[iy, ix] = area
    return mask


# ── Dataset ──────────────────────────────────────────────────────────────────

class SceneIQDataset(Dataset):
    """Loads (image, label, localization mask, scene-graph key) from a split manifest.

    Args:
        records: Split records loaded from ``{train,val,test}.json``.
        processor: HF ``ViTImageProcessor`` for resize + normalize.
        patch_grid: Patch grid side length (14).
    """

    def __init__(self, records: list, processor, patch_grid: int = 14):
        self.records = records
        self.processor = processor
        self.patch_grid = patch_grid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        W, H = img.size
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]

        mask = bbox_to_patch_mask(
            rec.get("paste_bbox"), W, H, self.patch_grid,
        )

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "target_mask": mask,                                # (grid, grid)
            "scene_image_id": int(rec.get("scene_image_id", rec.get("image_id", -1))),
            "image_path": rec["image_path"],
            "alien_object": rec.get("alien_object"),
        }


def make_collate(scene_graphs: dict, gat_module, device: torch.device | None, sg_max_objects: int):
    """Return a collate_fn that also packs the batch's scene graphs.

    Args:
        scene_graphs: ``{str(image_id): {"objects":..., "edges":...}}`` loaded
            from ``scene_graphs/graphs.json``. May be empty (``{}``) when the
            GAT branch isn't used — in that case graph_batch will still be
            produced but stays on PAD nodes only.
        gat_module: The :class:`SceneIQGAT` instance; needed for its
            ``collate_graphs`` static method. Pass ``None`` to skip graph
            collation entirely (ViT-only training).
        device: Optional device for the graph tensors.
        sg_max_objects: Cap on nodes per graph.
    """
    def _collate(batch: list) -> dict:
        out = {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
            "target_mask": torch.stack([b["target_mask"] for b in batch]),
            "image_paths": [b["image_path"] for b in batch],
            "alien_objects": [b["alien_object"] for b in batch],
            "scene_image_ids": [b["scene_image_id"] for b in batch],
        }
        if gat_module is None:
            out["graph_batch"] = None
            return out

        graphs = [scene_graphs.get(str(b["scene_image_id"]), None) for b in batch]
        n_max = min(
            sg_max_objects,
            max(1, *(len((g or {}).get("objects", []) or [1]) for g in graphs)),
        )
        out["graph_batch"] = gat_module.collate_graphs(graphs, n_max=n_max, device=device)
        return out

    return _collate
