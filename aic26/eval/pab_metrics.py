"""Reusable retrieval evaluation utilities for PAB local evaluation.

Usage::

    from aic26.eval.pab_metrics import (
        compute_single_positive_metrics,
        find_positive_ranks,
        build_topk_records,
        write_json,
        write_jsonl,
        write_metrics_markdown,
    )

Only standard library + numpy are imported here — no heavy ML dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def compute_single_positive_metrics(
    positive_ranks: Sequence[Optional[int]],
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Union[int, float]]:
    """Compute retrieval metrics assuming one positive image per query.

    Args:
        positive_ranks: 1-based positive ranks, one per query. Use ``None``
            when the positive was not found in the ranking (rank is infinite).
        k_values: Recall cut-offs to compute (default: 1, 5, 10).

    Returns:
        Dictionary with keys:

        - ``queries``    — total number of queries
        - ``found_any``  — number of queries where the positive was found
        - ``mAP``        — mean average precision (AP = 1/rank per query)
        - ``R@K``        — recall at each K in *k_values*
        - ``mean_rank``  — mean positive rank over found queries
        - ``median_rank``— median positive rank over found queries
    """
    n = len(positive_ranks)
    aps: List[float] = []
    recall_at_k: Dict[int, List[int]] = {k: [] for k in k_values}
    valid_ranks: List[int] = []

    for rank in positive_ranks:
        if rank is None:
            aps.append(0.0)
            for k in k_values:
                recall_at_k[k].append(0)
        else:
            aps.append(1.0 / rank)
            valid_ranks.append(rank)
            for k in k_values:
                recall_at_k[k].append(1 if rank <= k else 0)

    result: Dict[str, Union[int, float]] = {
        "queries": n,
        "found_any": len(valid_ranks),
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "mean_rank": float(np.mean(valid_ranks)) if valid_ranks else float("inf"),
        "median_rank": float(np.median(valid_ranks)) if valid_ranks else float("inf"),
    }
    for k in k_values:
        result[f"R@{k}"] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0

    return result


def find_positive_ranks(
    gallery_ids: List[str],
    ranked_indices: np.ndarray,
    positive_images: List[str],
) -> List[Optional[int]]:
    """Find 1-based ranks of positive images in a ranked gallery.

    Args:
        gallery_ids: Gallery image IDs / relative paths, length G.
        ranked_indices: Integer array of shape ``(Q, G)`` or ``(Q, K)``
            containing gallery indices in descending-similarity order.
            If only top-K indices are provided and the positive is not in
            the top-K, ``None`` is returned for that query.
        positive_images: Positive image ID for each query, length Q.
            Must use the same ID format as *gallery_ids*.

    Returns:
        List of 1-based ranks, one per query. ``None`` means the positive
        was not found in *ranked_indices*.
    """
    gid_to_idx: Dict[str, int] = {gid: i for i, gid in enumerate(gallery_ids)}
    ranks: List[Optional[int]] = []

    for q_idx, pos_img in enumerate(positive_images):
        pos_gallery_idx = gid_to_idx.get(pos_img)
        if pos_gallery_idx is None:
            ranks.append(None)
            continue
        row = ranked_indices[q_idx]
        match = np.where(row == pos_gallery_idx)[0]
        if len(match) == 0:
            ranks.append(None)
        else:
            ranks.append(int(match[0]) + 1)

    return ranks


def build_topk_records(
    query_ids: List[str],
    query_captions: List[str],
    positive_images: List[str],
    gallery_ids: List[str],
    ranked_indices: np.ndarray,
    positive_ranks: List[Optional[int]],
    topk: int = 10,
) -> List[Dict]:
    """Build per-query top-k ranking records for inspection.

    Args:
        query_ids: Query identifiers.
        query_captions: Query text captions.
        positive_images: Positive image ID per query.
        gallery_ids: Gallery image IDs / relative paths.
        ranked_indices: Integer array ``(Q, G)`` or ``(Q, K)`` of ranked indices.
        positive_ranks: 1-based positive ranks from :func:`find_positive_ranks`.
        topk: Number of top results to include per record (default 10).

    Returns:
        List of dicts, each containing:

        - ``query_id``
        - ``caption``
        - ``positive_image``
        - ``positive_rank``
        - ``top{topk}`` (e.g. ``top10``)
    """
    records = []
    for i, (qid, cap, pos_img, rank) in enumerate(
        zip(query_ids, query_captions, positive_images, positive_ranks)
    ):
        top_k_gallery_indices = ranked_indices[i, :topk]
        top_k_ids = [gallery_ids[int(j)] for j in top_k_gallery_indices]
        records.append(
            {
                "query_id": qid,
                "caption": cap,
                "positive_image": pos_img,
                "positive_rank": rank,
                f"top{topk}": top_k_ids,
            }
        )
    return records


def write_json(path: Union[str, Path], obj: object) -> None:
    """Write *obj* as an indented JSON file.

    Args:
        path: Destination file path.
        obj: JSON-serialisable object.
    """
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Union[str, Path], rows: List[Dict]) -> None:
    """Write a list of dicts as a JSONL file (one JSON object per line).

    Args:
        path: Destination file path.
        rows: Records to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_markdown(
    path: Union[str, Path],
    run_id: str,
    metrics: Dict[str, Union[int, float]],
    extra: Optional[Dict[str, str]] = None,
) -> None:
    """Write a human-readable markdown metrics table.

    Args:
        path: Destination file path.
        run_id: Run identifier shown in the ``# Local Eval:`` header.
        metrics: Dict from :func:`compute_single_positive_metrics` (may include
            additional keys such as ``run_id``, ``model``, ``dataset``).
        extra: Optional mapping of extra label→value rows inserted before
            the metric rows (e.g. ``{"Model": "PE-Core-G14-448"}``).
    """
    lines = [
        f"# Local Eval: {run_id}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"| {k} | {v} |")

    metric_keys = ("queries", "found_any", "mAP", "R@1", "R@5", "R@10", "median_rank", "mean_rank")
    for key in metric_keys:
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            if key in ("mAP", "R@1", "R@5", "R@10"):
                formatted = f"{val:.4f}"
            else:
                formatted = f"{val:.1f}"
        else:
            formatted = str(val)
        lines.append(f"| {key} | {formatted} |")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
