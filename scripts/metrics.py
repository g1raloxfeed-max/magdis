from __future__ import annotations

from typing import Iterable


def recall_at_k(results: list[str], ground_truth: set[str], k: int) -> float:
    """Recall@K. Example: recall_at_k(["a","b"], {"b"}, 2) -> 1.0."""
    if not ground_truth:
        return 0.0
    top = results[:k]
    return len(set(top) & ground_truth) / float(len(ground_truth))


def mAP(all_results: list[list[str]], all_gt: list[set[str]], k: int) -> float:
    """Mean Average Precision. Expects parallel lists of results and GT."""
    if not all_results:
        return 0.0
    ap_sum = 0.0
    for results, gt in zip(all_results, all_gt):
        if not gt:
            continue
        hits = 0
        precisions = []
        for i, r in enumerate(results[:k], start=1):
            if r in gt:
                hits += 1
                precisions.append(hits / i)
        ap = sum(precisions) / max(len(gt), 1)
        ap_sum += ap
    return ap_sum / max(len(all_results), 1)


def temporal_coverage(selected_segments: list[tuple[float, float]], gt_intervals: list[tuple[float, float]]) -> float:
    """Coverage of GT intervals by selected segments."""
    if not gt_intervals:
        return 0.0
    covered = 0.0
    total = 0.0
    for g0, g1 in gt_intervals:
        total += max(0.0, g1 - g0)
        overlap = 0.0
        for s0, s1 in selected_segments:
            inter = max(0.0, min(g1, s1) - max(g0, s0))
            overlap += inter
        covered += min(overlap, g1 - g0)
    return covered / max(total, 1e-9)


def frames_reduction_ratio(original_count: int, selected_count: int) -> float:
    """1 - selected/original (higher is more reduction)."""
    if original_count <= 0:
        return 0.0
    return 1.0 - (selected_count / float(original_count))


def embeddings_per_minute(emb_count: int, video_duration_sec: float) -> float:
    """Embeddings per minute of video."""
    if video_duration_sec <= 0:
        return 0.0
    return emb_count / (video_duration_sec / 60.0)


def processing_time_metrics(start: float, end: float) -> dict:
    """Return dict with processing_time_sec."""
    return {"processing_time_sec": float(max(0.0, end - start))}
