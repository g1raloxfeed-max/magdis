from __future__ import annotations

import numpy as np


def aggregate_embeddings(
    embeddings: list[np.ndarray],
    mode: str = "mean",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate embeddings.

    Modes:
    - mean
    - max
    - median
    - weighted_mean
    - medoid
    - pca_top1
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty")
    arr = np.stack(embeddings, axis=0)

    if mode == "mean":
        agg = np.mean(arr, axis=0)
    elif mode == "max":
        agg = np.max(arr, axis=0)
    elif mode == "median":
        agg = np.median(arr, axis=0)
    elif mode == "weighted_mean":
        if weights is None:
            raise ValueError("weights required for weighted_mean")
        w = np.asarray(weights, dtype=np.float32)
        w = w / (np.sum(w) + 1e-9)
        agg = np.sum(arr * w[:, None], axis=0)
    elif mode == "medoid":
        sims = arr @ arr.T
        dists = 1.0 - sims
        idx = int(np.argmin(np.sum(dists, axis=0)))
        agg = arr[idx]
    elif mode == "pca_top1":
        u, s, vt = np.linalg.svd(arr - np.mean(arr, axis=0), full_matrices=False)
        agg = vt[0]
    else:
        raise ValueError("Unknown aggregation mode")

    norm = np.linalg.norm(agg)
    if norm > 0:
        agg = agg / norm
    return agg.astype(np.float32)
