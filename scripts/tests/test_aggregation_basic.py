from __future__ import annotations

import numpy as np

from scripts.aggregation import aggregate_embeddings


def main() -> int:
    embs = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    out = aggregate_embeddings(embs, mode="mean")
    assert out.shape == (2,)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
