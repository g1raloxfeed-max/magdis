from __future__ import annotations

from scripts.metrics import frames_reduction_ratio, recall_at_k


def main() -> int:
    r = frames_reduction_ratio(100, 40)
    assert 0.5 < r < 1.0
    rec = recall_at_k(["a", "b", "c"], {"b"}, 2)
    assert rec == 1.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
