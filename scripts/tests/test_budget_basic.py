from __future__ import annotations

from scripts.budget import ResourceBudget


def main() -> int:
    b = ResourceBudget(max_frames=2, max_embeddings=2, max_compute_time_sec=1.0)
    assert b.can_add_frames(1)
    b.add_frames(1)
    b.add_embeddings(1)
    info = b.check()
    assert info["frames_used"] == 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
