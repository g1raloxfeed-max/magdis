from __future__ import annotations

from scripts.adaptive import adaptive_fps


def main() -> int:
    fps = adaptive_fps(
        prev_embedding=None,
        curr_embedding=None,
        prev_motion=None,
        curr_motion=0.2,
        policy_params={"type": "linear", "min_fps": 0.5, "max_fps": 2.0, "motion": 0.2, "delta": 0.2},
    )
    assert 0.5 <= fps <= 2.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
