from __future__ import annotations

import numpy as np

from scripts.scene import detect_scenes
from scripts.segment import Segment


def main() -> int:
    segs = [
        Segment(0.0, 1.0, [1], [np.array([1.0, 0.0], dtype=np.float32)]),
        Segment(1.0, 2.0, [2], [np.array([0.0, 1.0], dtype=np.float32)]),
    ]
    scenes = detect_scenes(segs, abs_threshold=0.1, rel_drop=0.1, local_min_threshold=0.1, use_local_min=False)
    assert len(scenes) >= 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
