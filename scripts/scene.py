from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .segment import Segment


@dataclass
class Scene:
    scene_id: str
    start_time: float
    end_time: float
    frame_indices: list[int]
    embeddings: list[np.ndarray]


def detect_scenes(
    segments: List[Segment],
    abs_threshold: float,
    rel_drop: float,
    local_min_threshold: float,
    use_local_min: bool,
) -> List[Scene]:
    """Detect scene boundaries using similarity between adjacent segment embeddings."""
    if not segments:
        return []
    seg_embs = [np.mean(np.stack(s.embeddings, axis=0), axis=0) for s in segments]
    seg_embs = [e / (np.linalg.norm(e) + 1e-9) for e in seg_embs]

    sims = [1.0]
    for i in range(1, len(seg_embs)):
        sims.append(float(np.dot(seg_embs[i - 1], seg_embs[i])))

    boundaries = set()
    for i in range(1, len(sims)):
        if sims[i] < abs_threshold:
            boundaries.add(i)
        if sims[i - 1] > 1e-6 and sims[i] < sims[i - 1] * (1.0 - rel_drop):
            boundaries.add(i)
        if use_local_min and 1 <= i < len(sims) - 1:
            if sims[i] < sims[i - 1] and sims[i] < sims[i + 1] and sims[i] < local_min_threshold:
                boundaries.add(i)

    scenes: List[Scene] = []
    start = 0
    for i in range(1, len(segments) + 1):
        if i in boundaries or i == len(segments):
            segs = segments[start:i]
            scene_id = f"scene_{len(scenes)+1:04d}"
            scenes.append(
                Scene(
                    scene_id=scene_id,
                    start_time=segs[0].start_time,
                    end_time=segs[-1].end_time,
                    frame_indices=[idx for s in segs for idx in s.frame_indices],
                    embeddings=[e for s in segs for e in s.embeddings],
                )
            )
            start = i
    return scenes
