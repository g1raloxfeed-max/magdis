from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Segment:
    start_time: float
    end_time: float
    frame_indices: List[int]
    embeddings: List[np.ndarray]  # per frame

    def duration(self) -> float:
        return float(self.end_time - self.start_time)
