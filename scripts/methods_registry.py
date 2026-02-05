from __future__ import annotations

from typing import Dict, Type

from .budget import ResourceBudget
from .methods import EmbeddingDeltaSampler, HybridSampler, MotionSampler, SamplingMethod, UniformSampler

METHODS: Dict[str, Type[SamplingMethod]] = {
    "uniform": UniformSampler,
    "motion": MotionSampler,
    "embedding_delta": EmbeddingDeltaSampler,
    "hybrid": HybridSampler,
}


def get_method(name: str, params: dict, budget: ResourceBudget) -> SamplingMethod:
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    return METHODS[name](params, budget)
