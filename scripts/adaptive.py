from __future__ import annotations

import math


def adaptive_fps(prev_embedding, curr_embedding, prev_motion, curr_motion, policy_params) -> float:
    """Return new fps in [min_fps, max_fps].

    Policy:
    - linear: fps = min_fps + (max_fps-min_fps) * score
    - logistic: fps = min_fps + (max_fps-min_fps) * sigmoid(k*(score-0.5))

    score combines motion + embedding delta thresholds:
    score = 0.5 * motion_norm + 0.5 * delta_norm
    motion_norm = min(1, motion / motion_threshold)
    delta_norm = min(1, delta / delta_threshold)
    """
    p = policy_params
    min_fps = float(p.get("min_fps", 0.5))
    max_fps = float(p.get("max_fps", 2.0))
    motion = float(p.get("motion", 0.0))
    delta = float(p.get("delta", 0.0))
    motion_threshold = float(p.get("motion_threshold", 0.15))
    delta_threshold = float(p.get("delta_threshold", 0.2))

    motion_norm = min(1.0, motion / max(motion_threshold, 1e-6))
    delta_norm = min(1.0, delta / max(delta_threshold, 1e-6))
    score = 0.5 * motion_norm + 0.5 * delta_norm

    policy_type = p.get("type", "linear")
    if policy_type == "logistic":
        k = float(p.get("k", 6.0))
        score = 1.0 / (1.0 + math.exp(-k * (score - 0.5)))

    fps = min_fps + (max_fps - min_fps) * score
    return float(min(max_fps, max(min_fps, fps)))
