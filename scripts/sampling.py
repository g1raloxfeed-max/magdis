from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class FrameInfo:
    index: int
    time_sec: float
    frame_bgr: np.ndarray


@dataclass(frozen=True)
class SampledFrame:
    index: int
    time_sec: float
    frame_bgr: np.ndarray
    score: float | None = None


class SamplingStrategy:
    name: str = "base"

    def reset(self) -> None:
        pass

    def sample(self, frames: Iterable[FrameInfo]) -> Iterator[SampledFrame]:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class UniformSampling(SamplingStrategy):
    name = "uniform"

    def __init__(self, fps: float = 1.0) -> None:
        self.fps = fps
        self._next_time = 0.0

    def reset(self) -> None:
        self._next_time = 0.0

    def params(self) -> dict:
        return {"fps": self.fps}

    def sample(self, frames: Iterable[FrameInfo]) -> Iterator[SampledFrame]:
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        self.reset()
        step = 1.0 / self.fps
        for f in frames:
            if f.time_sec + 1e-9 < self._next_time:
                continue
            yield SampledFrame(f.index, f.time_sec, f.frame_bgr)
            self._next_time += step


class MotionBasedSampling(SamplingStrategy):
    name = "motion"

    def __init__(self, motion_threshold: float, min_fps: float, max_fps: float) -> None:
        self.motion_threshold = motion_threshold
        self.min_fps = min_fps
        self.max_fps = max_fps
        self._prev_gray: np.ndarray | None = None
        self._next_time = 0.0

    def reset(self) -> None:
        self._prev_gray = None
        self._next_time = 0.0

    def params(self) -> dict:
        return {
            "motion_threshold": self.motion_threshold,
            "min_fps": self.min_fps,
            "max_fps": self.max_fps,
        }

    def _motion_score(self, gray: np.ndarray) -> float:
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        diff = np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)))
        self._prev_gray = gray
        return float(diff / 255.0)

    def sample(self, frames: Iterable[FrameInfo]) -> Iterator[SampledFrame]:
        if self.min_fps <= 0 or self.max_fps <= 0:
            raise ValueError("min_fps and max_fps must be > 0")
        if self.min_fps > self.max_fps:
            raise ValueError("min_fps must be <= max_fps")
        self.reset()
        for f in frames:
            gray = cvt_gray(f.frame_bgr)
            score = self._motion_score(gray)
            target_fps = self.max_fps if score >= self.motion_threshold else self.min_fps
            step = 1.0 / target_fps
            if f.time_sec + 1e-9 < self._next_time:
                continue
            yield SampledFrame(f.index, f.time_sec, f.frame_bgr, score=score)
            self._next_time += step


class EmbeddingDeltaSampling(SamplingStrategy):
    name = "embedding_delta"

    def __init__(
        self,
        delta_threshold: float,
        window_size: int,
        min_fps: float,
        max_fps: float,
        embedder,
    ) -> None:
        self.delta_threshold = delta_threshold
        self.window_size = max(1, window_size)
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.embedder = embedder
        self._prev_emb: np.ndarray | None = None
        self._deltas: list[float] = []
        self._next_time = 0.0

    def reset(self) -> None:
        self._prev_emb = None
        self._deltas = []
        self._next_time = 0.0

    def params(self) -> dict:
        return {
            "delta_threshold": self.delta_threshold,
            "window_size": self.window_size,
            "min_fps": self.min_fps,
            "max_fps": self.max_fps,
        }

    def _delta(self, emb: np.ndarray) -> float:
        if self._prev_emb is None:
            self._prev_emb = emb
            return 0.0
        sim = float(np.dot(self._prev_emb, emb))
        delta = 1.0 - sim
        self._prev_emb = emb
        return float(delta)

    def _smooth(self, value: float) -> float:
        self._deltas.append(value)
        if len(self._deltas) > self.window_size:
            self._deltas.pop(0)
        return float(np.mean(self._deltas))

    def sample(self, frames: Iterable[FrameInfo]) -> Iterator[SampledFrame]:
        if self.min_fps <= 0 or self.max_fps <= 0:
            raise ValueError("min_fps and max_fps must be > 0")
        if self.min_fps > self.max_fps:
            raise ValueError("min_fps must be <= max_fps")
        self.reset()
        for f in frames:
            emb = self.embedder(f.frame_bgr)
            delta = self._delta(emb)
            delta_sm = self._smooth(delta)
            target_fps = self.max_fps if delta_sm >= self.delta_threshold else self.min_fps
            step = 1.0 / target_fps
            if f.time_sec + 1e-9 < self._next_time:
                continue
            yield SampledFrame(f.index, f.time_sec, f.frame_bgr, score=delta_sm)
            self._next_time += step


class HybridSampling(SamplingStrategy):
    name = "hybrid"

    def __init__(
        self,
        motion_threshold: float,
        delta_threshold: float,
        window_size: int,
        min_fps: float,
        max_fps: float,
        motion_weight: float,
        delta_weight: float,
        embedder,
    ) -> None:
        self.motion = MotionBasedSampling(motion_threshold, min_fps, max_fps)
        self.delta = EmbeddingDeltaSampling(
            delta_threshold, window_size, min_fps, max_fps, embedder
        )
        self.motion_weight = motion_weight
        self.delta_weight = delta_weight
        self._next_time = 0.0

    def reset(self) -> None:
        self.motion.reset()
        self.delta.reset()
        self._next_time = 0.0

    def params(self) -> dict:
        p = {}
        p.update(self.motion.params())
        p.update(self.delta.params())
        p["motion_weight"] = self.motion_weight
        p["delta_weight"] = self.delta_weight
        return p

    def sample(self, frames: Iterable[FrameInfo]) -> Iterator[SampledFrame]:
        self.reset()
        for f in frames:
            gray = cvt_gray(f.frame_bgr)
            motion_score = self.motion._motion_score(gray)
            emb = self.delta.embedder(f.frame_bgr)
            delta = self.delta._delta(emb)
            delta_sm = self.delta._smooth(delta)
            combined = (self.motion_weight * motion_score) + (
                self.delta_weight * delta_sm
            )
            target_fps = self.motion.max_fps if combined >= self.delta.delta_threshold else self.motion.min_fps
            step = 1.0 / target_fps
            if f.time_sec + 1e-9 < self._next_time:
                continue
            yield SampledFrame(f.index, f.time_sec, f.frame_bgr, score=combined)
            self._next_time += step


def cvt_gray(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
