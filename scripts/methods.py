from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import time
import numpy as np
from PIL import Image

from .adaptive import adaptive_fps
from .budget import ResourceBudget
from .clip_model import encode_image
from .segment import Segment


@dataclass(frozen=True)
class FrameSample:
    index: int
    time_sec: float
    embedding: np.ndarray
    motion: float | None = None
    delta: float | None = None


class SamplingMethod(ABC):
    name: str

    def __init__(self, params: dict, budget: ResourceBudget):
        self.params = params
        self.budget = budget

    @abstractmethod
    def sample(self, video_path: str) -> List[Segment]:
        """Return list of Segments (frames + timestamps) respecting budget."""


def _frame_iter(video_path: str, fps_extract: float) -> Iterable[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 25.0
    frame_index = -1
    next_time = 0.0
    step = 1.0 / fps_extract if fps_extract > 0 else 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                ts_sec = pos_msec / 1000.0
            else:
                ts_sec = frame_index / video_fps
            if fps_extract > 0 and ts_sec + 1e-9 < next_time:
                continue
            yield frame_index, ts_sec, frame
            if fps_extract > 0:
                next_time += step
    finally:
        cap.release()


def _embed_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return encode_image(pil)


def _group_segments(samples: list[FrameSample], segment_gap_sec: float) -> List[Segment]:
    if not samples:
        return []
    segments: List[Segment] = []
    start = 0
    for i in range(1, len(samples)):
        if samples[i].time_sec - samples[i - 1].time_sec > segment_gap_sec:
            segments.append(
                Segment(
                    start_time=samples[start].time_sec,
                    end_time=samples[i - 1].time_sec,
                    frame_indices=[s.index for s in samples[start:i]],
                    embeddings=[s.embedding for s in samples[start:i]],
                )
            )
            start = i
    segments.append(
        Segment(
            start_time=samples[start].time_sec,
            end_time=samples[-1].time_sec,
            frame_indices=[s.index for s in samples[start:]],
            embeddings=[s.embedding for s in samples[start:]],
        )
    )
    return segments


def _time_exceeded(budget: ResourceBudget, start_time: float) -> bool:
    if budget.max_compute_time_sec is None:
        return False
    if (time.perf_counter() - start_time) > budget.max_compute_time_sec:
        budget.note("Max compute time exceeded; sampling truncated.")
        return True
    return False


class UniformSampler(SamplingMethod):
    name = "uniform"

    def sample(self, video_path: str) -> List[Segment]:
        fps = float(self.params.get("fps", 1.0))
        fps_extract = float(self.params.get("fps_extract", fps))
        segment_gap_sec = float(self.params.get("segment_gap_sec", 2.0))
        if fps <= 0:
            raise ValueError("fps must be > 0")
        next_time = 0.0
        step = 1.0 / fps
        samples: list[FrameSample] = []
        start_time = time.perf_counter()
        for idx, ts, frame in _frame_iter(video_path, fps_extract):
            if _time_exceeded(self.budget, start_time):
                break
            if ts + 1e-9 < next_time:
                continue
            if not self.budget.can_add_frames(1) or not self.budget.can_add_embeddings(1):
                self.budget.note("Budget reached; uniform sampling truncated.")
                break
            emb = _embed_bgr(frame)
            samples.append(FrameSample(idx, ts, emb))
            self.budget.add_frames(1)
            self.budget.add_embeddings(1)
            next_time += step
        return _group_segments(samples, segment_gap_sec)


class MotionSampler(SamplingMethod):
    name = "motion"

    def sample(self, video_path: str) -> List[Segment]:
        motion_threshold = float(self.params.get("motion_threshold", 0.15))
        min_fps = float(self.params.get("min_fps", 0.5))
        max_fps = float(self.params.get("max_fps", 2.0))
        fps_extract = float(self.params.get("fps_extract", max_fps))
        segment_gap_sec = float(self.params.get("segment_gap_sec", 2.0))
        if min_fps <= 0 or max_fps <= 0:
            raise ValueError("min_fps and max_fps must be > 0")
        prev_gray: np.ndarray | None = None
        next_time = 0.0
        samples: list[FrameSample] = []

        start_time = time.perf_counter()
        for idx, ts, frame in _frame_iter(video_path, fps_extract):
            if _time_exceeded(self.budget, start_time):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                motion = 0.0
            else:
                motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)
            prev_gray = gray

            target_fps = max_fps if motion >= motion_threshold else min_fps
            step = 1.0 / target_fps
            if ts + 1e-9 < next_time:
                continue
            if not self.budget.can_add_frames(1) or not self.budget.can_add_embeddings(1):
                self.budget.note("Budget reached; motion sampling truncated.")
                break
            emb = _embed_bgr(frame)
            samples.append(FrameSample(idx, ts, emb, motion=motion))
            self.budget.add_frames(1)
            self.budget.add_embeddings(1)
            next_time += step
        return _group_segments(samples, segment_gap_sec)


class EmbeddingDeltaSampler(SamplingMethod):
    name = "embedding_delta"

    def sample(self, video_path: str) -> List[Segment]:
        delta_threshold = float(self.params.get("delta_threshold", 0.2))
        min_fps = float(self.params.get("min_fps", 0.5))
        max_fps = float(self.params.get("max_fps", 2.0))
        fps_extract = float(self.params.get("fps_extract", max_fps))
        window_size = int(self.params.get("window_size", 3))
        segment_gap_sec = float(self.params.get("segment_gap_sec", 2.0))
        if min_fps <= 0 or max_fps <= 0:
            raise ValueError("min_fps and max_fps must be > 0")
        prev_emb: np.ndarray | None = None
        deltas: list[float] = []
        next_time = 0.0
        samples: list[FrameSample] = []

        start_time = time.perf_counter()
        for idx, ts, frame in _frame_iter(video_path, fps_extract):
            if _time_exceeded(self.budget, start_time):
                break
            emb = _embed_bgr(frame)
            if prev_emb is None:
                delta = 0.0
            else:
                delta = 1.0 - float(np.dot(prev_emb, emb))
            prev_emb = emb
            deltas.append(delta)
            if len(deltas) > max(1, window_size):
                deltas.pop(0)
            delta_sm = float(np.mean(deltas))

            target_fps = max_fps if delta_sm >= delta_threshold else min_fps
            step = 1.0 / target_fps
            if ts + 1e-9 < next_time:
                continue
            if not self.budget.can_add_frames(1) or not self.budget.can_add_embeddings(1):
                self.budget.note("Budget reached; embedding-delta sampling truncated.")
                break
            samples.append(FrameSample(idx, ts, emb, delta=delta_sm))
            self.budget.add_frames(1)
            self.budget.add_embeddings(1)
            next_time += step
        return _group_segments(samples, segment_gap_sec)


class HybridSampler(SamplingMethod):
    name = "hybrid"

    def sample(self, video_path: str) -> List[Segment]:
        motion_threshold = float(self.params.get("motion_threshold", 0.15))
        delta_threshold = float(self.params.get("delta_threshold", 0.2))
        min_fps = float(self.params.get("min_fps", 0.5))
        max_fps = float(self.params.get("max_fps", 2.0))
        fps_extract = float(self.params.get("fps_extract", max_fps))
        window_size = int(self.params.get("window_size", 3))
        segment_gap_sec = float(self.params.get("segment_gap_sec", 2.0))
        policy_params = self.params.get("policy", {"type": "linear", "min_fps": min_fps, "max_fps": max_fps})

        prev_gray: np.ndarray | None = None
        prev_emb: np.ndarray | None = None
        deltas: list[float] = []
        next_time = 0.0
        samples: list[FrameSample] = []

        start_time = time.perf_counter()
        for idx, ts, frame in _frame_iter(video_path, fps_extract):
            if _time_exceeded(self.budget, start_time):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                motion = 0.0
            else:
                motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)
            prev_gray = gray

            emb = _embed_bgr(frame)
            if prev_emb is None:
                delta = 0.0
            else:
                delta = 1.0 - float(np.dot(prev_emb, emb))
            prev_emb = emb
            deltas.append(delta)
            if len(deltas) > max(1, window_size):
                deltas.pop(0)
            delta_sm = float(np.mean(deltas))

            target_fps = adaptive_fps(
                prev_embedding=None,
                curr_embedding=emb,
                prev_motion=None,
                curr_motion=motion,
                policy_params={
                    "type": policy_params.get("type", "linear"),
                    "min_fps": min_fps,
                    "max_fps": max_fps,
                    "motion_threshold": motion_threshold,
                    "delta_threshold": delta_threshold,
                    "delta": delta_sm,
                    "motion": motion,
                },
            )

            step = 1.0 / float(target_fps)
            if ts + 1e-9 < next_time:
                continue
            if not self.budget.can_add_frames(1) or not self.budget.can_add_embeddings(1):
                self.budget.note("Budget reached; hybrid sampling truncated.")
                break
            samples.append(FrameSample(idx, ts, emb, motion=motion, delta=delta_sm))
            self.budget.add_frames(1)
            self.budget.add_embeddings(1)
            next_time += step

        return _group_segments(samples, segment_gap_sec)
