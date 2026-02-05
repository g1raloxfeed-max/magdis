from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceBudget:
    max_frames: int | None = None
    max_embeddings: int | None = None
    max_compute_time_sec: float | None = None
    max_storage_mb: float | None = None

    frames_used: int = 0
    embeddings_used: int = 0
    compute_time_sec: float = 0.0
    storage_mb: float = 0.0
    notes: list[str] = field(default_factory=list)

    def check(self) -> dict[str, Any]:
        """Return dict of current usage vs limits (for logging)."""
        return {
            "frames_used": self.frames_used,
            "max_frames": self.max_frames,
            "embeddings_used": self.embeddings_used,
            "max_embeddings": self.max_embeddings,
            "compute_time_sec": self.compute_time_sec,
            "max_compute_time_sec": self.max_compute_time_sec,
            "storage_mb": self.storage_mb,
            "max_storage_mb": self.max_storage_mb,
            "notes": list(self.notes),
        }

    def can_add_frames(self, count: int) -> bool:
        if self.max_frames is None:
            return True
        return (self.frames_used + count) <= self.max_frames

    def can_add_embeddings(self, count: int) -> bool:
        if self.max_embeddings is None:
            return True
        return (self.embeddings_used + count) <= self.max_embeddings

    def add_frames(self, count: int) -> None:
        self.frames_used += count

    def add_embeddings(self, count: int) -> None:
        self.embeddings_used += count

    def add_compute_time(self, seconds: float) -> None:
        self.compute_time_sec += seconds

    def add_storage(self, mb: float) -> None:
        self.storage_mb += mb

    def note(self, text: str) -> None:
        self.notes.append(text)
