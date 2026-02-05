from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluation import run_tests
from scripts.experiment import run_experiment
from scripts.es_index import get_es_client


def _make_synthetic_video(path: Path, seconds: int = 2, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(seconds * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 2, 20), (110 + i * 2, 120), (0, 0, 255), -1)
        cv2.putText(frame, f"f{i}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _es_available() -> bool:
    try:
        es = get_es_client()
        return bool(es.ping())
    except Exception:
        return False


def main() -> int:
    if not _es_available():
        print("Elasticsearch not available. Skipping test.")
        return 0

    video_path = Path("data/cache/synth_test.mp4")
    _make_synthetic_video(video_path)

    config = {
        "video_path": str(video_path),
        "video_id": "synthetic_test",
        "method_name": "uniform",
        "sampling_params": {"fps": 2.0, "fps_extract": 2.0},
        "scene_params": {"abs_threshold": 0.6, "rel_drop": 0.2},
        "aggregation": "mean",
        "budget": {"max_frames": 50, "max_embeddings": 50},
        "experiment": {"allow_reuse_index": False, "run_id": None},
    }

    result = run_experiment(config)
    assert result["num_scenes"] > 0, "Expected at least one scene"

    eval_res = run_tests(
        index_names=[result["index_name"]],
        queries=["red square", "moving object"],
        top_k=3,
    )
    assert eval_res["results"], "Expected evaluation results"
    assert eval_res["results"][0]["num_scenes"] > 0, "Expected ES hits"
    print("Basic pipeline test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
