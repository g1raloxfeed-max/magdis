from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Any

from .aggregation import aggregate_embeddings
from .budget import ResourceBudget
from .clip_model import embedding_dim
from .es_index import get_es_client, index_scene, ensure_scene_index
from .methods_registry import get_method
from .metrics import embeddings_per_minute, frames_reduction_ratio, processing_time_metrics
from .scene import detect_scenes
from .segment import Segment
from .utils import (
    ensure_dir,
    generate_run_id,
    maybe_load_config,
    params_hash,
    safe_text,
    to_iso_z,
    write_json,
    log_line,
)

DATA_DIR = Path("data")
EXPERIMENTS_DIR = DATA_DIR / "experiments"


def _video_info(video_path: str) -> tuple[int, float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return frame_count, duration


def _index_name(video_id: str, method_name: str, params_hash_: str, run_id: str) -> str:
    return f"video_{safe_text(video_id)}__method_{safe_text(method_name)}__params_{params_hash_}__run_{run_id}"


def _flatten_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def _save_artifacts(
    run_dir: Path,
    config: dict,
    segments: list[Segment],
    scenes: list[dict],
    metrics: dict,
    index_name: str,
    selected_frames: list[dict],
    log_path: Path,
) -> None:
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "selected_frames.json", selected_frames)
    write_json(
        run_dir / "segments.json",
        [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "frame_indices": s.frame_indices,
                "embeddings_count": len(s.embeddings),
            }
            for s in segments
        ],
    )
    write_json(run_dir / "scenes.json", scenes)
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "es_index_name.txt").write_text(index_name, encoding="utf-8")
    log_line(log_path, f"Saved artifacts to {run_dir}")


def run_experiment(config: dict) -> dict:
    """Executes: sampling -> segments -> scenes -> aggregation -> indexing + artifacts.

    Example:
    >>> run_experiment({"video_path":"data/videos/sample.mp4","video_id":"cam01","method_name":"uniform"})
    """
    video_path = config["video_path"]
    video_id = config["video_id"]
    method_name = config["method_name"]
    sampling_params = config.get("sampling_params", {})
    scene_params = config.get("scene_params", {})
    aggregation = config.get("aggregation", "mean")
    budget_cfg = config.get("budget", {})
    exp_cfg = config.get("experiment", {})

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    budget = ResourceBudget(**budget_cfg)
    method = get_method(method_name, sampling_params, budget)

    run_id = exp_cfg.get("run_id") or generate_run_id()
    run_dir = ensure_dir(EXPERIMENTS_DIR / run_id)
    log_path = run_dir / "logs.txt"

    p_hash = params_hash(sampling_params)
    index_name = _index_name(video_id, method_name, p_hash, run_id)
    allow_reuse = bool(exp_cfg.get("allow_reuse_index", False))

    es = get_es_client()
    if es.indices.exists(index=index_name) and not allow_reuse:
        run_id = generate_run_id()
        run_dir = ensure_dir(EXPERIMENTS_DIR / run_id)
        index_name = _index_name(video_id, method_name, p_hash, run_id)

    start = time.perf_counter()
    segments = method.sample(video_path)
    budget.add_compute_time(time.perf_counter() - start)
    scene_list = detect_scenes(
        segments=segments,
        abs_threshold=float(scene_params.get("abs_threshold", 0.6)),
        rel_drop=float(scene_params.get("rel_drop", 0.2)),
        local_min_threshold=float(scene_params.get("local_min_threshold", 0.55)),
        use_local_min=bool(scene_params.get("use_local_min", True)),
    )

    dims = embedding_dim()
    ensure_scene_index(es, dims, index_name=index_name)

    scenes_payload = []
    for scene in scene_list:
        scene_emb = aggregate_embeddings(scene.embeddings, aggregation)
        payload = {
            "scene_id": scene.scene_id,
            "video_id": video_id,
            "method_name": method_name,
            "params_hash": p_hash,
            "start_time": to_iso_z(scene.start_time),
            "end_time": to_iso_z(scene.end_time),
            "keyframe_path": None,
            "frame_count": len(scene.frame_indices),
            "scene_embedding": scene_emb.tolist(),
        }
        index_scene(payload, es=es, index_name=index_name)
        scenes_payload.append(payload)

    es.indices.refresh(index=index_name)
    end = time.perf_counter()
    budget.add_compute_time(end - start)

    selected_frames = [
        {"segment_id": i, "frame_indices": s.frame_indices} for i, s in enumerate(segments)
    ]
    total_embeddings = sum(len(s.embeddings) for s in segments)
    original_count, duration_sec = _video_info(video_path)
    metrics = {
        **processing_time_metrics(start, end),
        "num_segments": len(segments),
        "num_scenes": len(scene_list),
        "num_embeddings": total_embeddings,
        "frames_reduction": frames_reduction_ratio(original_count, total_embeddings),
        "embeddings_per_minute": embeddings_per_minute(total_embeddings, duration_sec),
        "method_name": method_name,
        "budget": budget.check(),
        "params_hash": p_hash,
    }

    _save_artifacts(
        run_dir=run_dir,
        config=config,
        segments=segments,
        scenes=scenes_payload,
        metrics=metrics,
        index_name=index_name,
        selected_frames=selected_frames,
        log_path=log_path,
    )

    return {
        "run_id": run_id,
        "index_name": index_name,
        "num_segments": len(segments),
        "num_scenes": len(scene_list),
        "artifacts_dir": str(run_dir),
        "metrics": metrics,
    }


def run_sweep(config: dict) -> list[dict]:
    sweep = config.get("sweep", {})
    grid = _flatten_grid(sweep.get("sampling_params", {}))
    results = []
    for params in grid:
        cfg = dict(config)
        cfg["sampling_params"] = {**config.get("sampling_params", {}), **params}
        results.append(run_experiment(cfg))
    summary = [
        {
            "method": r.get("metrics", {}).get("method_name", config.get("method_name")),
            "params_hash": r.get("metrics", {}).get("params_hash"),
            "num_embeddings": r.get("metrics", {}).get("num_embeddings"),
            "frames_reduction": r.get("metrics", {}).get("frames_reduction"),
            "processing_time_sec": r.get("metrics", {}).get("processing_time_sec"),
            "index_name": r.get("index_name"),
        }
        for r in results
    ]
    if results:
        run_dir = Path(results[0]["artifacts_dir"])
        write_json(run_dir / "sweep_summary.json", summary)
    return results


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = maybe_load_config(args.config)
    if config is None:
        raise RuntimeError("Config not loaded")
    if config.get("sweep"):
        results = run_sweep(config)
        print(f"Runs: {len(results)}")
    else:
        result = run_experiment(config)
        print(f"Run: {result['run_id']} Index: {result['index_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
