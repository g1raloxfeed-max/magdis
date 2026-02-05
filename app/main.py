from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from scripts.ingest_video import ingest_video
from scripts.search import search_text
from scripts.experiment import run_experiment
from scripts.evaluation import run_tests
from scripts.utils import maybe_load_config

APP_DIR = Path(__file__).parent
INDEX_HTML = APP_DIR / "index.html"

app = FastAPI(title="Magdis Search UI")


class IngestRequest(BaseModel):
    video_path: str = Field(..., description="Path to local video file")
    camera_id: str = Field(..., description="Camera ID")
    fps: float = Field(1.0, description="Sampling FPS")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text query")
    top_k: int = Field(5, description="Top-K results")


class ExperimentRequest(BaseModel):
    config: dict | None = None
    config_path: str | None = None


class EvaluationRequest(BaseModel):
    index_names: list[str]
    queries: list[str]
    top_k: int = 5
    q_meta: dict | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/api/ingest")
def api_ingest(payload: IngestRequest) -> JSONResponse:
    if not os.path.isfile(payload.video_path):
        raise HTTPException(status_code=400, detail="Video file not found")
    try:
        indexed = ingest_video(payload.video_path, payload.camera_id, fps=payload.fps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"indexed": indexed})


@app.post("/api/search")
def api_search(payload: SearchRequest) -> JSONResponse:
    try:
        results = search_text(payload.query, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"results": results})


@app.post("/api/experiment/run")
def api_experiment_run(payload: ExperimentRequest) -> JSONResponse:
    try:
        config = payload.config or maybe_load_config(payload.config_path)
        if config is None:
            raise ValueError("config or config_path must be provided")
        result = run_experiment(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/experiment/test")
def api_test(payload: EvaluationRequest) -> JSONResponse:
    try:
        result = run_tests(
            index_names=payload.index_names,
            queries=payload.queries,
            top_k=payload.top_k,
            q_meta=payload.q_meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/api/health")
def api_health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/example")
def api_example() -> JSONResponse:
    example = {
        "ingest": {
            "video_path": "data/videos/sample.mp4",
            "camera_id": "cam01",
            "fps": 1.0,
        },
        "search": {"query": "red car", "top_k": 5},
        "experiment": {
            "config_path": "configs/experiment_sample.json",
        },
        "test": {
            "index_names": ["video_cam01_2023_11_05__method_embedding_delta__params_ab12cd__run_20260203_120000"],
            "queries": ["red car", "person walking"],
            "top_k": 5,
        },
    }
    return JSONResponse(json.loads(json.dumps(example)))
