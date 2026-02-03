from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from scripts.ingest_video import ingest_video
from scripts.search import search_text

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
    }
    return JSONResponse(json.loads(json.dumps(example)))
