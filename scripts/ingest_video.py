from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import cv2

from .clip_model import encode_image_bgr
from .es_index import INDEX_NAME, get_es_client, index_frame
from .utils import to_iso_z


def ingest_video(video_path: str, camera_id: str, fps: float = 1.0) -> int:
    """Baseline ingestion (uniform FPS) into the legacy frames index."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    es = get_es_client()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 25.0

    next_sample_time = 0.0
    frame_index = -1
    indexed = 0

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
            if ts_sec + 1e-9 < next_sample_time:
                continue

            embedding = encode_image_bgr(frame)
            document = {
                "camera_id": camera_id,
                "source": "archive",
                "video_path": os.path.abspath(video_path),
                "timestamp": to_iso_z(ts_sec),
                "frame_index": frame_index,
                "frame_id": f"{camera_id}_{uuid.uuid4().hex}",
                "embedding": embedding.tolist(),
                "metadata": {},
            }
            index_frame(document, es=es, index_name=INDEX_NAME)
            indexed += 1
            next_sample_time += 1.0 / fps
    finally:
        cap.release()

    if indexed > 0:
        es.indices.refresh(index=INDEX_NAME)
    return indexed
