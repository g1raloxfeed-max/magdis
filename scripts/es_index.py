from __future__ import annotations

import os
from typing import Any

from elasticsearch import Elasticsearch

INDEX_NAME = os.getenv("ES_INDEX", "frames")


def get_es_client() -> Elasticsearch:
    url = os.getenv("ES_URL", "http://localhost:9200")
    api_key = os.getenv("ES_API_KEY")
    user = os.getenv("ES_USER")
    password = os.getenv("ES_PASSWORD")

    if api_key:
        return Elasticsearch(url, api_key=api_key)
    if user and password:
        return Elasticsearch(url, basic_auth=(user, password))
    return Elasticsearch(url)


def ensure_index(es: Elasticsearch, dims: int, index_name: str = INDEX_NAME) -> None:
    if es.indices.exists(index=index_name):
        mapping = es.indices.get_mapping(index=index_name)
        try:
            existing_dims = mapping[index_name]["mappings"]["properties"]["embedding"]["dims"]
        except Exception:
            existing_dims = None
        if existing_dims is not None and existing_dims != dims:
            raise ValueError(
                f"Index {index_name} has dims={existing_dims}, expected {dims}"
            )
        return

    mappings = {
        "properties": {
            "camera_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "video_path": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "frame_index": {"type": "integer"},
            "frame_id": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {"type": "object"},
        }
    }
    settings = {"index": {"number_of_shards": 1, "number_of_replicas": 0}}
    es.indices.create(index=index_name, mappings=mappings, settings=settings)


def index_frame(
    document: dict[str, Any],
    es: Elasticsearch | None = None,
    index_name: str = INDEX_NAME,
) -> dict[str, Any]:
    if es is None:
        es = get_es_client()
    embedding = document.get("embedding")
    if embedding is None:
        raise ValueError("Document is missing 'embedding'")
    ensure_index(es, len(embedding), index_name=index_name)
    return es.index(index=index_name, id=document.get("frame_id"), document=document)
