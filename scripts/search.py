from __future__ import annotations

from typing import Any

from .clip_model import encode_text
from .es_index import INDEX_NAME, ensure_index, get_es_client


def search_text(
    query: str,
    top_k: int = 5,
    es=None,
    index_name: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if es is None:
        es = get_es_client()

    query_vector = encode_text(query)
    ensure_index(es, len(query_vector), index_name=index_name)

    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector.tolist()},
                },
            }
        },
    }
    response = es.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])

    results: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.get("_source", {})
        results.append(
            {
                "score": hit.get("_score", 0.0),
                "camera_id": src.get("camera_id"),
                "video_path": src.get("video_path"),
                "timestamp": src.get("timestamp"),
                "frame_index": src.get("frame_index"),
                "frame_id": src.get("frame_id"),
            }
        )
    return results
