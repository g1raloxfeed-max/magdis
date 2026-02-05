from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .clip_model import encode_text
from .es_index import ensure_scene_index, get_es_client
from .utils import ensure_dir, generate_run_id, write_json

DATA_DIR = Path("data")
EXPERIMENTS_DIR = DATA_DIR / "experiments"


def _search(index_name: str, query: str, top_k: int) -> dict[str, Any]:
    es = get_es_client()
    query_vector = encode_text(query)
    ensure_scene_index(es, len(query_vector), index_name=index_name)
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'scene_embedding') + 1.0",
                    "params": {"query_vector": query_vector.tolist()},
                },
            }
        },
    }
    return es.search(index=index_name, body=body)


def run_tests(
    index_names: list[str],
    queries: list[str],
    top_k: int,
    q_meta: dict | None = None,
) -> dict:
    """Run queries against indices, collect metrics, return aggregated table.

    Example:
    >>> run_tests(["video_cam01__method_uniform__params_ab12cd__run_20260205_120000"], ["red car"], 5)
    """
    run_id = generate_run_id()
    run_dir = ensure_dir(EXPERIMENTS_DIR / run_id)
    results_dir = ensure_dir(run_dir / "queries_results")

    table = []
    for index_name in index_names:
        total_latency = 0.0
        total_similarity = 0.0
        total_hits = 0
        stability_scores = []

        for q in queries:
            t0 = time.perf_counter()
            resp1 = _search(index_name, q, top_k)
            latency = time.perf_counter() - t0
            total_latency += latency
            hits1 = resp1.get("hits", {}).get("hits", [])
            total_hits += len(hits1)
            if hits1:
                total_similarity += sum((h.get("_score", 0.0) - 1.0) for h in hits1) / len(hits1)

            resp2 = _search(index_name, q, top_k)
            hits2 = resp2.get("hits", {}).get("hits", [])
            ids1 = {h.get("_id") for h in hits1}
            ids2 = {h.get("_id") for h in hits2}
            if ids1 or ids2:
                stability_scores.append(len(ids1 & ids2) / len(ids1 | ids2))
            else:
                stability_scores.append(1.0)

            write_json(results_dir / f"{safe_name(index_name)}__{safe_name(q)}.json", resp1)

        avg_similarity = total_similarity / max(len(queries), 1)
        avg_latency = total_latency / max(len(queries), 1)
        avg_stability = sum(stability_scores) / max(len(stability_scores), 1)

        table.append(
            {
                "index": index_name,
                "num_scenes": total_hits,
                "avg_similarity": avg_similarity,
                "latency": avg_latency,
                "stability": avg_stability,
                "storage_cost": "N/A",
            }
        )

    out = {
        "run_id": run_id,
        "queries": queries,
        "q_meta": q_meta or {},
        "results": table,
        "top_k": top_k,
    }
    write_json(run_dir / "results.json", out)
    return {"run_id": run_id, "results": table, "artifacts_dir": str(run_dir)}


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
