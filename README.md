- FastAPI backend
- HTML UI to trigger ingest, experiments, and tests

---
## 2. Project Structure

```
magdis/
  app/                # Web UI (FastAPI + HTML)
    main.py
    index.html
  scripts/            # Core logic and research pipeline
    clip_model.py
    ingest_video.py
    es_index.py
    search.py
    sampling.py
    scene.py
    experiment.py
    evaluation.py
    utils.py
  data/
    videos/           # Input videos
    indexes/          # Optional local artifacts
    embeddings/       # Optional local artifacts
    cache/            # Optional local artifacts
    experiments/      # Logs of experiments and evaluations
  req.txt             # Requirements
```

---
## 3. How It Works (Technical + Logical Flow)

### A) Baseline Ingest + Search
1. **Video → frames**  
2. **CLIP image embeddings**  
3. **Index frames in ES (dense_vector)**  
4. **Text query → CLIP text embedding**  
5. **Cosine similarity search in ES**  

### B) Research Mode (Sampling + Scenes)
1. **Video → frames**
2. **Sampling strategy selects subset**
3. **Embeddings for selected frames**
4. **Scene detection**
5. **Scene embedding aggregation**
6. **Index scenes in ES**
7. **Evaluation module compares methods**

---
## 4. Core Modules

### `scripts/clip_model.py`
Loads CLIP, exposes:
- `encode_image(pil_image)`
- `encode_text(text)`

### `scripts/ingest_video.py`
Baseline ingestion (uniform sampling):
- Reads video
- Extracts frames at fixed FPS
- Indexes frames into ES

### `scripts/es_index.py`
Elasticsearch index helpers:
- `ensure_index` for frame embeddings
- `ensure_scene_index` for scene embeddings
- `index_frame` / `index_scene`

### `scripts/sampling.py`
Defines the interface and implementations of sampling strategies.

### `scripts/scene.py`
Scene boundary detection and embedding aggregation.

### `scripts/experiment.py`
Runs a full experiment:
- Sampling → embeddings → scene detection → indexing
- Logs everything to `data/experiments/`

### `scripts/evaluation.py`
Runs evaluation:
- Query set over multiple indices
- Aggregates metrics

### `app/main.py`, `app/index.html`
Minimal Web UI to trigger:
- ingest
- experiment
- evaluation

---
## 5. Running the System

### 5.1 Baseline CLI ingest + search
```powershell
python -m scripts.main --video data\videos\sample.mp4 --camera-id cam01 --fps 1 --query "красная машина" --top-k 5
```

### 5.2 Web UI
```powershell
uvicorn app.main:app --reload
```
Open: `http://127.0.0.1:8000`

---
## 5.3 Basic Test

Run a minimal end-to-end check:
```powershell
python -m scripts.tests.test_pipeline_basic
```

---
## 6. Run Experiment From Config

Create a config in `configs/` (JSON or YAML). Example: `configs/experiment_sample.json`.

Run:
```powershell
python -m scripts.experiment --config configs/experiment_sample.json
```

Artifacts appear under:
```
data/experiments/{run_id}/
```

Files:
- `config.json`
- `selected_frames.json`
- `segments.json`
- `scenes.json`
- `metrics.json`
- `es_index_name.txt`
- `logs.txt`

---
## 7. Run Tests / Evaluation

Use the testing module to compare multiple methods:
```powershell
python - <<'PY'
from scripts.evaluation import run_tests
print(run_tests(
    index_names=["video_cam01_20260203__method_embedding_delta__params_ab12cd__run_20260205_120000"],
    queries=["red car", "person walking"],
    top_k=5,
))
PY
```

Results saved to:
```
data/experiments/{run_id}/results.json
data/experiments/{run_id}/queries_results/
```

---
## 8. Index Format (Scenes)

Index name pattern:
```
video_{video_id}__method_{method_name}__params_{params_hash}__run_{YYYYmmdd_HHMMSS}
```

Fields in ES:
- `scene_id`
- `video_id`
- `method_name`
- `params_hash`
- `start_time`
- `end_time`
- `keyframe_path`
- `frame_count`
- `scene_embedding` (dense_vector)

---
## 9. Grid Search (Sweep)

Add a `sweep` section to the config:
```json
{
  "sweep": {
    "sampling_params": {
      "delta_threshold": [0.1, 0.2, 0.3],
      "min_fps": [0.5, 1.0]
    }
  }
}
```

Run:
```powershell
python -m scripts.experiment --config configs/experiment_sample.json
```

Each run creates its own `data/experiments/{run_id}/` directory. A `sweep_summary.json` is stored in the first run directory.

---
## 6. Experiment Workflow (Research Mode)

### Run Experiment
API endpoint: `POST /api/experiment/run`

Example payload:
```json
{
  "video_path": "data/videos/sample.mp4",
  "video_id": "cam01_2023_11_05",
  "method_name": "embedding_delta",
  "sampling_params": {
    "delta_threshold": 0.2,
    "window_size": 3,
    "min_fps": 0.5,
    "max_fps": 2.0
  },
  "scene_params": {
    "abs_threshold": 0.6,
    "rel_drop": 0.2,
    "local_min_threshold": 0.55,
    "use_local_min": true
  },
  "aggregation": "mean"
}
```

### Run Evaluation
API endpoint: `POST /api/test`

Example payload:
```json
{
  "index_names": [
    "video_cam01_2023_11_05__method_embedding_delta__params_ab12cd__run_20260203_120000"
  ],
  "queries": ["red car", "person walking"],
  "top_k": 5
}
```

---
## 7. Reproducibility
Every experiment:
- produces a **unique ES index**
- saves a log with:
  - configuration
  - selected frames
  - scenes
  - index name
  - processing time
  - counts

This makes it possible to reproduce and compare runs.

---
## 8. Notes / Constraints
- Only **offline video files** (no streaming).
- Only **Elasticsearch** as vector DB.
- **CLIP** is the shared embedding space.
- **Cosine similarity** for search.
- Accuracy + reproducibility > speed.
