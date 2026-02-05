from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_run_id() -> str:
    """Return run_id in format YYYYmmdd_HHMMSS (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def params_hash(params: dict[str, Any]) -> str:
    """Deterministic short hash of params (sorted JSON)."""
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:6]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_load_config(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".json"}:
        return read_json(p)
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML configs") from exc
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Unsupported config format. Use JSON or YAML.")


def to_iso_z(seconds: float) -> str:
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta_safe(seconds)
    return dt.isoformat().replace("+00:00", "Z")


def timedelta_safe(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)


def safe_text(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


def log_line(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + os.linesep)
