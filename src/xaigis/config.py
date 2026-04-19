from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["__config_path__"] = path
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), base=path.parent)
    return cfg


def write_default_config(out_path: str | Path, cfg: dict[str, Any]) -> Path:
    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def _resolve_paths(paths: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in paths.items():
        resolved[key] = _resolve_path_value(value, base=base)
    return resolved


def _resolve_path_value(value: Any, base: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        return p
    if isinstance(value, list):
        return [_resolve_path_value(item, base=base) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_path_value(v, base=base) for k, v in value.items()}
    return value
