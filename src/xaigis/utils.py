from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
