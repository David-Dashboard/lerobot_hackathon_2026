"""Load and access the shared YAML config (ports, calibration paths, limits, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml into a plain nested dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {cfg_path} did not parse to a mapping")
    cfg["_path"] = str(cfg_path)
    return cfg


def resolve_path(cfg: dict, value: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
