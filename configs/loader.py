from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from configs.data import DataConfig
from configs.model import ModelConfig
from configs.trainer import TrainerConfig


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_experiment_config(
    path: str | Path,
) -> tuple[ModelConfig, DataConfig, TrainerConfig, str]:
    """Load a YAML experiment config and return (model, data, trainer, run_name).

    The YAML is expected to have optional top-level keys: `model`, `data`,
    `trainer`, `run_name`. Any missing field falls back to the pydantic default.
    If `run_name` is absent, it's auto-derived as `<filename-stem>_<timestamp>`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    model_cfg = ModelConfig(**(raw.get("model") or {}))
    data_cfg = DataConfig(**(raw.get("data") or {}))
    trainer_cfg = TrainerConfig(**(raw.get("trainer") or {}))

    run_name = raw.get("run_name") or f"{path.stem}_{_now_stamp()}"

    return model_cfg, data_cfg, trainer_cfg, run_name
