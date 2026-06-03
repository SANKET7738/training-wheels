from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from configs.data import DataConfig
from configs.distillation import DistillationConfig
from configs.model import ModelConfig
from configs.trainer import TrainerConfig


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_experiment_config(
    path: str | Path,
) -> tuple[ModelConfig, DataConfig, TrainerConfig, DistillationConfig | None, str]:
    """Load a YAML experiment config and return
    (model, data, trainer, distillation, run_name).

    The YAML may include the optional top-level keys `model`, `data`,
    `trainer`, `distillation`, `run_name`. Missing fields fall back to
    pydantic defaults. `distillation` is None when absent, which signals
    a standard training run; when present, it triggers the distillation
    code path. If `run_name` is absent it's auto-derived as
    `<filename-stem>_<timestamp>`.
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

    distill_raw = raw.get("distillation")
    distill_cfg = DistillationConfig(**distill_raw) if distill_raw else None

    run_name = raw.get("run_name") or f"{path.stem}_{_now_stamp()}"

    return model_cfg, data_cfg, trainer_cfg, distill_cfg, run_name
