from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)


class TrainerConfig(BaseModel):
    epochs: int = Field(default=5, gt=0)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    seed: int = 42
    log_interval: int = Field(default=100, gt=0)
    eval_every_epoch: bool = True
    grad_clip: float | None = Field(default=None, gt=0.0)
    output_dir: Path | None = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
