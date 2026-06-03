from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    name: Literal["mnist"] = "mnist"
    root: Path = Path("./data")
    batch_size: int = Field(default=128, gt=0)
    eval_batch_size: int = Field(default=256, gt=0)
    num_workers: int = Field(default=2, ge=0)
    val_split: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalize: bool = True
