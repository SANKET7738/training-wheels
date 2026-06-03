from pathlib import Path

from pydantic import BaseModel, Field


class DistillationConfig(BaseModel):
    teacher_ckpt: Path
    temperature: float = Field(default=3.0, gt=0.0)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
