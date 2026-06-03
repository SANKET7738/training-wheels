from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    arch: Literal["mlp"] = "mlp"
    input_dim: int = Field(default=784, gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [256])
    num_classes: int = Field(default=10, gt=0)
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @field_validator("hidden_dims")
    @classmethod
    def _validate_hidden_dims(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("hidden_dims must contain at least one layer width")
        if any(h <= 0 for h in v):
            raise ValueError("hidden_dims entries must be positive")
        return v
