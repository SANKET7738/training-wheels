from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[cfg.activation]

        layers: list[nn.Module] = [nn.Flatten()]
        prev = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg)
    raise ValueError(f"Unsupported arch: {cfg.arch}")


def load_model_from_ckpt(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
    freeze: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint produced by Trainer._save_final_ckpt and return
    (model, payload).

    Reconstructs the model from the saved `model_config`, loads the
    state dict, moves to `device`, and optionally puts it in eval mode
    and freezes parameters (no grad). The full payload is also returned
    so callers can inspect saved configs / final metrics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    model_cfg_raw = payload.get("model_config")
    if model_cfg_raw is None:
        raise ValueError(
            f"Checkpoint {path} has no 'model_config' key; cannot reconstruct model."
        )

    model_cfg = ModelConfig(**model_cfg_raw)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])

    if eval_mode:
        model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model, payload
