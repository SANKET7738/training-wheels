from __future__ import annotations

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
