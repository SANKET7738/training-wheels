from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configs.loader import load_experiment_config
from src.trainer import train_mlp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MNIST MLP experiment from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment config (see experiment-configs/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_cfg, data_cfg, trainer_cfg, run_name = load_experiment_config(args.config)
    result = train_mlp(model_cfg, data_cfg, trainer_cfg, run_name)
    print()
    print(f"Run: {result['run_name']}")
    print(f"Final test acc:  {result['final_test_acc']:.4f}")
    print(f"Final test loss: {result['final_test_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
