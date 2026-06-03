from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from configs.loader import load_experiment_config
from src.distillation import train_distillation_mlp
from src.trainer import train_mlp

ASSETS_ROOT = Path("assets")


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


def _load_run_metrics(run_name: str) -> dict[str, Any] | None:
    path = ASSETS_ROOT / run_name / "logs" / "metrics.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _print_distillation_comparison(distill_result: dict[str, Any]) -> None:
    """Print a 3-row comparison of teacher / student_small / current
    distillation run, if the reference runs' metrics are on disk.
    Gracefully degrades when references are missing.
    """
    teacher = _load_run_metrics("teacher")
    student_baseline = _load_run_metrics("student_small")

    print()
    print("=== Distillation comparison ===")
    rows: list[tuple[str, dict[str, Any] | None, str]] = [
        ("teacher", teacher, "(reference)"),
        ("student_small", student_baseline, "hard labels only"),
        (
            distill_result["run_name"],
            distill_result,
            "soft+hard (T={t} α={a})".format(
                t=distill_result.get("distill_temperature", "?"),
                a=distill_result.get("distill_alpha", "?"),
            ),
        ),
    ]
    for name, m, note in rows:
        if m is None:
            print(f"  {name:18s}  <metrics.json missing>  {note}")
            continue
        acc = m.get("final_test_acc")
        params = m.get("num_params", "?")
        if isinstance(params, int):
            params_str = f"{params:>7,}"
        else:
            params_str = f"{params:>7}"
        agreement = m.get("final_test_agreement")
        agree_str = f"  agree={agreement:.4f}" if agreement is not None else ""
        print(f"  {name:18s}  acc={acc:.4f}  params={params_str}  {note}{agree_str}")

    if student_baseline is not None:
        delta = distill_result["final_test_acc"] - student_baseline["final_test_acc"]
        print(f"  Δ vs student_small (hard only): {delta:+.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_cfg, data_cfg, trainer_cfg, distill_cfg, run_name = load_experiment_config(
        args.config
    )

    if distill_cfg is None:
        result = train_mlp(model_cfg, data_cfg, trainer_cfg, run_name)
    else:
        result = train_distillation_mlp(
            model_cfg, data_cfg, trainer_cfg, distill_cfg, run_name
        )
        # Carry the hyperparams onto the result for the comparison printout.
        result["distill_temperature"] = distill_cfg.temperature
        result["distill_alpha"] = distill_cfg.alpha

    print()
    print(f"Run: {result['run_name']}")
    print(f"Final test acc:  {result['final_test_acc']:.4f}")
    print(f"Final test loss: {result['final_test_loss']:.4f}")
    if "final_test_agreement" in result:
        print(f"Final test agreement (teacher↔student): {result['final_test_agreement']:.4f}")

    if distill_cfg is not None:
        _print_distillation_comparison(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
