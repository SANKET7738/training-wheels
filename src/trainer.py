from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import DataConfig, ModelConfig, OptimizerConfig, TrainerConfig
from .data import MNISTDataModule
from .model import build_model

ASSETS_ROOT = Path("assets")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(params, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {cfg.name}")


def _setup_run_dirs(run_name: str) -> dict[str, Path]:
    run_root = ASSETS_ROOT / run_name
    dirs = {
        "root": run_root,
        "ckpts": run_root / "ckpts",
        "plots": run_root / "plots",
        "logs": run_root / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _setup_logger(run_name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"trainer.{run_name}")
    logger.setLevel(logging.INFO)
    # Remove any pre-existing handlers to avoid duplicates across re-runs in the
    # same process (e.g., notebooks, tests).
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _save_curves(history: dict[str, list[float]], plots_dir: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], marker="o", label="train")
    if history["test_loss"]:
        ax.plot(epochs, history["test_loss"], marker="s", label="test")
    if history["val_loss"]:
        ax.plot(epochs, history["val_loss"], marker="^", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    if history["test_acc"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history["test_acc"], marker="s", label="test")
        if history["val_acc"]:
            ax.plot(epochs, history["val_acc"], marker="^", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "acc_curve.png", dpi=120)
        plt.close(fig)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        datamodule: MNISTDataModule,
        cfg: TrainerConfig,
        run_name: str,
        model_cfg: ModelConfig | None = None,
        data_cfg: DataConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        _seed_everything(cfg.seed)

        self.dirs = _setup_run_dirs(run_name)
        self.logger = _setup_logger(run_name, self.dirs["logs"] / "train.log")

        self.device = _resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.datamodule = datamodule
        self.optimizer = _build_optimizer(self.model.parameters(), cfg.optimizer)
        self.loss_fn = nn.CrossEntropyLoss()

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        n_samples = 0
        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            if step % self.cfg.log_interval == 0:
                self.logger.info(
                    f"  epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}"
                )

        return running_loss / max(n_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n_samples = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n_samples += x.size(0)
        return {
            "loss": total_loss / max(n_samples, 1),
            "acc": correct / max(n_samples, 1),
        }

    def _save_final_ckpt(self, final_metrics: dict[str, float]) -> None:
        ckpt_path = self.dirs["ckpts"] / "final.pt"
        payload: dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "trainer_config": self.cfg.model_dump(mode="json"),
            "final_metrics": final_metrics,
            "run_name": self.run_name,
        }
        if self.model_cfg is not None:
            payload["model_config"] = self.model_cfg.model_dump(mode="json")
        if self.data_cfg is not None:
            payload["data_config"] = self.data_cfg.model_dump(mode="json")
        torch.save(payload, ckpt_path)
        self.logger.info(f"Saved final checkpoint: {ckpt_path}")

    def fit(self) -> dict[str, Any]:
        self.datamodule.setup()
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model params: {n_params:,}")
        self.logger.info(f"Assets dir: {self.dirs['root']}")

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
            "epoch_time_s": [],
        }

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch)
            elapsed = time.time() - t0
            history["train_loss"].append(train_loss)
            history["epoch_time_s"].append(elapsed)

            line = f"epoch {epoch} train_loss={train_loss:.4f} time={elapsed:.1f}s"

            if self.cfg.eval_every_epoch:
                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader)
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_acc"].append(val_metrics["acc"])
                    line += f" val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
                test_metrics = self.evaluate(test_loader)
                history["test_loss"].append(test_metrics["loss"])
                history["test_acc"].append(test_metrics["acc"])
                line += f" test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}"

            self.logger.info(line)

        final_test = self.evaluate(test_loader)
        result: dict[str, Any] = {
            "run_name": self.run_name,
            "device": str(self.device),
            "num_params": n_params,
            "history": history,
            "final_test_loss": final_test["loss"],
            "final_test_acc": final_test["acc"],
        }

        with open(self.dirs["logs"] / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        _save_curves(history, self.dirs["plots"])
        self.logger.info(f"Saved plots to {self.dirs['plots']}")

        self._save_final_ckpt({
            "test_loss": final_test["loss"],
            "test_acc": final_test["acc"],
        })

        return result


def train_mlp(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    trainer_cfg: TrainerConfig,
    run_name: str,
) -> dict[str, Any]:
    """End-to-end MLP training orchestrator. Builds the datamodule, model, and
    Trainer, runs `.fit()`, and returns the result dict. All artifacts are
    written under `assets/<run_name>/`.
    """
    datamodule = MNISTDataModule(data_cfg)
    datamodule.prepare()

    model = build_model(model_cfg)
    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        cfg=trainer_cfg,
        run_name=run_name,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )
    return trainer.fit()
