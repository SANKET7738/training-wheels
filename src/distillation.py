from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import (
    DataConfig,
    DistillationConfig,
    ModelConfig,
    TrainerConfig,
)
from .data import MNISTDataModule
from .model import build_model, load_model_from_ckpt
from .trainer import Trainer, _save_components_plot


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float = 3.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hinton-style knowledge distillation loss.

    Returns (total, soft, hard) where:
      soft = KL(softmax(student/T) || softmax(teacher/T)) * T^2
      hard = cross_entropy(student_logits, labels)
      total = alpha * soft + (1 - alpha) * hard

    The T^2 scaling preserves gradient magnitudes relative to the hard
    term as T varies (see Hinton et al., "Distilling the Knowledge in a
    Neural Network").
    """
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    total = alpha * soft + (1.0 - alpha) * hard
    return total, soft, hard


class DistillationTrainer(Trainer):
    """Knowledge distillation trainer. Inherits the base training loop;
    overrides per-batch loss to combine soft (teacher) and hard (label)
    targets, and adds a teacher-student agreement metric at eval time.
    """

    def __init__(
        self,
        model: nn.Module,
        datamodule: MNISTDataModule,
        cfg: TrainerConfig,
        run_name: str,
        distill_cfg: DistillationConfig,
        teacher: nn.Module,
        model_cfg: ModelConfig | None = None,
        data_cfg: DataConfig | None = None,
    ) -> None:
        super().__init__(
            model=model,
            datamodule=datamodule,
            cfg=cfg,
            run_name=run_name,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
        )
        self.distill_cfg = distill_cfg
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.logger.info(
            f"Distillation: T={distill_cfg.temperature} alpha={distill_cfg.alpha} "
            f"teacher_ckpt={distill_cfg.teacher_ckpt}"
        )

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        T = self.distill_cfg.temperature
        alpha = self.distill_cfg.alpha

        running_total = 0.0
        running_soft = 0.0
        running_hard = 0.0
        correct = 0
        n_samples = 0

        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = self.teacher(x)

            self.optimizer.zero_grad(set_to_none=True)
            student_logits = self.model(x)
            total, soft, hard = distillation_loss(
                student_logits, teacher_logits, y, T=T, alpha=alpha
            )
            total.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            bs = x.size(0)
            running_total += total.item() * bs
            running_soft += soft.item() * bs
            running_hard += hard.item() * bs
            with torch.no_grad():
                correct += (student_logits.argmax(dim=1) == y).sum().item()
            n_samples += bs

            if step % self.cfg.log_interval == 0:
                self.logger.info(
                    f"  epoch {epoch} step {step}/{len(loader)} "
                    f"loss={total.item():.4f} soft={soft.item():.4f} hard={hard.item():.4f}"
                )

        n = max(n_samples, 1)
        return {
            "loss": running_total / n,
            "acc": correct / n,
            "soft_loss": running_soft / n,
            "hard_loss": running_hard / n,
        }

    @torch.no_grad()
    def _extra_test_metrics(self, loader: DataLoader) -> dict[str, float]:
        """Compute fraction of test samples where student and teacher
        agree on the argmax prediction. A distillation-specific signal:
        the student can have high test accuracy yet still disagree with
        the teacher on many examples (or vice versa).
        """
        self.model.eval()
        self.teacher.eval()
        agree = 0
        n_samples = 0
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            student_pred = self.model(x).argmax(dim=1)
            teacher_pred = self.teacher(x).argmax(dim=1)
            agree += (student_pred == teacher_pred).sum().item()
            n_samples += x.size(0)
        return {"agreement": agree / max(n_samples, 1)}

    def _save_extra_plots(
        self, history: dict[str, list[float]], plots_dir: Path
    ) -> None:
        _save_components_plot(
            history,
            plots_dir,
            filename="loss_components.png",
            title="Train loss components",
            keys=["train_loss", "train_soft_loss", "train_hard_loss"],
        )


def _evaluate_teacher_acc(teacher: nn.Module, loader: DataLoader, device: torch.device) -> float:
    teacher.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            correct += (teacher(x).argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return correct / max(n, 1)


def train_distillation_mlp(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    trainer_cfg: TrainerConfig,
    distill_cfg: DistillationConfig,
    run_name: str,
) -> dict[str, Any]:
    """End-to-end distillation orchestrator. Loads the frozen teacher
    from `distill_cfg.teacher_ckpt`, builds the student from `model_cfg`,
    and trains the student with the combined soft+hard loss.
    """
    if not Path(distill_cfg.teacher_ckpt).exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {distill_cfg.teacher_ckpt}. "
            "Train the teacher first (e.g., experiment-configs/teacher.yaml)."
        )

    datamodule = MNISTDataModule(data_cfg)
    datamodule.prepare()
    datamodule.setup()

    student = build_model(model_cfg)
    # Load teacher onto CPU first; the trainer will move it to its device.
    teacher, teacher_payload = load_model_from_ckpt(
        distill_cfg.teacher_ckpt, device="cpu", eval_mode=True, freeze=True
    )

    trainer = DistillationTrainer(
        model=student,
        datamodule=datamodule,
        cfg=trainer_cfg,
        run_name=run_name,
        distill_cfg=distill_cfg,
        teacher=teacher,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
    )

    # Sanity print of teacher accuracy so we know the ckpt loaded sensibly.
    teacher_acc = _evaluate_teacher_acc(
        trainer.teacher, datamodule.test_dataloader(), trainer.device
    )
    trainer.logger.info(
        f"Teacher test acc: {teacher_acc:.4f} "
        f"(from {distill_cfg.teacher_ckpt})"
    )
    teacher_run = teacher_payload.get("run_name", "<unknown>")
    trainer.logger.info(f"Teacher source run: {teacher_run}")

    return trainer.fit()
