from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from configs import DataConfig

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MNISTDataModule:
    def __init__(self, cfg: DataConfig) -> None:
        if cfg.name != "mnist":
            raise ValueError(f"Unsupported dataset: {cfg.name}")
        self.cfg = cfg
        self._train: torch.utils.data.Dataset | Subset | None = None
        self._val: Subset | None = None
        self._test: torch.utils.data.Dataset | None = None

    def _build_transform(self) -> transforms.Compose:
        ops: list = [transforms.ToTensor()]
        if self.cfg.normalize:
            ops.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
        return transforms.Compose(ops)

    def prepare(self) -> None:
        datasets.MNIST(str(self.cfg.root), train=True, download=True)
        datasets.MNIST(str(self.cfg.root), train=False, download=True)

    def setup(self) -> None:
        transform = self._build_transform()
        train_full = datasets.MNIST(
            str(self.cfg.root), train=True, download=True, transform=transform
        )
        self._test = datasets.MNIST(
            str(self.cfg.root), train=False, download=True, transform=transform
        )

        if self.cfg.val_split > 0.0:
            val_size = int(len(train_full) * self.cfg.val_split)
            train_size = len(train_full) - val_size
            generator = torch.Generator().manual_seed(0)
            self._train, self._val = random_split(
                train_full, [train_size, val_size], generator=generator
            )
        else:
            self._train = train_full
            self._val = None

    def _loader(self, dataset, *, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None, "Call setup() first"
        return self._loader(self._train, shuffle=True, batch_size=self.cfg.batch_size)

    def val_dataloader(self) -> DataLoader | None:
        if self._val is None:
            return None
        return self._loader(self._val, shuffle=False, batch_size=self.cfg.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None, "Call setup() first"
        return self._loader(self._test, shuffle=False, batch_size=self.cfg.eval_batch_size)
