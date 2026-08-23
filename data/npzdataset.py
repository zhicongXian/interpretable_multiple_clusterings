from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from pathlib import Path
import torch
from torch import Tensor, nn


class NpzImageDataset(Dataset[tuple[Tensor, int]]):
    """Load only images and sample indices; labels are deliberately unused."""

    def __init__(self, path: str | Path, split: str) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'.")
        archive = np.load(path, allow_pickle=False)
        if "images" not in archive:
            raise KeyError("The NPZ archive must contain an 'images' array.")

        images = archive["images"]
        if images.ndim != 4:
            raise ValueError("images must have shape [N,C,H,W] or [N,H,W,C].")
        if images.shape[1] not in {1, 3, 4} and images.shape[-1] in {1, 3, 4}:
            images = np.transpose(images, (0, 3, 1, 2))
        if images.shape[1] == 4:
            images = images[:, :3]

        images = images.astype(np.float32, copy=False)
        if float(images.max(initial=0.0)) > 1.0:
            images = images / 255.0

        if split == "all":
            indices = np.arange(len(images), dtype=np.int64)
        else:
            key = f"{split}_indices"
            indices = (
                archive[key].astype(np.int64, copy=False)
                if key in archive
                else np.arange(len(images), dtype=np.int64)
            )
        self.images = torch.from_numpy(np.ascontiguousarray(images[indices]))
        self.original_indices = torch.from_numpy(np.asarray(indices).copy())

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.images[index], int(self.original_indices[index])