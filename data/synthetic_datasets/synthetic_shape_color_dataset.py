"""Generate a balanced synthetic shape--color image dataset.

The two independent clustering views are:

* shape: triangle, circle, star
* color: red, green, blue

Every shape/color combination contains the same number of samples. Random
translation, scale, rotation, brightness, background, and pixel noise act as
nuisance factors. The saved NPZ uses uint8 NCHW images and stores stratified
train/test indices.

Example
-------
python synthetic_shape_color_dataset.py \
    --samples-per-combination 100 \
    --output shape_color_multiview_dataset.npz \
    --preview shape_color_multiview_preview.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SHAPE_NAMES = ("triangle", "circle", "star")
COLOR_NAMES = ("red", "green", "blue")

RGB_COLORS = np.asarray(
    [
        [232.0, 55.0, 58.0],
        [52.0, 205.0, 82.0],
        [58.0, 94.0, 235.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GenerationConfig:
    samples_per_combination: int = 100
    image_size: int = 64
    test_fraction: float = 0.2
    seed: int = 0
    supersampling: int = 3
    noise_std: float = 3.0

    def validate(self) -> None:
        if self.samples_per_combination < 2:
            raise ValueError("samples_per_combination must be at least two.")
        if self.image_size < 24:
            raise ValueError("image_size must be at least 24.")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must lie strictly between 0 and 1.")
        if self.supersampling < 1:
            raise ValueError("supersampling must be positive.")
        n_test = round(self.samples_per_combination * self.test_fraction)
        if n_test < 1 or n_test >= self.samples_per_combination:
            raise ValueError("test_fraction produces an empty train or test split.")


def _star_points(
    center: tuple[float, float], outer_radius: float, inner_radius: float
) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        radius = outer_radius if index % 2 == 0 else inner_radius
        points.append(
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
        )
    return points


def _render_mask(
    shape_index: int,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    size = config.image_size
    high_size = size * config.supersampling
    canvas = Image.new("L", (high_size, high_size), 0)
    draw = ImageDraw.Draw(canvas)
    center = high_size / 2.0
    radius = high_size * 0.30 * float(rng.uniform(0.82, 1.12))

    if shape_index == 0:
        draw.polygon(
            [
                (center, center - radius),
                (center - 0.90 * radius, center + 0.72 * radius),
                (center + 0.90 * radius, center + 0.72 * radius),
            ],
            fill=255,
        )
    elif shape_index == 1:
        draw.ellipse(
            (center - radius, center - radius, center + radius, center + radius),
            fill=255,
        )
    elif shape_index == 2:
        draw.polygon(
            _star_points((center, center), radius, 0.44 * radius),
            fill=255,
        )
    else:
        raise ValueError(f"Unknown shape index: {shape_index}")

    maximum_shift = 0.10 * size
    shift_x, shift_y = rng.uniform(-maximum_shift, maximum_shift, size=2)
    resampling = getattr(Image, "Resampling", Image)
    canvas = canvas.rotate(
        float(rng.uniform(-30.0, 30.0)),
        resample=resampling.BICUBIC,
        translate=(
            float(shift_x * config.supersampling),
            float(shift_y * config.supersampling),
        ),
    )
    canvas = canvas.resize((size, size), resample=resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def render_sample(
    shape_index: int,
    color_index: int,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one uint8 RGB image in HWC layout."""

    size = config.image_size
    mask = _render_mask(shape_index, config, rng)

    color_jitter = rng.uniform(0.90, 1.08, size=3)
    foreground = np.clip(RGB_COLORS[color_index] * color_jitter, 0.0, 255.0)
    foreground = foreground[None, None, :]

    base = float(rng.uniform(15.0, 42.0))
    background_tint = rng.uniform(-4.0, 4.0, size=3)
    background = np.full((size, size, 3), base, dtype=np.float32)
    background += background_tint[None, None, :]

    x_gradient = np.linspace(-3.0, 3.0, size, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(2.5, -2.5, size, dtype=np.float32)[:, None, None]
    background += x_gradient + y_gradient

    alpha = mask[:, :, None]
    image = alpha * foreground + (1.0 - alpha) * background
    image += rng.normal(scale=config.noise_std, size=image.shape).astype(np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def generate_dataset(config: GenerationConfig) -> dict[str, np.ndarray]:
    """Generate balanced combinations and a per-combination stratified split."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    n_test = round(config.samples_per_combination * config.test_fraction)

    images: list[np.ndarray] = []
    shape_labels: list[int] = []
    color_labels: list[int] = []
    is_test: list[bool] = []

    for shape_index in range(len(SHAPE_NAMES)):
        for color_index in range(len(COLOR_NAMES)):
            local_test = np.zeros(config.samples_per_combination, dtype=bool)
            selected = rng.choice(
                config.samples_per_combination,
                size=n_test,
                replace=False,
            )
            local_test[selected] = True

            for local_index in range(config.samples_per_combination):
                images.append(
                    render_sample(shape_index, color_index, config, rng)
                )
                shape_labels.append(shape_index)
                color_labels.append(color_index)
                is_test.append(bool(local_test[local_index]))

    image_array = np.stack(images)
    shape_array = np.asarray(shape_labels, dtype=np.int64)
    color_array = np.asarray(color_labels, dtype=np.int64)
    combination_array = (
        shape_array * len(COLOR_NAMES) + color_array
    ).astype(np.int64)
    test_array = np.asarray(is_test, dtype=bool)

    permutation = rng.permutation(len(image_array))
    image_array = image_array[permutation]
    shape_array = shape_array[permutation]
    color_array = color_array[permutation]
    combination_array = combination_array[permutation]
    test_array = test_array[permutation]

    return {
        "images": image_array.transpose(0, 3, 1, 2),
        "shape_labels": shape_array,
        "color_labels": color_array,
        "combination_labels": combination_array,
        "train_indices": np.flatnonzero(~test_array).astype(np.int64),
        "test_indices": np.flatnonzero(test_array).astype(np.int64),
        "shape_names": np.asarray(SHAPE_NAMES),
        "color_names": np.asarray(COLOR_NAMES),
        "image_layout": np.asarray("NCHW"),
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
    }


def validate_dataset(dataset: dict[str, np.ndarray], config: GenerationConfig) -> None:
    n_combinations = len(SHAPE_NAMES) * len(COLOR_NAMES)
    n_expected = config.samples_per_combination * n_combinations
    expected_shape = (n_expected, 3, config.image_size, config.image_size)

    if dataset["images"].shape != expected_shape:
        raise AssertionError(
            f"Expected images shaped {expected_shape}, got {dataset['images'].shape}."
        )
    if dataset["images"].dtype != np.uint8:
        raise AssertionError("Images must be stored as uint8.")

    counts = np.bincount(
        dataset["combination_labels"], minlength=n_combinations
    )
    if not np.all(counts == config.samples_per_combination):
        raise AssertionError(f"Unbalanced combinations: {counts.tolist()}")

    train = dataset["train_indices"]
    test = dataset["test_indices"]
    if np.intersect1d(train, test).size:
        raise AssertionError("Train and test indices overlap.")
    if len(train) + len(test) != n_expected:
        raise AssertionError("Train and test indices do not cover the dataset.")


def save_preview(dataset: dict[str, np.ndarray], output_path: Path) -> None:
    """Save one representative image for each shape/color combination."""

    images = dataset["images"]
    labels = dataset["combination_labels"]
    image_size = images.shape[-1]
    cell_width = image_size + 14
    cell_height = image_size + 12
    left_margin = 82
    top_margin = 34

    canvas = Image.new(
        "RGB",
        (
            left_margin + len(COLOR_NAMES) * cell_width,
            top_margin + len(SHAPE_NAMES) * cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for color_index, color_name in enumerate(COLOR_NAMES):
        x = left_margin + color_index * cell_width + 10
        draw.text((x, 9), color_name, fill="black", font=font)

    for shape_index, shape_name in enumerate(SHAPE_NAMES):
        y = top_margin + shape_index * cell_height
        draw.text(
            (7, y + image_size // 2 - 7),
            shape_name,
            fill="black",
            font=font,
        )
        for color_index in range(len(COLOR_NAMES)):
            combination = shape_index * len(COLOR_NAMES) + color_index
            sample_index = int(np.flatnonzero(labels == combination)[0])
            sample = images[sample_index].transpose(1, 2, 0)
            tile = Image.fromarray(sample, mode="RGB")
            x = left_margin + color_index * cell_width
            canvas.paste(tile, (x, y))

    canvas.save(output_path)


class ShapeColorTorchDataset:
    """PyTorch adapter that imports torch only when a sample is requested."""

    def __init__(self, npz_path: str | Path, split: str = "train") -> None:
        archive = np.load(npz_path, allow_pickle=False)
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'.")
        self.images = archive["images"]
        self.shape_labels = archive["shape_labels"]
        self.color_labels = archive["color_labels"]
        self.combination_labels = archive["combination_labels"]

        if split == "train":
            self.indices = archive["train_indices"]
        elif split == "test":
            self.indices = archive["test_indices"]
        else:
            self.indices = np.arange(len(self.images), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch

        sample_index = int(self.indices[index])
        image = torch.from_numpy(self.images[sample_index].copy()).float() / 255.0
        labels = {
            "shape": torch.tensor(self.shape_labels[sample_index]),
            "color": torch.tensor(self.color_labels[sample_index]),
            "combination": torch.tensor(self.combination_labels[sample_index]),
        }
        return image, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-combination", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shape_color_multiview_dataset.npz"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("shape_color_multiview_preview.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        samples_per_combination=args.samples_per_combination,
        image_size=args.image_size,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    dataset = generate_dataset(config)
    validate_dataset(dataset, config)
    np.savez_compressed(args.output, **dataset)
    save_preview(dataset, args.preview)
    print(f"saved dataset: {args.output}")
    print(f"saved preview: {args.preview}")
    print(f"images={dataset['images'].shape}, dtype={dataset['images'].dtype}")
    print(
        f"train={len(dataset['train_indices'])}, "
        f"test={len(dataset['test_indices'])}"
    )


if __name__ == "__main__":
    main()
