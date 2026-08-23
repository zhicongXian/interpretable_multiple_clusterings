import numpy as np
from torch import Tensor, nn
import itertools
import torch
from utils.visualizers import plotly_visualizer
import random

def make_shape_color_images(
    *,
    samples_per_combination: int = 60,
    image_size: int = 32,
    noise: float = 0.03,
    seed: int = 0,
) -> tuple[Tensor, np.ndarray, np.ndarray]:
    """Generate RGB images with independent shape and color clusterings."""

    if image_size < 12:
        raise ValueError("image_size must be at least 12 pixels.")
    generator = np.random.default_rng(seed)
    coordinates = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(coordinates, coordinates, indexing="ij")
    colors = np.asarray(
        [[0.95, 0.15, 0.15], [0.15, 0.90, 0.20], [0.15, 0.25, 0.95]],
        dtype=np.float32,
    )
    images = []
    shape_labels = []
    color_labels = []

    for shape_index, color_index in itertools.product(range(3), repeat=2):
        for _ in range(samples_per_combination):
            shift_x, shift_y = generator.uniform(-0.12, 0.12, size=2)
            scale = float(generator.uniform(0.85, 1.15))
            x = (grid_x - shift_x) / scale
            y = (grid_y - shift_y) / scale

            if shape_index == 0:
                mask = x**2 + y**2 <= 0.48**2
            elif shape_index == 1:
                mask = (np.abs(x) <= 0.44) & (np.abs(y) <= 0.44)
            else:
                mask = (
                    (y >= -0.50)
                    & (y <= 0.55)
                    & (np.abs(x) <= 0.70 * (0.55 - y))
                )

            image = generator.normal(
                loc=0.04,
                scale=noise,
                size=(3, image_size, image_size),
            ).astype(np.float32)
            image[:, mask] = colors[color_index, :, None]
            images.append(np.clip(image, 0.0, 1.0))
            shape_labels.append(shape_index)
            color_labels.append(color_index)

    permutation = generator.permutation(len(images))
    image_array = np.stack(images)[permutation]
    return (
        torch.as_tensor(image_array, dtype=torch.float32),
        np.asarray(shape_labels)[permutation],
        np.asarray(color_labels)[permutation],
    )

if __name__ == "__main__":
    dataset, shape_labels, color_labels = make_shape_color_images()
    dataset_np = dataset.detach().cpu().numpy()

    # randomly pick one dataset and visualize:
    idx = random.randint(0, len(dataset_np))

    plotly_visualizer(dataset_np[idx])
