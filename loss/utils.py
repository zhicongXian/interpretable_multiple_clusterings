import math
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence
import itertools


def beta_entropy(beta: Tensor, eps: float = 1e-8) -> Tensor:
    return -(beta * (beta + eps).log()).sum(dim=0).mean()


def beta_mass_balance_loss(beta: Tensor) -> Tensor:
    """Give every view comparable total latent capacity."""

    normalized_mass = beta.sum(dim=1) / beta.shape[1]
    target = torch.full_like(normalized_mass, 1.0 / beta.shape[0])
    return (normalized_mass - target).square().mean()

## --TODO why do I need the intrinsic dimension here?
def beta_effective_dimension_loss(
    beta: Tensor,
    minimum_dimensions: float,
    eps: float = 1e-8,
) -> Tensor:
    effective_dimensions = beta.sum(dim=1).square() / (
        beta.square().sum(dim=1) + eps
    )
    return F.relu(minimum_dimensions - effective_dimensions).square().mean()

def normalized_linear_hsic(left: Tensor, right: Tensor, eps: float = 1e-8) -> Tensor:
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross = left.transpose(0, 1) @ right
    numerator = cross.square().sum()
    left_scale = (left.transpose(0, 1) @ left).square().sum().sqrt()
    right_scale = (right.transpose(0, 1) @ right).square().sum().sqrt()
    return numerator / (left_scale * right_scale + eps)


def semi_orthogonal_overlap_loss(projection_weights: Sequence[Tensor]) -> Tensor:
    overlaps = [
        (left @ right.transpose(0, 1)).square().mean()
        for left, right in itertools.combinations(projection_weights, 2)
    ]
    return (
        torch.stack(overlaps).mean()
        if overlaps
        else projection_weights[0].new_zeros(())
    )

def smooth_worst_view(values: Sequence[Tensor], temperature: float = 0.1) -> Tensor:
    """A normalized smooth maximum that cannot hide one poor view in a mean."""

    stacked = torch.stack(tuple(values))
    if len(values) == 1 or temperature <= 0:
        return stacked.max()
    return temperature * (
        torch.logsumexp(stacked / temperature, dim=0)
        - math.log(len(values))
    )