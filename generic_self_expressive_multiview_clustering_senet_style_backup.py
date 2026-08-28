# In this v1 version, I will add the diversity regularization loss and minimizing normalized eigenmap

r"""Generic deep self-expressive multiple-view clustering for images.

This program contains no semantic view-specific preprocessing.  Every image is
processed by one shared ResNet autoencoder.  An orthogonal latent rotation,
learned beta masks, and semi-orthogonal projection heads discover an arbitrary
user-defined number of non-redundant views.  Each projected view is trained
with its own SENet-style neural coefficient generator.  The generated
self-expression matrices are sparse, signed, and zero diagonal.  Final labels
are obtained by spectral clustering of the learned full-view affinities.

Required NPZ arrays::

    images          [N,C,H,W] or [N,H,W,C]
    train_indices   [N_train]
    test_indices    [N_test]

Any one-dimensional ``*_labels`` arrays are optional and used only for final
ACC/NMI/ARI evaluation.  They are never supplied to the optimizer.

Example::

    python generic_self_expressive_multiview_clustering.py \
        --dataset dataset.npz \
        --clusters 3,3,4 \
        --view-names view_1,view_2,view_3 \
        --pretrain-epochs 15 --view-epochs 15 --joint-epochs 30 \
        --no-show
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
# from ksg_nmi_torch import *
from data.npzdataset import NpzImageDataset
import random
from models.resnet import ResNetDecoder, ResNetEncoder
from loss.utils import (beta_effective_dimension_loss, beta_mass_balance_loss,
                        beta_entropy, normalized_linear_hsic,
                        smooth_worst_view, semi_orthogonal_overlap_loss)
from data.load_nr_objects import load_nr_objects
from data.load_stickfigures import load_stickfigures
from utils.general_utils import set_seed

def _integer_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def _string_tuple(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("View names must be nonempty and unique.")
    return values


def _limited(dataset: Dataset[Any], maximum: int | None, seed: int) -> Dataset[Any]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices) # Subset of a dataset at specified indices.


class SENetSelfExpression(nn.Module):
    """Generate sparse signed self-expression coefficients from embeddings.

    For samples ``z_i`` and ``z_j``, this module implements

    ``C_ij = alpha * T_b(cos(q(z_i), k(z_j)) / temperature)``,

    where ``q`` and ``k`` are learned MLPs and ``T_b`` is signed soft
    thresholding.  The diagonal is always zero.  Optionally retaining the
    largest absolute ``n_neighbors`` scores makes the batch graph sparse
    without imposing the convex-combination constraint of a row-wise softmax.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        coefficient_dim: int | None = None,
        temperature: float = 1.0,
        n_neighbors: int = 16,
        initial_threshold: float = 0.1,
        initial_coefficient_scale: float = 0.1,
        elastic_net_l1_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("self-expression input_dim must be positive.")
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError("hidden_dim must be positive when provided.")
        if coefficient_dim is not None and coefficient_dim < 1:
            raise ValueError("coefficient_dim must be positive when provided.")
        if temperature <= 0:
            raise ValueError("self-expression temperature must be positive.")
        if n_neighbors < 1:
            raise ValueError("self-expression n_neighbors must be positive.")
        if initial_threshold <= 0:
            raise ValueError("initial_threshold must be positive.")
        if initial_coefficient_scale <= 0:
            raise ValueError("initial_coefficient_scale must be positive.")
        if not 0.0 <= elastic_net_l1_ratio <= 1.0:
            raise ValueError("elastic_net_l1_ratio must lie in [0, 1].")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or max(32, input_dim))
        self.coefficient_dim = int(coefficient_dim or input_dim)
        self.temperature = float(temperature)
        self.n_neighbors = int(n_neighbors)
        self.elastic_net_l1_ratio = float(elastic_net_l1_ratio)

        def make_pair_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.coefficient_dim),
            )

        self.query_network = make_pair_network()
        self.key_network = make_pair_network()
        # Identical initialization starts from a learned-metric similarity
        # graph; q and k are then free to specialize independently.
        self.key_network.load_state_dict(self.query_network.state_dict())

        self.raw_threshold = nn.Parameter(
            torch.tensor(self._inverse_softplus(initial_threshold))
        )
        self.raw_coefficient_scale = nn.Parameter(
            torch.tensor(self._inverse_softplus(initial_coefficient_scale))
        )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    @property
    def threshold(self) -> Tensor:
        # Cosine scores lie in [-1 / temperature, 1 / temperature].  A larger
        # threshold is equivalent to this upper bound but can create Inf * 0
        # later if both learned positive scalars diverge.
        return F.softplus(self.raw_threshold).clamp(
            min=1e-8,
            max=1.0 / self.temperature,
        )

    @property
    def coefficient_scale(self) -> Tensor:
        # Keep generated coefficients in a numerically useful range.  This is
        # deliberately well above the default 0.1 and does not impose a convex
        # combination constraint.
        return F.softplus(self.raw_coefficient_scale).clamp(
            min=1e-8,
            max=10.0,
        )

    def pair_embeddings(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized learned query and key embeddings."""

        query = F.normalize(self.query_network(latent), dim=1, eps=1e-8)
        key = F.normalize(self.key_network(latent), dim=1, eps=1e-8)
        return query, key

    def coefficient_matrix(
        self,
        query: Tensor,
        key: Tensor,
        *,
        n_neighbors: int | None = None,
    ) -> Tensor:
        """Construct ``C`` from precomputed query and key embeddings."""

        if query.ndim != 2 or key.ndim != 2 or query.shape != key.shape:
            raise ValueError("query and key must be equally shaped matrices.")
        sample_count = query.shape[0]
        if sample_count < 1:
            raise ValueError("Self-expression received an empty batch.")
        if sample_count == 1:
            return query.new_zeros((1, 1))

        scores = (query @ key.transpose(0, 1)) / self.temperature
        diagonal = torch.eye(
            sample_count, dtype=torch.bool, device=scores.device
        )
        scores = scores.masked_fill(diagonal, 0.0)

        requested_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        if requested_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        neighbor_count = min(int(requested_neighbors), sample_count - 1)
        support = ~diagonal
        if neighbor_count < sample_count - 1:
            ranking_scores = scores.abs().masked_fill(diagonal, float("-inf"))
            indices = torch.topk(
                ranking_scores, k=neighbor_count, dim=1
            ).indices
            support = torch.zeros_like(diagonal)
            support.scatter_(1, indices, True)

        threshold = self.threshold.to(device=scores.device, dtype=scores.dtype)
        scale = self.coefficient_scale.to(
            device=scores.device, dtype=scores.dtype
        )
        coefficients = scale * scores.sign() * F.relu(scores.abs() - threshold)
        coefficients = coefficients.masked_fill(~support, 0.0)
        coefficients = coefficients.masked_fill(diagonal, 0.0)
        return coefficients

    @staticmethod
    def affinity_from_coefficients(coefficients: Tensor) -> Tensor:
        """Create the nonnegative symmetric affinity used by spectral clustering."""

        affinity = 0.5 * (
            coefficients.abs() + coefficients.transpose(0, 1).abs()
        )
        diagonal = torch.eye(
            coefficients.shape[0],
            dtype=torch.bool,
            device=coefficients.device,
        )
        return affinity.masked_fill(diagonal, 0.0)

    def elastic_net_regularization(self, coefficients: Tensor) -> Tensor:
        """Batch-size-stable elastic-net penalty for generated coefficients."""

        if coefficients.shape[0] <= 1:
            return coefficients.new_zeros(())
        neighbor_count = min(self.n_neighbors, coefficients.shape[1] - 1)
        normalizer = float(max(neighbor_count, 1))
        l1 = coefficients.abs().sum(dim=1).mean() / normalizer
        l2 = coefficients.square().sum(dim=1).mean() / normalizer
        ratio = self.elastic_net_l1_ratio
        return ratio * l1 + 0.5 * (1.0 - ratio) * l2

    def forward(
        self,
        latent: Tensor,
        *,
        n_neighbors: int | None = None,
    ) -> dict[str, Tensor]:
        query, key = self.pair_embeddings(latent)
        coefficients = self.coefficient_matrix(
            query, key, n_neighbors=n_neighbors
        )
        return {
            "coefficients": coefficients,
            "affinity": self.affinity_from_coefficients(coefficients),
            "reconstructed_latent": coefficients @ latent,
            "elastic_net": self.elastic_net_regularization(coefficients),
            "query": query,
            "key": key,
        }

class SemiOrthogonalProjectionHead(nn.Module):
    """A row-orthonormal Stiefel projection without cross-view hard exclusion.
    --TODO learn the projection
    """

    def __init__(self, input_dim: int, projection_dim: int, exactly_orthogonal: bool = False) -> None:
        super().__init__()
        if not 1 <= projection_dim <= input_dim:
            raise ValueError("projection_dim must lie between one and input_dim.")
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.weight_raw = nn.Parameter(torch.empty(projection_dim, input_dim))
        nn.init.orthogonal_(self.weight_raw)
        self.exactly_orthogonal = exactly_orthogonal

    def orthonormal_weight(self) -> Tensor:
        # QR on P^T creates orthonormal columns, hence row-orthonormal P.
        basis, triangular = torch.linalg.qr(self.weight_raw.transpose(0, 1), mode="reduced")
        signs = torch.sign(torch.diagonal(triangular)).detach() #
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return (basis * signs.unsqueeze(0)).transpose(0, 1)

    def effective_weight(self) -> Tensor:
        """Return the same projection matrix that is used in ``forward``."""

        return self.orthonormal_weight() if self.exactly_orthogonal else self.weight_raw

    def forward(self, inputs: Tensor, beta: Tensor) -> Tensor:
        weighted = inputs * beta.clamp_min(1e-8).sqrt().unsqueeze(0)
        return F.linear(weighted, self.effective_weight())


class SoftClusterAssignmentHead(nn.Module):
    """Map a projected view to differentiable cluster probabilities."""

    def __init__(self, input_dim: int, cluster_count: int) -> None:
        super().__init__()
        if input_dim < 1 or cluster_count < 2:
            raise ValueError("The assignment head needs positive dimensions and K >= 2.")
        self.normalization = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, cluster_count)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, projected: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
        logits = self.linear(self.normalization(projected))
        probabilities = torch.softmax(logits / temperature, dim=1)
        return logits, probabilities


class GenericSelfExpressiveMultiView(nn.Module):
    """Shared nonlinear representation with learned generic latent views."""

    def __init__(
        self,
        image_shape: Sequence[int],
        n_clusters: Sequence[int],
        *,
        view_names: Sequence[str] | None = None,
        latent_dim: int = 64,
        projection_dim: int | None = None,
        encoder_channels: Sequence[int] = (16, 32, 64),
        encoder_blocks: Sequence[int] = (1, 1, 1),
        decoder_channels: Sequence[int] = (64, 32, 16, 8),
        self_expression_temperature: float = 1.0,
        self_expression_neighbors: int = 16,
        self_expression_hidden_dim: int | None = None,
        self_expression_coefficient_dim: int | None = None,
        self_expression_threshold: float = 0.1,
        self_expression_coefficient_scale: float = 0.1,
        elastic_net_l1_ratio: float = 0.9,
        cluster_assignment_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(value) for value in image_shape)
        self.n_clusters = tuple(int(value) for value in n_clusters)
        self.n_views = len(self.n_clusters)
        if self.n_views < 2:
            raise ValueError("Generic multiple-view clustering requires >= 2 views.")
        self.view_names = tuple(
            view_names or [f"view_{index + 1}" for index in range(self.n_views)]
        )
        if len(self.view_names) != self.n_views or len(set(self.view_names)) != self.n_views:
            raise ValueError("Provide one unique name for every view.")
        self.latent_dim = int(latent_dim)
        self.projection_dim = int(
            projection_dim
            if projection_dim is not None
            else max(2, latent_dim // self.n_views)
        )
        if self.projection_dim > self.latent_dim:
            raise ValueError("projection_dim cannot exceed latent_dim.")
        if cluster_assignment_temperature <= 0.0:
            raise ValueError("cluster_assignment_temperature must be positive.")
        self.cluster_assignment_temperature = float(cluster_assignment_temperature)

        self.encoder = ResNetEncoder(
            self.image_shape,
            encoder_channels,
            encoder_blocks,
            self.latent_dim,
            dropout=0.1,
        )
        self.decoder = ResNetDecoder(
            self.latent_dim,
            self.image_shape,
            decoder_channels,
            dropout=0.1,
        )
        self.rotation_raw = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        nn.init.orthogonal_(self.rotation_raw)

        beta_logits = torch.full((self.n_views, self.latent_dim), -1.0)
        for dimension in range(self.latent_dim):
            beta_logits[dimension % self.n_views, dimension] = 1.0
        self.beta_logits = nn.Parameter(beta_logits)
        self.projection_heads = nn.ModuleList(
            [
                SemiOrthogonalProjectionHead(self.latent_dim, self.projection_dim)
                for _ in range(self.n_views)
            ]
        )
        self.self_expression_heads = nn.ModuleList(
            [
                SENetSelfExpression(
                    self.projection_dim,
                    hidden_dim=self_expression_hidden_dim,
                    coefficient_dim=self_expression_coefficient_dim,
                    temperature=self_expression_temperature,
                    n_neighbors=self_expression_neighbors,
                    initial_threshold=self_expression_threshold,
                    initial_coefficient_scale=self_expression_coefficient_scale,
                    elastic_net_l1_ratio=elastic_net_l1_ratio,
                )
                for _ in range(self.n_views)
            ]
        )
        self.cluster_assignment_heads = nn.ModuleList(
            [
                SoftClusterAssignmentHead(self.projection_dim, cluster_count)
                for cluster_count in self.n_clusters
            ]
        )

    def rotation(self) -> Tensor:
        basis, triangular = torch.linalg.qr(self.rotation_raw)
        signs = torch.sign(torch.diagonal(triangular)).detach()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return basis * signs.unsqueeze(0) # this only creates rotation basis

    def beta(self) -> Tensor:
        return torch.softmax(self.beta_logits, dim=0)

    def encode_rotated(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shared = self.encoder(images)
        rotation = self.rotation()
        return shared, shared @ rotation, rotation

    def project_views(
        self,
        rotated: Tensor,
        beta: Tensor | None = None,
    ) -> list[Tensor]:
        beta = self.beta() if beta is None else beta
        return [
            head(rotated, beta[view])
            for view, head in enumerate(self.projection_heads)
        ]

    def autoencode(self, images: Tensor) -> Tensor:
        return self.decoder(self.encoder(images))

    def forward(self, images: Tensor) -> dict[str, Any]:
        shared, rotated, rotation = self.encode_rotated(images)
        reconstruction = self.decoder(shared)
        beta = self.beta()
        projected_views = self.project_views(rotated, beta)
        self_expression = [
            head(projected)
            for head, projected in zip(self.self_expression_heads, projected_views)
        ]
        cluster_assignments = [
            head(projected, self.cluster_assignment_temperature)
            for head, projected in zip(
                self.cluster_assignment_heads, projected_views
            )
        ]
        return {
            "shared": shared,
            "rotated": rotated,
            "rotation": rotation,
            "reconstruction": reconstruction,
            "beta": beta,
            "projected_views": projected_views,
            "projection_weights": [
                head.effective_weight() for head in self.projection_heads
            ],
            "self_expression": self_expression,
            "coefficients": [item["coefficients"] for item in self_expression],
            "affinities": [item["affinity"] for item in self_expression],
            "cluster_logits": [item[0] for item in cluster_assignments],
            "soft_assignments": [item[1] for item in cluster_assignments],
        }

@dataclass # (frozen=True)
class LossWeightsForAugmentation:
    # Reconstruction is already optimized during pretraining. A smaller joint
    # weight lets clustering reorganize the shared representation.
    reconstruction: float = 1.0
    self_expression: float = 0.2
    coefficient_regularization: float = 0.02
    stability: float = 0.05
    augmentation_consistency: float = 0.2 # 0.1#0.05
    independence: float = 0.2 #0.02 # 0.02
    projection_orthogonality: float = 0.001#0.01
    projection_overlap: float = 0.005 # 0.05
    beta_entropy: float = 0.01
    beta_mass_balance: float = 0.2
    beta_effective_dimension: float = 0.0 #0.05
    latent_variance: float = 0.05
    worst_view_temperature: float = 0.1
    embedding_diversity: float = 0.0 # 0.00002
    normalized_cut: float = 0.0 # 0.1
    cluster_assignment_orthogonality: float = 0.05
@dataclass # (frozen=True)
class LossWeights:
    reconstruction: float = 1.0
    self_expression: float = 0.2
    coefficient_regularization: float = 0.02
    stability: float = 0.05
    independence: float = 0.05
    projection_overlap: float = 0.02
    beta_entropy: float = 0.01
    beta_mass_balance: float = 0.0 #0.2
    beta_effective_dimension: float = 0.0 #0.05
    latent_variance: float = 0.05
    worst_view_temperature: float = 0.1
    embedding_diversity: float = 0.0 # 0.00002


def _pairwise_hsic(values: Sequence[Tensor]) -> Tensor:
    if len(values) < 2:
        return values[0].new_zeros(())
    return torch.stack(
        [
            normalized_linear_hsic(left, right)
            for left, right in itertools.combinations(values, 2)
        ]
    ).mean()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0, decimals=6):
        self.patience = patience
        self.min_delta = min_delta
        self.decimals = decimals
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        validation_loss = round(validation_loss, self.decimals)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss) and abs(validation_loss - self.min_validation_loss) / (self.min_validation_loss + 1e-16) < self.min_delta:  #:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        return False

class TotalCodingRate(nn.Module):
    """ from https://github.com/ryanchankh/mcr2
    """

    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W1, W2):
        """Discriminative Loss."""
        p, m = W1.shape  # [d, B]
        I = torch.eye(p, device=W1.device)
        scalar = p / (m * self.eps)
        sign, logdet = torch.linalg.slogdet(I + scalar * W1.matmul(W2.T))  # torch.logdet(I + scalar * W.matmul(W.T))
        if sign <= 0:
            print("Matrix not positive definite!")
        return logdet / 2.

    def forward(self, Z1, Z2):
        """

        :param Z1: [B,d]
        :param Z2: [B,d]
        :return:
        """

        return - self.compute_discrimn_loss(Z1.T, Z2.T)
# nmi = ksg_normalized_mutual_information(
#     x,
#     y,
#     k=5,
#     chunk_size=1024,
#     dimension_adjusted=False,
# )
def _maximize_latent_diversity(values: Tensor, eps = 1.0) -> Tensor:
    total_coding_rate = TotalCodingRate(eps=eps)
    return total_coding_rate(values, values)

def _pairwise_hsic_projected_view(values: Sequence[Tensor], eps = 1.0 ) -> Tensor:
    if len(values) < 2:
        return values[0].new_zeros(())
    total_coding_rate = TotalCodingRate(eps=eps)
    return torch.stack(
        [
            soft_nmi(left, right) # total_coding_rate(left, right)
            for left, right in itertools.combinations(values, 2)
        ]
    ).mean()

def soft_nmi(
    zi: torch.Tensor,
    zj: torch.Tensor,
    from_logits: bool = True,
    eps: float = 1e-8,
):
    """
    Differentiable normalized mutual information.

    Parameters
    ----------
    zi : Tensor [batch_size, num_clusters_i]
        Logits or soft cluster assignments for view i.
    zj : Tensor [batch_size, num_clusters_j]
        Logits or soft cluster assignments for view j.
    from_logits : bool
        If True, apply softmax to zi and zj.

    Returns
    -------
    nmi : scalar Tensor
    mi  : scalar Tensor
    hi  : entropy of Zi
    hj  : entropy of Zj
    """

    if zi.ndim != 2 or zj.ndim != 2:
        raise ValueError("zi and zj must have shape [batch_size, num_clusters].")

    if zi.shape[0] != zj.shape[0]:
        raise ValueError("zi and zj must have the same batch size.")

    if from_logits:
        pi_given_x = F.softmax(zi, dim=1)
        pj_given_x = F.softmax(zj, dim=1)
    else:
        pi_given_x = zi / zi.sum(dim=1, keepdim=True).clamp_min(eps)
        pj_given_x = zj / zj.sum(dim=1, keepdim=True).clamp_min(eps)

    # Estimated joint distribution:
    # P(a, b) = (1/N) sum_n P(a|x_n) P(b|x_n)
    joint = pi_given_x.T @ pj_given_x
    joint = joint / joint.sum().clamp_min(eps)

    # Marginal distributions
    marginal_i = joint.sum(dim=1)
    marginal_j = joint.sum(dim=0)

    # I(Zi; Zj) = sum_ab p(a,b) log[p(a,b)/(p(a)p(b))]
    independent_joint = marginal_i[:, None] * marginal_j[None, :]

    mi = (
        joint
        * (
            torch.log(joint.clamp_min(eps))
            - torch.log(independent_joint.clamp_min(eps))
        )
    ).sum()

    # I(Zi; Zi) = H(Zi), and similarly for Zj
    entropy_i = -(marginal_i * torch.log(marginal_i.clamp_min(eps))).sum()
    entropy_j = -(marginal_j * torch.log(marginal_j.clamp_min(eps))).sum()

    nmi = mi / torch.sqrt(
        (entropy_i * entropy_j).clamp_min(eps)
    )

    return nmi# , mi, entropy_i, entropy_j


def multiview_loss(
    images: Tensor,
    outputs: dict[str, Any],
    *,
    weights: LossWeights,
    minimum_effective_dimensions: float,
    augmented: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    self_expression = smooth_worst_view(
        [
            F.mse_loss(item["reconstructed_latent"], latent)
            for item, latent in zip(
                outputs["self_expression"], outputs["projected_views"]
            )
        ],
        weights.worst_view_temperature,
    )
    coefficient_regularization = smooth_worst_view(
        [item["elastic_net"] for item in outputs["self_expression"]],
        weights.worst_view_temperature,
    )
    stability = (
        smooth_worst_view(
            [
                F.mse_loss(left, right)
                for left, right in zip(
                    augmented[0]["coefficients"], augmented[1]["coefficients"]
                )
            ],
            weights.worst_view_temperature,
        )
        if augmented is not None
        else images.new_zeros(())
    )
    latent_variance = smooth_worst_view(
        [
            F.relu(
                0.5 - latent.var(dim=0, unbiased=False).add(1e-4).sqrt() ##--TODO: is this necessary? latent_var <= 0.5
            ).mean()
            for latent in outputs["projected_views"]
        ],
        weights.worst_view_temperature,
    )
    independence = 0.5 * (
        _pairwise_hsic(outputs["projected_views"])
        + _pairwise_hsic(outputs["affinities"])#_pairwise_hsic(outputs["affinities"])
    )
    maximize_latent_diversity = _maximize_latent_diversity(outputs["shared"])
    terms = {
        "reconstruction": F.mse_loss(outputs["reconstruction"], images),
        "self_expression": self_expression,
        "coefficient_regularization": coefficient_regularization,
        "stability": stability,
        "independence": independence,
        "projection_overlap": semi_orthogonal_overlap_loss(
            outputs["projection_weights"]
        ),
        "beta_entropy": beta_entropy(outputs["beta"]),
        "beta_mass_balance": beta_mass_balance_loss(outputs["beta"]),
        "beta_effective_dimension": beta_effective_dimension_loss(
            outputs["beta"], minimum_effective_dimensions
        ),
        "latent_variance": latent_variance,
        "embedding_diversity":  maximize_latent_diversity
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms


def _make_writer(path: Path | None) -> Any | None:
    if path is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError("Install tensorboard to enable logging.") from error
    return SummaryWriter(log_dir=str(path))


def pretrain_autoencoder(
    model: GenericSelfExpressiveMultiView,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    noise_std: float,
    device: torch.device,
    writer: Any | None,
    loss_weight = 0.0 #0.002
) -> list[float]:
    parameters = itertools.chain(model.encoder.parameters(), model.decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    total_coding_rate = TotalCodingRate(eps=1.0)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        reconstruction_loss = 0.0
        mcr_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            corrupted = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            recon_loss = F.mse_loss(model.autoencode(corrupted), images)
            latent_image = model.encoder(images)
            latent_corrupted= model.encoder(corrupted)
            mcr_loss_tensor = 0.5 * (total_coding_rate(latent_image, latent_image) + total_coding_rate(latent_corrupted,
                                                                                                   latent_corrupted))
            loss = recon_loss +  loss_weight * mcr_loss_tensor
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * images.shape[0]
            reconstruction_loss += float(recon_loss.detach()) * images.shape[0]
            mcr_loss += float(mcr_loss_tensor.detach())
            count += images.shape[0]
        value = running / max(count, 1)
        history.append(value)
        print(f"pretrain epoch {epoch:03d}/{epochs:03d} total={value:.6f},"
              f" reconstruction_loss={reconstruction_loss/max(count, 1)}, "
              f"mcr_loss={mcr_loss}")
        if writer is not None:
            writer.add_scalar("pretrain/reconstruction", value, epoch)
            writer.add_scalar("pretrain/mcr", mcr_loss, epoch)
    return history


def _configure_phase(model: GenericSelfExpressiveMultiView, phase: str) -> None:
    if phase not in {"view", "joint"}:
        raise ValueError(f"Unknown phase: {phase}")
    train_shared = phase == "joint"
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(train_shared)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(train_shared)
    model.rotation_raw.requires_grad_(True)
    model.beta_logits.requires_grad_(True)
    for head in model.projection_heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    for head in model.self_expression_heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    for head in model.cluster_assignment_heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)

######## 1. for view training add the constrastive loss ########

def _mild_spatial_augmentation(images: Tensor) -> Tensor:
    """Random flips and small translations that preserve shape and color."""

    augmented = images.clone()
    batch_size, _, height, width = augmented.shape
    horizontal = torch.rand(batch_size, device=images.device) < 0.5
    vertical = torch.rand(batch_size, device=images.device) < 0.2
    augmented[horizontal] = augmented[horizontal].flip(-1)
    augmented[vertical] = augmented[vertical].flip(-2)
    maximum_shift = max(1, min(height, width) // 16)
    shifts = torch.randint(
        -maximum_shift,
        maximum_shift + 1,
        (batch_size, 2),
        device=images.device,
    )
    return torch.stack(
        [
            torch.roll(
                image,
                shifts=(int(shift[0]), int(shift[1])),
                dims=(-2, -1),
            )
            for image, shift in zip(augmented, shifts)
        ]
    )


def _randomize_color_preserve_shape(images: Tensor, strength: float) -> Tensor:
    """Change foreground color while retaining its spatial silhouette."""

    if images.shape[1] == 1:
        scale = 0.4 + 1.2 * torch.rand(
            images.shape[0], 1, 1, 1, device=images.device, dtype=images.dtype
        )
        transformed = images * scale
    else:
        # Synthetic objects normally use a dark background. Channel maximum
        # retains the foreground mask, while the random RGB vector changes hue.
        intensity = images.amax(dim=1, keepdim=True)
        random_color = 0.15 + 0.85 * torch.rand(
            images.shape[0],
            images.shape[1],
            1,
            1,
            device=images.device,
            dtype=images.dtype,
        )
        transformed = intensity * random_color
    return ((1.0 - strength) * images + strength * transformed).clamp(0.0, 1.0)


def _shuffle_space_preserve_color(images: Tensor, strength: float) -> Tensor:
    """Destroy global shape while preserving each sample's color histogram."""

    batch_size, channels, height, width = images.shape
    flattened = images.flatten(start_dim=2)
    permutations = torch.stack(
        [torch.randperm(height * width, device=images.device) for _ in range(batch_size)]
    )
    gather_indices = permutations[:, None, :].expand(-1, channels, -1)
    shuffled = flattened.gather(2, gather_indices).view_as(images)
    return (1.0 - strength) * images + strength * shuffled


def _mask_vertical_region(
    images: Tensor,
    region: str,
    *,
    split_fraction: float,
    strength: float,
) -> Tensor:
    """Mask the upper or lower image region with the zero/background value.

    ``split_fraction`` is the boundary measured from the top of the image.
    ``strength=1`` completely removes the selected region, while smaller
    values blend the original pixels with the masked image.
    """

    if region not in {"upper", "lower"}:
        raise ValueError("region must be either 'upper' or 'lower'.")
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("split_fraction must lie strictly between 0 and 1.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("mask strength must lie in [0, 1].")
    if images.ndim != 4 or images.shape[-2] < 2:
        raise ValueError("images must have shape [N, C, H, W] with H >= 2.")

    height = images.shape[-2]
    split_row = min(max(int(round(height * split_fraction)), 1), height - 1)
    keep_mask = images.new_ones((1, 1, height, 1))
    if region == "upper":
        keep_mask[..., :split_row, :] = 0.0
    else:
        keep_mask[..., split_row:, :] = 0.0
    masked = images * keep_mask
    return (1.0 - strength) * images + strength * masked

def augment_for_view(
    images: Tensor,
    role: str,
    *,
    noise_std: float,
    shape_recolor_strength: float,
    color_shuffle_strength: float,
    upper_lower_mask_split: float = 0.5,
    upper_lower_mask_strength: float = 1.0,
) -> Tensor:
    """Create an augmentation that removes nuisance information for one view.

    The role names describe the content that should be preserved. Therefore,
    ``upper`` masks the lower region and ``lower`` masks the upper region.
    """


    if role == "shape": # --TODO is the way to do data augmentation correct?
        augmented = _mild_spatial_augmentation(images)
        augmented = _randomize_color_preserve_shape(
            augmented, shape_recolor_strength
        )
    elif role == "color":
        augmented = _mild_spatial_augmentation(images)
        augmented = _shuffle_space_preserve_color(
            augmented, color_shuffle_strength
        )
    elif role == "upper":
        augmented = images
        augmented = _mask_vertical_region(
            augmented,
            "lower",
            split_fraction=upper_lower_mask_split,
            strength=upper_lower_mask_strength,
        )
    elif role == "lower":
        augmented = images
        augmented = _mask_vertical_region(
            augmented,
            "upper",
            split_fraction=upper_lower_mask_split,
            strength=upper_lower_mask_strength,
        )
    elif role != "generic":
        raise ValueError(f"Unknown augmentation role: {role}")
    if noise_std > 0:
        augmented = augmented + noise_std * torch.randn_like(augmented)
    return augmented.clamp(0.0, 1.0)


def view_specific_augmented_outputs(
    model: GenericSelfExpressiveMultiView,
    images: Tensor,
    roles: Sequence[str],
    *,
    noise_std: float,
    shape_recolor_strength: float,
    color_shuffle_strength: float,
    upper_lower_mask_split: float = 0.5,
    upper_lower_mask_strength: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Efficiently extract matching view outputs from semantic augmentations."""

    if len(roles) != model.n_views:
        raise ValueError("Provide exactly one augmentation role for every view.")
    batch_size = images.shape[0]
    augmented_batches = [
        augment_for_view(
            images,
            role,
            noise_std=noise_std,
            shape_recolor_strength=shape_recolor_strength,
            color_shuffle_strength=color_shuffle_strength,
            upper_lower_mask_split=upper_lower_mask_split,
            upper_lower_mask_strength=upper_lower_mask_strength,
        )
        for _ in range(2)
        for role in roles
    ]
    stacked = torch.cat(augmented_batches, dim=0)
    _, rotated, _ = model.encode_rotated(stacked)
    all_projected = model.project_views(rotated)

    left_projected: list[Tensor] = []
    right_projected: list[Tensor] = []
    left_coefficients: list[Tensor] = []
    right_coefficients: list[Tensor] = []
    view_count = len(roles)
    for view, head in enumerate(model.self_expression_heads):
        left_start = view * batch_size
        right_start = (view_count + view) * batch_size
        left = all_projected[view][left_start : left_start + batch_size]
        right = all_projected[view][right_start : right_start + batch_size]
        left_projected.append(left)
        right_projected.append(right)
        left_coefficients.append(head(left)["coefficients"])
        right_coefficients.append(head(right)["coefficients"])

    return (
        {
            "projected_views": left_projected,
            "coefficients": left_coefficients,
        },
        {
            "projected_views": right_projected,
            "coefficients": right_coefficients,
        },
    )

def semi_orthogonality_loss(projection_weights: Sequence[Tensor]) -> Tensor:
    """Softly enforce row orthogonality: ``W W^T = I`` for every view."""

    penalties: list[Tensor] = []
    for weight in projection_weights:
        identity = torch.eye(
            weight.shape[0], device=weight.device, dtype=weight.dtype
        )
        gram_error = weight @ weight.transpose(0, 1) - identity
        penalties.append(gram_error.square().mean())
    return torch.stack(penalties).mean()


def differentiable_normalized_cut_losses(
    affinities: Sequence[Tensor],
    soft_assignments: Sequence[Tensor],
    n_clusters: Sequence[int],
    *,
    worst_view_temperature: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return normalized MinCut and assignment anti-collapse losses.

    For each view with affinity ``A``, degree matrix ``D``, and soft cluster
    assignments ``P``, the MinCut relaxation is

    ``-tr(P^T A P) / tr(P^T D P)``.

    The accompanying Frobenius penalty pushes ``P^T P`` toward a scaled
    identity.  It discourages both the single-cluster solution and identical
    uniform assignments while encouraging approximately balanced clusters.
    Neither term requires an eigendecomposition.
    """

    if not (
        len(affinities) == len(soft_assignments) == len(n_clusters)
    ):
        raise ValueError(
            "Provide one affinity, assignment matrix, and cluster count per view."
        )

    cut_per_view: list[Tensor] = []
    orthogonality_per_view: list[Tensor] = []
    for view_index, (affinity, assignment, cluster_count) in enumerate(zip(
        affinities, soft_assignments, n_clusters
    )):
        sample_count = affinity.shape[0]
        if affinity.ndim != 2 or affinity.shape != (sample_count, sample_count):
            raise ValueError("Every affinity must be a square matrix.")
        if assignment.shape != (sample_count, cluster_count):
            raise ValueError(
                "Every soft assignment must have shape [batch, cluster_count]."
            )
        affinity_is_finite = torch.isfinite(affinity)
        assignment_is_finite = torch.isfinite(assignment)
        if not bool(affinity_is_finite.all()) or not bool(
            assignment_is_finite.all()
        ):
            diagnostics: list[str] = []
            for name, value, finite_mask in (
                ("affinity", affinity, affinity_is_finite),
                ("soft_assignment", assignment, assignment_is_finite),
            ):
                if bool(finite_mask.all()):
                    continue
                nan_count = int(torch.isnan(value).sum())
                positive_inf_count = int((torch.isinf(value) & (value > 0)).sum())
                negative_inf_count = int((torch.isinf(value) & (value < 0)).sum())
                finite_values = value[finite_mask]
                finite_abs_max = (
                    float(finite_values.abs().amax())
                    if finite_values.numel() > 0
                    else float("nan")
                )
                diagnostics.append(
                    f"{name}: nan={nan_count}, +inf={positive_inf_count}, "
                    f"-inf={negative_inf_count}, finite_abs_max={finite_abs_max:.4g}"
                )
            raise FloatingPointError(
                f"Normalized MinCut view {view_index} received non-finite values; "
                + "; ".join(diagnostics)
            )

        affinity = 0.5 * (affinity + affinity.transpose(0, 1))
        affinity = affinity.clamp_min(0.0)
        # MinCut is invariant to multiplying A by a positive scalar.  Scaling
        # by a detached maximum prevents overflow in A @ P while preserving
        # gradients with respect to the graph structure.
        affinity_scale_floor = torch.finfo(affinity.dtype).tiny
        affinity = affinity / affinity.amax().detach().clamp_min(
            affinity_scale_floor
        )
        assignment = assignment.clamp_min(0.0)
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(eps)
        degree = affinity.sum(dim=1)

        # tr(P^T A P), evaluated without constructing the K x K product.
        within_cluster_affinity = (assignment * (affinity @ assignment)).sum()
        # tr(P^T D P), with D represented by its degree vector.
        cluster_volume = (degree[:, None] * assignment.square()).sum()
        cut_per_view.append(
            -within_cluster_affinity / cluster_volume.clamp_min(eps)
        )

        assignment_gram = assignment.transpose(0, 1) @ assignment
        normalized_gram = assignment_gram / torch.linalg.vector_norm(
            assignment_gram
        ).clamp_min(eps)
        target = torch.eye(
            cluster_count,
            device=assignment.device,
            dtype=assignment.dtype,
        ) / math.sqrt(float(cluster_count))
        orthogonality_per_view.append(
            torch.linalg.vector_norm(normalized_gram - target)
        )

    return (
        smooth_worst_view(cut_per_view, worst_view_temperature),
        smooth_worst_view(orthogonality_per_view, worst_view_temperature),
    )

def multiview_loss_with_augmentation(
    images: Tensor,
    outputs: dict[str, Any],
    *,
    weights: LossWeightsForAugmentation,
    n_clusters: Sequence[int],
    minimum_effective_dimensions: float,
    augmented: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if len(n_clusters) != len(outputs["affinities"]):
        raise ValueError("Provide one cluster count for every projected view.")
    self_expression = smooth_worst_view(
        [
            F.mse_loss(item["reconstructed_latent"], latent)
            for item, latent in zip(
                outputs["self_expression"], outputs["projected_views"]
            )
        ],
        weights.worst_view_temperature,
    )
    coefficient_regularization = smooth_worst_view(
        [item["elastic_net"] for item in outputs["self_expression"]],
        weights.worst_view_temperature,
    )
    stability = (
        smooth_worst_view(
            [
                F.mse_loss(left, right)
                for left, right in zip(
                    augmented[0]["coefficients"], augmented[1]["coefficients"]
                )
            ],
            weights.worst_view_temperature,
        )
        if augmented is not None
        else images.new_zeros(())
    )
    augmentation_consistency = (
        smooth_worst_view(
            [
                1.0 - F.cosine_similarity(left, right, dim=1).mean()
                for left, right in zip(
                    augmented[0]["projected_views"],
                    augmented[1]["projected_views"],
                )
            ],
            weights.worst_view_temperature,
        )
        if augmented is not None
        else images.new_zeros(())
    )
    latent_variance = smooth_worst_view(
        [
            F.relu(
                0.5 - latent.var(dim=0, unbiased=False).add(1e-4).sqrt()
            ).mean()
            for latent in outputs["projected_views"]
        ],
        weights.worst_view_temperature,
    )
    independence = 0.5 * (
        _pairwise_hsic(outputs["projected_views"])
        + _pairwise_hsic(outputs["affinities"])
    )

    maximize_latent_diversity = _maximize_latent_diversity(outputs["shared"])
    normalized_cut, cluster_assignment_orthogonality = (
        differentiable_normalized_cut_losses(
            outputs["affinities"],
            outputs["soft_assignments"],
            n_clusters,
            worst_view_temperature=weights.worst_view_temperature,
        )
    )
    terms = {
        "reconstruction": F.mse_loss(outputs["reconstruction"], images),
        "self_expression": self_expression,
        "coefficient_regularization": coefficient_regularization,
        "stability": stability,
        "augmentation_consistency": augmentation_consistency,
        "independence": independence,
        "projection_orthogonality": semi_orthogonality_loss(
            outputs["projection_weights"]
        ),
        "projection_overlap": semi_orthogonal_overlap_loss(
            outputs["projection_weights"]
        ), # Not so sure whether this is necessary
        "normalized_cut": normalized_cut,
        "cluster_assignment_orthogonality": cluster_assignment_orthogonality,
        "beta_entropy": beta_entropy(outputs["beta"]),
        "beta_mass_balance": beta_mass_balance_loss(outputs["beta"]),
        "beta_effective_dimension": beta_effective_dimension_loss(
            outputs["beta"], minimum_effective_dimensions
        ),
        "latent_variance": latent_variance,
        "embedding_diversity": maximize_latent_diversity
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms

def train_phase(
    model: GenericSelfExpressiveMultiView,
    loader: DataLoader,
    *,
    phase: str,
    epochs: int,
    learning_rate: float,
    noise_std: float,
    minimum_effective_dimensions: float,
    device: torch.device,
    weights: LossWeightsForAugmentation,
    writer: Any | None,
    augmentation_roles=None,
    shape_recolor_strength = 0.6,
    color_shuffle_strength = 0.6,
    upper_lower_mask_split: float = 0.5,
    upper_lower_mask_strength: float = 1.0,
) -> list[dict[str, float]]:
    _configure_phase(model, phase)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []

    early_stopper = EarlyStopper(patience=70, min_delta=0.005, decimals=10)
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {"total": 0.0}
        count = 0
        # This term stays disabled in every phase. View independence is already
        # enforced explicitly; dispersing the shared embedding is not needed.
        if phase=="view":
            weights.embedding_diversity = 0.01
        else:
            weights.embedding_diversity = 0.0

        for images, _ in loader:
            images = images.to(device)
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError(
                    "The input batch contains NaN or Inf values before the model."
                )
            left = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
            right = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if phase == "view" or phase == "joint":
                if augmentation_roles is None:
                    left = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
                    right = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
                    augmented = (model(left), model(right))
                else:
                    augmented = view_specific_augmented_outputs(
                        model,
                        images,
                        augmentation_roles,
                        noise_std=noise_std,
                        shape_recolor_strength=shape_recolor_strength,
                        color_shuffle_strength=color_shuffle_strength,
                        upper_lower_mask_split=upper_lower_mask_split,
                        upper_lower_mask_strength=upper_lower_mask_strength,
                    )

                total, terms = multiview_loss_with_augmentation(
                    images,
                    outputs,
                    weights=weights,
                    n_clusters=model.n_clusters,
                    minimum_effective_dimensions=minimum_effective_dimensions,
                    augmented=augmented,
                )
            else:
                total, terms = multiview_loss(
                    images,
                    outputs,
                    weights=weights,
                    minimum_effective_dimensions=minimum_effective_dimensions,
                    augmented=(model(left), model(right)),
                )
            if not bool(torch.isfinite(total)):
                nonfinite_terms = [
                    name for name, value in terms.items()
                    if not bool(torch.isfinite(value))
                ]
                raise FloatingPointError(
                    "The total loss became non-finite. Non-finite terms: "
                    + ", ".join(nonfinite_terms)
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            batch_size = images.shape[0]
            sums["total"] += float(total.detach()) * batch_size
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
            count += batch_size
        metrics = {name: value / max(count, 1) for name, value in sums.items()}
        history.append(metrics)
        details = " ".join(f"{name}={value:.5f}" for name, value in metrics.items())
        print(f"{phase} epoch {epoch:03d}/{epochs:03d} {details}")
        if early_stopper.early_stop(metrics['total']):
            break
        if writer is not None:
            for name, value in metrics.items():
                writer.add_scalar(f"{phase}/{name}", value, epoch)
            beta = model.beta().detach()
            for view, name in enumerate(model.view_names):
                writer.add_scalar(
                    f"{phase}/beta_mass/{name}",
                    float(beta[view].mean()),
                    epoch,
                )
                self_expression_head = model.self_expression_heads[view]
                writer.add_scalar(
                    f"{phase}/self_expression_threshold/{name}",
                    float(self_expression_head.threshold.detach()),
                    epoch,
                )
                writer.add_scalar(
                    f"{phase}/self_expression_scale/{name}",
                    float(self_expression_head.coefficient_scale.detach()),
                    epoch,
                )
                assignment = outputs["soft_assignments"][view].detach()
                marginal = assignment.mean(dim=0)
                entropy_scale = math.log(float(model.n_clusters[view]))
                sample_entropy = -(
                    assignment * assignment.clamp_min(1e-8).log()
                ).sum(dim=1).mean() / entropy_scale
                balance_entropy = -(
                    marginal * marginal.clamp_min(1e-8).log()
                ).sum() / entropy_scale
                writer.add_scalar(
                    f"{phase}/cluster_assignment_entropy/{name}",
                    float(sample_entropy),
                    epoch,
                )
                writer.add_scalar(
                    f"{phase}/cluster_balance_entropy/{name}",
                    float(balance_entropy),
                    epoch,
                )
    _configure_phase(model, "joint")
    return history


@torch.no_grad()
def collect_views_synthetic_shape_color(
    model: GenericSelfExpressiveMultiView,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Tensor], np.ndarray]:
    was_training = model.training
    model.eval()
    batches: list[list[Tensor]] = [[] for _ in range(model.n_views)]
    index_batches: list[np.ndarray] = []
    for images, original_indices in loader:
        outputs = model(images.to(device))
        for view, latent in enumerate(outputs["projected_views"]):
            batches[view].append(latent.detach().cpu())
        index_batches.append(np.asarray(original_indices, dtype=np.int64))
    model.train(was_training)
    return [torch.cat(items) for items in batches], np.concatenate(index_batches)

@torch.no_grad()
def collect_views_general(
    model: GenericSelfExpressiveMultiView,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Tensor], np.ndarray]:
    was_training = model.training
    model.eval()
    batches: list[list[Tensor]] = [[] for _ in range(model.n_views)]
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        outputs = model(images.to(device))
        for view, latent in enumerate(outputs["projected_views"]):
            batches[view].append(latent.detach().cpu())
        label_batches.append(np.asarray(labels, dtype=np.int64))
    model.train(was_training)
    return [torch.cat(items) for items in batches], np.concatenate(label_batches)


@torch.no_grad()
def self_expression_affinity(
    head: SENetSelfExpression,
    latent: Tensor,
    *,
    n_neighbors: int,
    device: torch.device,
    transform_batch_size: int = 4096,
) -> np.ndarray:
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("Full-view self-expression requires at least two samples.")
    if n_neighbors < 1 or transform_batch_size < 1:
        raise ValueError("n_neighbors and transform_batch_size must be positive.")

    # Only the small neural transformations are evaluated on the model device.
    # The O(N^2) coefficient/affinity matrices stay on CPU, as in the previous
    # full-dataset evaluation path, to avoid unnecessary GPU memory pressure.
    query_batches: list[Tensor] = []
    key_batches: list[Tensor] = []
    for batch in latent.float().split(transform_batch_size):
        query, key = head.pair_embeddings(batch.to(device))
        query_batches.append(query.detach().cpu())
        key_batches.append(key.detach().cpu())
    query = torch.cat(query_batches, dim=0)
    key = torch.cat(key_batches, dim=0)
    coefficients = head.coefficient_matrix(
        query, key, n_neighbors=n_neighbors
    )
    affinity = head.affinity_from_coefficients(coefficients)
    return affinity.numpy()


def _clustering_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    _, encoded_labels = np.unique(labels, return_inverse=True)
    _, encoded_predictions = np.unique(predictions, return_inverse=True)
    contingency = np.zeros(
        (encoded_predictions.max() + 1, encoded_labels.max() + 1), dtype=np.int64
    )
    np.add.at(contingency, (encoded_predictions, encoded_labels), 1)
    rows, columns = linear_sum_assignment(contingency, maximize=True)
    return float(contingency[rows, columns].sum() / len(labels))


@torch.no_grad()
def evaluate_clustering(
    model: GenericSelfExpressiveMultiView,
    loader: DataLoader,
    dataset_path: str | Path,
    device: torch.device,
    dataset: str,
    *,
    n_neighbors: int,
    random_state: int,
) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    if dataset.lower() == "synthetic_shape_color":
        latents, original_indices = collect_views_synthetic_shape_color(model, loader, device)
    else:
        latents, labels = collect_views_general(model, loader, device)
    predictions: list[np.ndarray] = []
    for view, latent in enumerate(latents):
        affinity = self_expression_affinity(
            model.self_expression_heads[view],
            latent,
            n_neighbors=n_neighbors,
            device=device,
        )
        predictions.append(
            SpectralClustering(
                n_clusters=model.n_clusters[view],
                affinity="precomputed",
                assign_labels="kmeans",
                n_init=20,
                random_state=random_state,
            ).fit_predict(affinity)
        )
    if args.dataset == "synthetic_shape_color":
        archive = np.load(dataset_path, allow_pickle=False)
        label_keys = sorted(
            key
            for key in archive.files
            if key.endswith("_labels")
            and key != "combination_labels"
            and archive[key].ndim == 1
            and len(archive[key]) > int(original_indices.max(initial=-1))
        )
        if not label_keys:
            print("\nNo *_labels arrays found; returning predictions without evaluation.")
            return {
                "facets": [],
                "predictions": [prediction.tolist() for prediction in predictions],
                "matrices": {"acc": [], "nmi": [], "ari": []},
                "matches": [],
                "mean_metrics": None,
            }
        facets = [key.removesuffix("_labels") for key in label_keys]
        labels = [np.asarray(archive[key])[original_indices] for key in label_keys]
    elif args.dataset.lower() == "nr_objects":
        facets=["color", "material", "shape"] # material_objects_colors, i need to get the label information
    elif args.dataset.lower() == "stickfigures":
        facets=["upper", "lower"]

    predictions=np.asarray(predictions)
    labels=np.asarray(labels)
    if predictions.shape != labels.shape:
        labels = labels.transpose()
    acc = np.asarray(
        [[_clustering_accuracy(y, p) for y in labels] for p in predictions]
    )
    nmi = np.asarray(
        [[normalized_mutual_info_score(y, p) for y in labels] for p in predictions]
    )
    ari = np.asarray(
        [[adjusted_rand_score(y, p) for y in labels] for p in predictions]
    )
    matrices = {"acc": acc, "nmi": nmi, "ari": ari}
    matched_views, matched_facets = linear_sum_assignment(acc, maximize=True)
    matches = [
        {
            "view": model.view_names[int(view)],
            "facet": facets[int(facet)],
            "acc": float(acc[view, facet]),
            "nmi": float(nmi[view, facet]),
            "ari": float(ari[view, facet]),
        }
        for view, facet in zip(matched_views, matched_facets)
    ]
    mean_metrics = {
        metric: float(np.mean([match[metric] for match in matches]))
        for metric in ("acc", "nmi", "ari")
    }
    name_width = max(12, max(len(name) for name in model.view_names) + 2)
    column_width = max(11, max(len(name) for name in facets) + 2)
    for metric_name, matrix in (("ACC", acc), ("NMI", nmi), ("ARI", ari)):
        suffix = " (cluster IDs matched)" if metric_name == "ACC" else ""
        print(f"\nTest spectral self-expression {metric_name}{suffix}:")
        print(
            "discovered".ljust(name_width)
            + "".join(name.rjust(column_width) for name in facets)
        )
        for view, view_name in enumerate(model.view_names):
            print(
                view_name.ljust(name_width)
                + "".join(
                    f"{value:>{column_width}.4f}" for value in matrix[view]
                )
            )
    print("\nOptimal one-to-one correspondence (evaluation only):")
    for match in matches:
        print(
            f"  {match['view']} -> {match['facet']}: "
            f"ACC={match['acc']:.4f}, NMI={match['nmi']:.4f}, "
            f"ARI={match['ari']:.4f}"
        )
    print(
        "  mean matched: "
        f"ACC={mean_metrics['acc']:.4f}, NMI={mean_metrics['nmi']:.4f}, "
        f"ARI={mean_metrics['ari']:.4f}"
    )
    print("  (Labels were not used during training.)")

    output_file = f"./outputs/{args.dataset}_{args.experiment_name}_clustering_results.txt"

    with open(output_file, "a+", encoding="utf-8") as file:
        print("\nOptimal one-to-one correspondence (evaluation only):", file=file)

        for match in matches:
            print(
                f"  {match['view']} -> {match['facet']}: "
                f"ACC={match['acc']:.4f}, "
                f"NMI={match['nmi']:.4f}, "
                f"ARI={match['ari']:.4f}",
                file=file,
            )

        print(
            "  mean matched: "
            f"ACC={mean_metrics['acc']:.4f}, "
            f"NMI={mean_metrics['nmi']:.4f}, "
            f"ARI={mean_metrics['ari']:.4f}",
            file=file,
        )
    return {
        "facets": facets,
        "predictions": [prediction.tolist() for prediction in predictions],
        "matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
        "matches": matches,
        "mean_metrics": mean_metrics,
    }


def _rgb_image(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return np.rint(255 * tensor[:3].permute(1, 2, 0).numpy()).astype(np.uint8)


def view_saliency(
    model: GenericSelfExpressiveMultiView,
    image: Tensor,
    view: int,
) -> Tensor:
    """Input-gradient attribution for one projected view's latent energy."""

    image = image.detach().clone().requires_grad_(True)
    _, rotated, _ = model.encode_rotated(image)
    projected = model.projection_heads[view](rotated, model.beta()[view])
    gradient = torch.autograd.grad(projected.square().mean(), image)[0]
    saliency = gradient.abs().mean(dim=1, keepdim=True)
    scale = saliency.flatten(start_dim=1).amax(dim=1).view(-1, 1, 1, 1)
    return (saliency / scale.clamp_min(1e-8)).clamp(0, 1).detach()


def visualize_views(
    model: GenericSelfExpressiveMultiView,
    image: Tensor,
    *,
    output_file: str | Path,
    show: bool,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    device = next(model.parameters()).device
    image = image[:1].to(device)
    model.eval()
    with torch.no_grad():
        reconstruction = model.autoencode(image)
        beta = model.beta().detach().cpu().numpy()
    saliencies = [view_saliency(model, image, view) for view in range(model.n_views)]
    columns = max(3, model.n_views)
    titles = ["Input", "Shared reconstruction", "Learned beta masks"]
    titles.extend([""] * (columns - 3))
    titles.extend(list(model.view_names))
    titles.extend([""] * (columns - model.n_views))
    figure = make_subplots(rows=2, cols=columns, subplot_titles=titles)
    original = _rgb_image(image)
    figure.add_trace(go.Image(z=original), row=1, col=1)
    figure.add_trace(go.Image(z=_rgb_image(reconstruction)), row=1, col=2)
    figure.add_trace(
        go.Heatmap(
            z=beta,
            x=[f"z{dimension}" for dimension in range(model.latent_dim)],
            y=list(model.view_names),
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar={"title": "beta"},
        ),
        row=1,
        col=3,
    )
    for view, saliency in enumerate(saliencies):
        figure.add_trace(go.Image(z=original, hoverinfo="skip"), row=2, col=view + 1)
        figure.add_trace(
            go.Heatmap(
                z=saliency[0, 0].cpu().numpy(),
                colorscale="Reds",
                zmin=0,
                zmax=1,
                opacity=0.6,
                showscale=False,
            ),
            row=2,
            col=view + 1,
        )
    for column in (1, 2):
        figure.update_xaxes(showticklabels=False, showgrid=False, row=1, col=column)
        figure.update_yaxes(
            showticklabels=False,
            showgrid=False,
            autorange="reversed",
            row=1,
            col=column,
        )
    for column in range(1, model.n_views + 1):
        figure.update_xaxes(showticklabels=False, showgrid=False, row=2, col=column)
        figure.update_yaxes(
            showticklabels=False,
            showgrid=False,
            autorange="reversed",
            row=2,
            col=column,
        )
    figure.update_layout(
        title={"text": "Generic learned views and input saliency", "x": 0.5},
        template="plotly_white",
        width=max(1050, 390 * columns),
        height=760,
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 90, "b": 30},
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() in {".html", ".htm"}:
        figure.write_html(str(output_file), include_plotlyjs=True, full_html=True)
    else:
        try:
            figure.write_image(str(output_file))
        except Exception as error:
            raise RuntimeError("Use .html or install kaleido for static export.") from error
    print(f"saved visualization: {output_file}")
    if show:
        figure.show()
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)#default=Path("shape_color_multiview_dataset.npz"))
    parser.add_argument("--clusters", type=_integer_tuple, required=True)
    parser.add_argument("--view-names", type=_string_tuple, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--encoder-channels", type=_integer_tuple, default=(16, 32, 64))
    parser.add_argument("--encoder-blocks", type=_integer_tuple, default=(1, 1, 1))
    parser.add_argument("--decoder-channels", type=_integer_tuple, default=(64, 32, 16, 8))
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature applied to learned query-key similarities.",
    )
    parser.add_argument("--self-expression-neighbors", type=int, default=16)
    parser.add_argument("--senet-hidden-dim", type=int, default=None)
    parser.add_argument("--senet-coefficient-dim", type=int, default=None)
    parser.add_argument("--senet-threshold", type=float, default=0.1)
    parser.add_argument("--senet-coefficient-scale", type=float, default=0.1)
    parser.add_argument("--elastic-net-l1-ratio", type=float, default=0.9)
    parser.add_argument(
        "--cluster-assignment-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature of the per-view cluster assignment heads.",
    )
    parser.add_argument(
        "--normalized-cut-weight",
        type=float,
        default=0.1,
        help="Weight of the differentiable normalized MinCut loss.",
    )
    parser.add_argument(
        "--cluster-orthogonality-weight",
        type=float,
        default=0.05,
        help="Weight preventing collapsed or uniform cluster assignments.",
    )
    parser.add_argument("--spectral-neighbors", type=int, default=20)
    parser.add_argument("--minimum-effective-dimensions", type=float, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--view-epochs", type=int, default=300)
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-4)
    parser.add_argument("--view-learning-rate", type=float, default=5e-5)
    parser.add_argument("--joint-learning-rate", type=float, default=5e-5)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--upper-lower-mask-split",
        type=float,
        default=0.5,
        help="Vertical upper/lower boundary as a fraction of image height.",
    )
    parser.add_argument(
        "--upper-lower-mask-strength",
        type=float,
        default=1.0,
        help="Strength of upper/lower masking in [0, 1].",
    )
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="v1")
    parser.add_argument(
        "--visualization", type=Path, default=Path("generic_multiview_views_soft_nmi_aug.html")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("generic_multiview_model_aug.pt")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--inference", action="store_true")

    parser.add_argument(
        "--augmentation-roles",
        type=_string_tuple,
        default=None,
        help=(
            "Comma-separated semantic content preserved by each view. "
            "Use shape,color for the shape-color dataset or upper,lower "
            "for stick figures. The upper role masks the lower image region, "
            "and the lower role masks the upper region."
        ),
    )
    return parser.parse_args()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_dataloaders_for_synthetic_shape_color(args):
    train_data = _limited(
        NpzImageDataset(args.dataset_path, "train"), args.max_train_samples, args.seed
    )
    test_data = _limited(
        NpzImageDataset(args.dataset_path, "test"), args.max_test_samples, args.seed + 1
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def run(args: argparse.Namespace) -> None:
    datasets_list = ["synthetic_shape_color", "nr_objects", "stickfigures"]
    assert args.dataset.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))
    same_seeds(args.seed)
    if min(args.pretrain_epochs, args.view_epochs, args.joint_epochs) < 0:
        raise ValueError("Epoch counts must be nonnegative.")
    if args.cluster_assignment_temperature <= 0.0:
        raise ValueError("cluster-assignment-temperature must be positive.")
    if min(args.normalized_cut_weight, args.cluster_orthogonality_weight) < 0.0:
        raise ValueError("Normalized-cut loss weights must be nonnegative.")
    if args.view_names is not None and len(args.view_names) != len(args.clusters):
        raise ValueError("view-names and clusters must have the same length.")
    if not 0.0 < args.upper_lower_mask_split < 1.0:
        raise ValueError("upper-lower-mask-split must lie strictly between 0 and 1.")
    if not 0.0 <= args.upper_lower_mask_strength <= 1.0:
        raise ValueError("upper-lower-mask-strength must lie in [0, 1].")
    if args.dataset.lower() == "stickfigures" and args.augmentation_roles is None:
        if len(args.clusters) != 2:
            raise ValueError(
                "Automatic stick-figure augmentation requires exactly two views. "
                "Otherwise provide --augmentation-roles explicitly."
            )
        args.augmentation_roles = ("upper", "lower")
    if (
        args.augmentation_roles is not None
        and len(args.augmentation_roles) != len(args.clusters)
    ):
        raise ValueError("Provide exactly one augmentation role for every view.")
    set_seed(args.seed)
    if args.dataset.lower() == "synthetic_shape_color":
        train_loader, test_loader = get_dataloaders_for_synthetic_shape_color(args)
    elif args.dataset.lower() == "nr_objects":
        # here comes the nr-objects loading
        train_loader, test_loader = load_nr_objects(args)
    elif args.dataset.lower() == "stickfigures":
        train_loader, test_loader = load_stickfigures(args)
    else:
        raise ValueError("dataset must be either 'synthetic' or 'nr_objects'.")


    sample_images, _ = next(iter(train_loader))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = GenericSelfExpressiveMultiView(
        sample_images.shape[1:],
        args.clusters,
        view_names=args.view_names,
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_channels=args.decoder_channels,
        self_expression_temperature=args.temperature,
        self_expression_neighbors=args.self_expression_neighbors,
        self_expression_hidden_dim=args.senet_hidden_dim,
        self_expression_coefficient_dim=args.senet_coefficient_dim,
        self_expression_threshold=args.senet_threshold,
        self_expression_coefficient_scale=args.senet_coefficient_scale,
        elastic_net_l1_ratio=args.elastic_net_l1_ratio,
        cluster_assignment_temperature=args.cluster_assignment_temperature,
    ).to(device)
    minimum_effective_dimensions = (
        float(args.minimum_effective_dimensions)
        if args.minimum_effective_dimensions is not None
        else max(2.0, model.latent_dim / (2.0 * model.n_views))
    )
    weights = LossWeightsForAugmentation(
        normalized_cut=args.normalized_cut_weight,
        cluster_assignment_orthogonality=args.cluster_orthogonality_weight,
    )
    writer = _make_writer(args.tensorboard_log_dir)
    try:
        pretrain_history = pretrain_autoencoder(
            model,
            train_loader,
            epochs=args.pretrain_epochs,
            learning_rate=args.pretrain_learning_rate,
            noise_std=args.noise_std,
            device=device,
            writer=writer,
        )
        # save latent embeddings after pretraining:
        train_x = []
        train_idx = []
        for images, indices in train_loader:
            images = images.to(device)
            latent_images = model.encoder(images)
            train_x.append(latent_images.detach().cpu().numpy())
            train_idx.append(indices.detach().cpu().numpy())
        train_x = np.concatenate(train_x)
        train_idx = np.concatenate(train_idx)
        np.savez(f"outputs/{args.dataset}_pretrain_embedding_train_data.npz", train_x=train_x, train_idx=train_idx)

        test_x = []
        test_idx = []
        for images, indices in test_loader:
            images = images.to(device)
            latent_images = model.encoder(images)
            test_x.append(latent_images.detach().cpu().numpy())
            test_idx.append(indices.detach().cpu().numpy())
        test_x = np.concatenate(test_x)
        test_idx = np.concatenate(test_idx)
        np.savez(f"outputs/{args.dataset}_pretrain_embedding_test_data.npz", test_x=test_x, test_idx=test_idx)



        print("evaluation after PRETRAINING")
        test_metrics = evaluate_clustering(
            model,
            test_loader,
            args.dataset_path,
            device,
            dataset = args.dataset,
            n_neighbors=args.spectral_neighbors,
            random_state=args.seed,
        )
        view_history = train_phase(
            model,
            train_loader,
            phase="view",
            epochs=args.view_epochs,
            learning_rate=args.view_learning_rate,
            noise_std=args.noise_std,
            minimum_effective_dimensions=minimum_effective_dimensions,
            device=device,
            weights=weights,
            writer=writer,
            augmentation_roles=args.augmentation_roles,
            upper_lower_mask_split=args.upper_lower_mask_split,
            upper_lower_mask_strength=args.upper_lower_mask_strength,
        )
        print("evaluation after VIEW TRAINING")
        test_metrics = evaluate_clustering(
            model,
            test_loader,
            args.dataset_path,
            device,
            dataset = args.dataset,
            n_neighbors=args.spectral_neighbors,
            random_state=args.seed,
        )
        joint_history = train_phase(
            model,
            train_loader,
            phase="joint",
            epochs=args.joint_epochs,
            learning_rate=args.joint_learning_rate,
            noise_std=args.noise_std,
            minimum_effective_dimensions=minimum_effective_dimensions,
            device=device,
            weights=weights,
            writer=writer,
            augmentation_roles=args.augmentation_roles,
            upper_lower_mask_split=args.upper_lower_mask_split,
            upper_lower_mask_strength=args.upper_lower_mask_strength,
        )
        test_metrics = evaluate_clustering(
            model,
            test_loader,
            args.dataset_path,
            device,
            dataset = args.dataset,
            n_neighbors=args.spectral_neighbors,
            random_state=args.seed,
        )
        test_images, _ = next(iter(test_loader))
        if not 0 <= args.sample_index < len(test_images):
            raise IndexError("sample-index is outside the first test batch.")
        visualize_views(
            model,
            test_images[args.sample_index : args.sample_index + 1],
            output_file=args.visualization,
            show=not args.no_show,
        )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "configuration": vars(args),
                "loss_weights": asdict(weights),
                "minimum_effective_dimensions": minimum_effective_dimensions,
                "pretrain_history": pretrain_history,
                "view_history": view_history,
                "joint_history": joint_history,
                "test_metrics": test_metrics,
            },
            args.checkpoint,
        )
        print(f"saved checkpoint: {args.checkpoint}")
    finally:
        if writer is not None:
            writer.close()

def inference(args: argparse.Namespace):
    same_seeds(args.seed)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    train_data = _limited(
        NpzImageDataset(args.dataset, "train"), args.max_train_samples, args.seed
    )
    test_data = _limited(
        NpzImageDataset(args.dataset, "test"), args.max_test_samples, args.seed + 1
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    sample_images, _ = next(iter(train_loader))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = GenericSelfExpressiveMultiView(
        sample_images.shape[1:],
        args.clusters,
        view_names=args.view_names,
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_channels=args.decoder_channels,
        self_expression_temperature=args.temperature,
        self_expression_neighbors=args.self_expression_neighbors,
        self_expression_hidden_dim=args.senet_hidden_dim,
        self_expression_coefficient_dim=args.senet_coefficient_dim,
        self_expression_threshold=args.senet_threshold,
        self_expression_coefficient_scale=args.senet_coefficient_scale,
        elastic_net_l1_ratio=args.elastic_net_l1_ratio,
        cluster_assignment_temperature=args.cluster_assignment_temperature,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    test_metrics = evaluate_clustering(
        model,
        test_loader,
        args.dataset_path,
        device,
        dataset=args.dataset,
        n_neighbors=args.spectral_neighbors,
        random_state=args.seed,
        )
    test_images, _ = next(iter(test_loader))
    if not 0 <= args.sample_index < len(test_images):
        raise IndexError("sample-index is outside the first test batch.")
    random.seed(args.seed)

    # Inclusive: can return 1 through 10
    sample_index = random.randint(0, len(test_loader) - 1)
    visualize_views(
        model,
        test_images[sample_index : sample_index + 1],
        output_file=args.visualization,
        show=not args.no_show,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.inference:
        inference(args)
    else:
        run(args)
