r"""Generic differentiable k-means multiple-view clustering for images.

This program contains no semantic view-specific preprocessing.  Every image is
processed by one shared ResNet autoencoder.  An orthogonal latent rotation,
learned beta masks, and semi-orthogonal projection heads discover an arbitrary
user-defined number of non-redundant views.  Each projected view is passed
through a parameter-free differentiable k-means (DKM) layer following Cho et
al., ICLR 2022 (https://arxiv.org/abs/2108.12659).  DKM alternates soft
distance-based assignments with attention-weighted centroid updates.  Final
labels are the hard argmax of the learned DKM assignments; no self-expression
matrix or spectral clustering post-processing is used.

Required NPZ arrays::

    images          [N,C,H,W] or [N,H,W,C]
    train_indices   [N_train]
    test_indices    [N_test]

Any one-dimensional ``*_labels`` arrays are optional and used only for final
ACC/NMI/ARI evaluation.  They are never supplied to the optimizer.

Example::

    python generic_self_expressive_multiview_clustering_senet_style.py \
        --dataset dataset.npz \
        --clusters 3,3,4 \
        --view-names view_1,view_2,view_3 \
        --pretrain-epochs 15 --view-epochs 15 --joint-epochs 30 \
        --no-show
"""

from __future__ import annotations

import argparse
import itertools
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


class DifferentiableKMeans(nn.Module):
    """Parameter-free, unrolled differentiable k-means (DKM) layer.

    Given samples ``Z`` and centroids ``C``, one iteration computes

    ``A = softmax(-||z_i - c_j||^2 / temperature)``

    followed by ``C_j = sum_i A_ij z_i / sum_i A_ij``.  The loop is unrolled,
    so gradients propagate through assignments and centroid updates.  As in
    Cho et al. (ICLR 2022), the first call initializes centroids from randomly
    selected inputs and subsequent batches start from the last cached
    centroids.  The cache is a buffer, not a learnable parameter.
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int,
        *,
        temperature: float = 1.0,
        max_iterations: int = 5,
        tolerance: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("DKM input_dim must be positive.")
        if n_clusters < 2:
            raise ValueError("DKM n_clusters must be at least two.")
        if temperature <= 0:
            raise ValueError("DKM temperature must be positive.")
        if max_iterations < 1:
            raise ValueError("DKM max_iterations must be positive.")
        if tolerance < 0:
            raise ValueError("DKM tolerance must be nonnegative.")
        if eps <= 0:
            raise ValueError("DKM eps must be positive.")

        self.input_dim = int(input_dim)
        self.n_clusters = int(n_clusters)
        self.temperature = float(temperature)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.eps = float(eps)
        self.register_buffer(
            "centroids", torch.zeros(self.n_clusters, self.input_dim)
        ) # here register the variables
        self.register_buffer("centroids_initialized", torch.tensor(False))

    @torch.no_grad()
    def reset_centroids(self) -> None:
        self.centroids.zero_()
        self.centroids_initialized.fill_(False)

    @torch.no_grad()
    def _initialize_centroids(self, latent: Tensor) -> Tensor:
        sample_count = latent.shape[0]
        if sample_count < self.n_clusters:
            raise ValueError(
                f"DKM needs at least {self.n_clusters} samples, got "
                f"{sample_count}. Increase the batch size or reduce clusters."
            )
        indices = torch.randperm(sample_count, device=latent.device)[
            : self.n_clusters
        ]
        return latent[indices].detach().clone()

    def _assign(self, latent: Tensor, centroids: Tensor) -> tuple[Tensor, Tensor]:
        # Squared Euclidean distance is the k-means metric and corresponds to
        # the Gaussian/EM interpretation in Appendix G of the DKM paper.
        distances_squared = torch.cdist(latent, centroids, p=2).square()
        assignments = F.softmax(-distances_squared / self.temperature, dim=1)
        return assignments, distances_squared

    def forward(
        self,
        latent: Tensor,
        *,
        update_state: bool = True,
    ) -> dict[str, Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.input_dim:
            raise ValueError(
                f"DKM expected [N, {self.input_dim}], got {tuple(latent.shape)}."
            )
        if latent.shape[0] < self.n_clusters and not bool(
            self.centroids_initialized
        ):
            raise ValueError(
                f"DKM needs at least {self.n_clusters} samples for first-call "
                f"initialization, got {latent.shape[0]}."
            )

        if bool(self.centroids_initialized):
            centroids = self.centroids.to(dtype=latent.dtype).clone()
        else:
            centroids = self._initialize_centroids(latent)

        iterations = 0
        centroid_shift = latent.new_zeros(())
        # A final short minibatch can still be assigned using the cached
        # centroids, but it should not redefine K centroids from fewer than K
        # observations.
        if latent.shape[0] >= self.n_clusters:
            centroid_shift = latent.new_tensor(float("inf"))
            for iteration in range(self.max_iterations):
                assignments, _ = self._assign(latent, centroids)
                cluster_mass = assignments.sum(dim=0)
                candidates = (
                    assignments.transpose(0, 1) @ latent
                ) / cluster_mass.clamp_min(self.eps).unsqueeze(1)
                nonempty = cluster_mass > self.eps
                candidates = torch.where(
                    nonempty.unsqueeze(1), candidates, centroids
                )
                centroid_shift = (candidates - centroids).abs().amax()
                centroids = candidates
                iterations = iteration + 1
                if float(centroid_shift.detach()) <= self.tolerance:
                    break

        assignments, distances_squared = self._assign(latent, centroids)
        quantized_latent = assignments @ centroids
        expected_distortion = (
            assignments * distances_squared
        ).sum(dim=1).mean()

        if update_state and latent.shape[0] >= self.n_clusters:
            with torch.no_grad():
                self.centroids.copy_(centroids.detach().to(self.centroids.dtype))
                self.centroids_initialized.fill_(True)

        return {
            "assignments": assignments,
            "centroids": centroids,
            "quantized_latent": quantized_latent,
            "distances_squared": distances_squared,
            "expected_distortion": expected_distortion,
            "hard_labels": assignments.argmax(dim=1),
            "iterations": latent.new_tensor(iterations),
            "centroid_shift": centroid_shift,
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

    def forward(self, inputs: Tensor, beta: Tensor) -> Tensor:
        weighted = inputs * beta.clamp_min(1e-8).sqrt().unsqueeze(0)
        if self.exactly_orthogonal:
            return F.linear(weighted, self.orthonormal_weight())
        else:
            return F.linear(weighted, self.weight_raw)

class GenericDifferentiableKMeansMultiView(nn.Module):
    """Shared nonlinear representation with one DKM layer per latent view."""

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
        dkm_temperature: float = 1.0,
        dkm_max_iterations: int = 5,
        dkm_tolerance: float = 1e-4,
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
        self.dkm_heads = nn.ModuleList(
            [
                DifferentiableKMeans(
                    self.projection_dim,
                    cluster_count,
                    temperature=dkm_temperature,
                    max_iterations=dkm_max_iterations,
                    tolerance=dkm_tolerance,
                )
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

    def forward(
        self,
        images: Tensor,
        *,
        update_dkm_state: bool = True,
    ) -> dict[str, Any]:
        shared, rotated, rotation = self.encode_rotated(images)
        reconstruction = self.decoder(shared)
        beta = self.beta()
        projected_views = self.project_views(rotated, beta)
        dkm = [
            head(projected, update_state=update_dkm_state)
            for head, projected in zip(self.dkm_heads, projected_views)
        ]
        return {
            "shared": shared,
            "rotated": rotated,
            "rotation": rotation,
            "reconstruction": reconstruction,
            "beta": beta,
            "projected_views": projected_views,
            "projection_weights": [
                head.orthonormal_weight() for head in self.projection_heads
            ],
            "dkm": dkm,
            "assignments": [item["assignments"] for item in dkm],
            "centroids": [item["centroids"] for item in dkm],
            "quantized_views": [item["quantized_latent"] for item in dkm],
        }

@dataclass # (frozen=True)
class LossWeightsForAugmentation:
    # Reconstruction is already optimized during pretraining. A smaller joint
    # weight lets clustering reorganize the shared representation.
    reconstruction: float = 1.0
    kmeans: float = 0.2
    stability: float = 0.05
    augmentation_consistency: float = 0.2 # 0.1#0.05
    independence: float = 0.2 #0.02 # 0.02
    projection_orthogonality: float = 0.01
    projection_overlap: float = 0.005 # 0.05
    beta_entropy: float = 0.01
    beta_mass_balance: float = 0.2
    beta_effective_dimension: float = 0.0 #0.05
    latent_variance: float = 0.05
    worst_view_temperature: float = 0.1
    embedding_diversity: float = 0.0
@dataclass # (frozen=True)
class LossWeights:
    reconstruction: float = 1.0
    kmeans: float = 0.2
    stability: float = 0.05
    independence: float = 0.05
    projection_overlap: float = 0.02
    beta_entropy: float = 0.01
    beta_mass_balance: float = 0.0 #0.2
    beta_effective_dimension: float = 0.0 #0.05
    latent_variance: float = 0.05
    worst_view_temperature: float = 0.1
    embedding_diversity: float = 0.0


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

def _pairwise_soft_nmi(values: Sequence[Tensor]) -> Tensor:
    """Dependence between soft DKM assignments from different views."""

    if len(values) < 2:
        return values[0].new_zeros(())
    return torch.stack(
        [
            soft_nmi(left, right, from_logits=False)
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
    kmeans = smooth_worst_view(
        [item["expected_distortion"] for item in outputs["dkm"]],
        weights.worst_view_temperature,
    )
    stability = (
        smooth_worst_view(
            [
                F.mse_loss(left, right)
                for left, right in zip(
                    augmented[0]["assignments"], augmented[1]["assignments"]
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
        + _pairwise_soft_nmi(outputs["assignments"])
    )
    terms = {
        "reconstruction": F.mse_loss(outputs["reconstruction"], images),
        "kmeans": kmeans,
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
        # Kept as an explicit zero-valued diagnostic to guarantee that this
        # objective is disabled in every training phase.
        "embedding_diversity": images.new_zeros(()),
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
    model: GenericDifferentiableKMeansMultiView,
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


def _configure_phase(
    model: GenericDifferentiableKMeansMultiView, phase: str
) -> None:
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
    # DKM heads contain only cached centroid buffers and no parameters.

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
    model: GenericDifferentiableKMeansMultiView,
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
    left_assignments: list[Tensor] = []
    right_assignments: list[Tensor] = []
    view_count = len(roles)
    for view, head in enumerate(model.dkm_heads):
        left_start = view * batch_size
        right_start = (view_count + view) * batch_size
        left = all_projected[view][left_start : left_start + batch_size]
        right = all_projected[view][right_start : right_start + batch_size]
        left_projected.append(left)
        right_projected.append(right)
        left_assignments.append(head(left, update_state=False)["assignments"])
        right_assignments.append(head(right, update_state=False)["assignments"])

    return (
        {
            "projected_views": left_projected,
            "assignments": left_assignments,
        },
        {
            "projected_views": right_projected,
            "assignments": right_assignments,
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

def multiview_loss_with_augmentation(
    images: Tensor,
    outputs: dict[str, Any],
    *,
    weights: LossWeightsForAugmentation,
    n_clusters: Sequence[int],
    minimum_effective_dimensions: float,
    augmented: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if len(n_clusters) != len(outputs["assignments"]):
        raise ValueError("Provide one cluster count for every projected view.")
    kmeans = smooth_worst_view(
        [item["expected_distortion"] for item in outputs["dkm"]],
        weights.worst_view_temperature,
    )
    stability = (
        smooth_worst_view(
            [
                F.mse_loss(left, right)
                for left, right in zip(
                    augmented[0]["assignments"], augmented[1]["assignments"]
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
        + _pairwise_soft_nmi(outputs["assignments"])
    )
    terms = {
        "reconstruction": F.mse_loss(outputs["reconstruction"], images),
        "kmeans": kmeans,
        "stability": stability,
        "augmentation_consistency": augmentation_consistency,
        "independence": independence,
        "projection_orthogonality": semi_orthogonality_loss(
            outputs["projection_weights"]
        ),
        "projection_overlap": semi_orthogonal_overlap_loss(
            outputs["projection_weights"]
        ),
        "beta_entropy": beta_entropy(outputs["beta"]),
        "beta_mass_balance": beta_mass_balance_loss(outputs["beta"]),
        "beta_effective_dimension": beta_effective_dimension_loss(
            outputs["beta"], minimum_effective_dimensions
        ),
        "latent_variance": latent_variance,
        "embedding_diversity": images.new_zeros(()),
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms

def train_phase(
    model: GenericDifferentiableKMeansMultiView,
    loader: DataLoader,
    *,
    phase: str,
    epochs: int,
    learning_rate: float,
    noise_std: float,
    minimum_effective_dimensions: float,
    device: torch.device,
    weights: LossWeights,
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

    weights = LossWeightsForAugmentation()
    early_stopper = EarlyStopper(patience=70, min_delta=0.005, decimals=10)
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {"total": 0.0}
        count = 0
        for images, _ in loader:
            images = images.to(device)
            left = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
            right = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if phase == "view" or phase == "joint":
                if augmentation_roles is None:
                    left = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
                    right = (images + noise_std * torch.randn_like(images)).clamp(0, 1)
                    augmented = (
                        model(left, update_dkm_state=False),
                        model(right, update_dkm_state=False),
                    )
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
                    augmented=(
                        model(left, update_dkm_state=False),
                        model(right, update_dkm_state=False),
                    ),
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
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
                dkm_head = model.dkm_heads[view]
                writer.add_scalar(
                    f"{phase}/dkm_centroid_norm/{name}",
                    float(dkm_head.centroids.norm().detach()),
                    epoch,
                )
                writer.add_scalar(
                    f"{phase}/dkm_temperature/{name}",
                    dkm_head.temperature,
                    epoch,
                )
    _configure_phase(model, "joint")
    return history


@torch.no_grad()
def collect_views_synthetic_shape_color(
    model: GenericDifferentiableKMeansMultiView,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Tensor], np.ndarray]:
    was_training = model.training
    model.eval()
    batches: list[list[Tensor]] = [[] for _ in range(model.n_views)]
    index_batches: list[np.ndarray] = []
    for images, original_indices in loader:
        _, rotated, _ = model.encode_rotated(images.to(device))
        projected_views = model.project_views(rotated)
        for view, latent in enumerate(projected_views):
            batches[view].append(latent.detach().cpu())
        index_batches.append(np.asarray(original_indices, dtype=np.int64))
    model.train(was_training)
    return [torch.cat(items) for items in batches], np.concatenate(index_batches)

@torch.no_grad()
def collect_views_general(
    model: GenericDifferentiableKMeansMultiView,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Tensor], np.ndarray]:
    was_training = model.training
    model.eval()
    batches: list[list[Tensor]] = [[] for _ in range(model.n_views)]
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        _, rotated, _ = model.encode_rotated(images.to(device))
        projected_views = model.project_views(rotated)
        for view, latent in enumerate(projected_views):
            batches[view].append(latent.detach().cpu())
        label_batches.append(np.asarray(labels, dtype=np.int64))
    model.train(was_training)
    return [torch.cat(items) for items in batches], np.concatenate(label_batches)


@torch.no_grad()
def dkm_predictions(
    head: DifferentiableKMeans,
    latent: Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Run DKM on a complete projected view and return snapped assignments."""

    if latent.ndim != 2 or latent.shape[0] < head.n_clusters:
        raise ValueError(
            "Full-view DKM needs a matrix with at least one sample per cluster."
        )
    result = head(latent.float().to(device), update_state=False)
    return result["hard_labels"].detach().cpu().numpy()


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
    model: GenericDifferentiableKMeansMultiView,
    loader: DataLoader,
    dataset_path: str | Path,
    device: torch.device,
    dataset: str,
) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    if dataset.lower() == "synthetic_shape_color":
        latents, original_indices = collect_views_synthetic_shape_color(model, loader, device)
    else:
        latents, labels = collect_views_general(model, loader, device)
    predictions: list[np.ndarray] = []
    for view, latent in enumerate(latents):
        predictions.append(
            dkm_predictions(
                model.dkm_heads[view],
                latent,
                device=device,
            )
        )
    if dataset.lower() == "synthetic_shape_color":
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
    elif dataset.lower() == "nr_objects":
        facets=["color", "material", "shape"] # material_objects_colors, i need to get the label information
    elif dataset.lower() == "stickfigures":
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
        print(f"\nTest differentiable k-means {metric_name}{suffix}:")
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
    model: GenericDifferentiableKMeansMultiView,
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
    model: GenericDifferentiableKMeansMultiView,
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
        "--dkm-temperature",
        type=float,
        default=1.0,
        help="Temperature for DKM soft distance-based assignments.",
    )
    parser.add_argument(
        "--dkm-max-iterations",
        type=int,
        default=5,
        help="Maximum unrolled centroid updates (the paper uses five).",
    )
    parser.add_argument(
        "--dkm-tolerance",
        type=float,
        default=1e-4,
        help="Stop DKM when the maximum centroid change is below this value.",
    )
    parser.add_argument("--minimum-effective-dimensions", type=float, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--view-epochs", type=int, default=300)
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-4)
    parser.add_argument("--view-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
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
        "--visualization", type=Path, default=Path("generic_multiview_dkm_views.html")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("generic_multiview_dkm_model.pt")
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
    if args.view_names is not None and len(args.view_names) != len(args.clusters):
        raise ValueError("view-names and clusters must have the same length.")
    if args.batch_size < max(args.clusters):
        raise ValueError("batch-size must be at least the largest cluster count.")
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

    model = GenericDifferentiableKMeansMultiView(
        sample_images.shape[1:],
        args.clusters,
        view_names=args.view_names,
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_channels=args.decoder_channels,
        dkm_temperature=args.dkm_temperature,
        dkm_max_iterations=args.dkm_max_iterations,
        dkm_tolerance=args.dkm_tolerance,
    ).to(device)
    minimum_effective_dimensions = (
        float(args.minimum_effective_dimensions)
        if args.minimum_effective_dimensions is not None
        else max(2.0, model.latent_dim / (2.0 * model.n_views))
    )
    weights = LossWeights()
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
        NpzImageDataset(args.dataset_path, "train"), args.max_train_samples, args.seed
    )
    test_data = _limited(
        NpzImageDataset(args.dataset_path, "test"), args.max_test_samples, args.seed + 1
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    sample_images, _ = next(iter(train_loader))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = GenericDifferentiableKMeansMultiView(
        sample_images.shape[1:],
        args.clusters,
        view_names=args.view_names,
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        decoder_channels=args.decoder_channels,
        dkm_temperature=args.dkm_temperature,
        dkm_max_iterations=args.dkm_max_iterations,
        dkm_tolerance=args.dkm_tolerance,
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
