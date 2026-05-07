"""Distance functions between output belief distributions.

Unified collection of distance functions for comparing probability
distributions over discrete categories (e.g., weekdays, months, years).

Includes:
  - fisher_rao: Fisher-Rao geodesic distance on the probability simplex
  - hellinger: Hellinger distance on the probability simplex
  - wasserstein1_cyclic: W1 (Earth Mover's) distance with cyclic ground metric
  - wasserstein2_cyclic: W2 distance with cyclic ground metric
  - wasserstein1_noncyclic: W1 distance with linear (non-cyclic) ground metric
  - wasserstein2_noncyclic: W2 distance with linear (non-cyclic) ground metric
  - euclidean_log_prob: Euclidean distance in log-probability space
  - fisher_rao_gaussian: Fisher-Rao distance on the 1D Gaussian manifold
  - pairwise_output_distance: compute full pairwise distance matrix
"""

from __future__ import annotations

import dataclasses
import functools
import math
import random as _random
from typing import Callable

import numpy as np
import ot
import torch
from torch import Tensor

from typing import Any

DistanceFn = Callable[[Tensor, Tensor], Tensor]

# Conformal cost function: (K, D) path on simplex -> (K,) non-negative cost.
# Must be differentiable for L-BFGS.
ConformalCostFn = Callable[[Tensor], Tensor]


# ---------------------------------------------------------------------------
# Cyclic cost matrices (shared by wasserstein and displacement interpolation)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _get_cyclic_costs(n: int) -> tuple[Tensor, Tensor, Tensor]:
    """Compute and cache cyclic cost matrices for an n-category cycle.

    Returns:
        (cost_sq, cost, displacement) tensors of shape (n, n).
        cost_sq[i,j] = min(|i-j|, n-|i-j|)^2
        cost[i,j]    = min(|i-j|, n-|i-j|)
        displacement[i,j] = signed shortest-path displacement from i to j
    """
    cost_sq = torch.zeros(n, n)
    cost = torch.zeros(n, n)
    displacement = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            cost_sq[i, j] = d**2
            cost[i, j] = d
            fwd = (j - i) % n
            displacement[i, j] = fwd if fwd <= n - fwd else fwd - n
    return cost_sq, cost, displacement


# ---------------------------------------------------------------------------
# Cyclic displacement weights (used by wasserstein geodesics)
# ---------------------------------------------------------------------------


def cyclic_displacement_weights(
    positions: Tensor, sigma: float | None, W: int
) -> Tensor:
    """Compute deposit weights for mass at cyclic positions.

    Args:
        positions: (...,) intermediate positions on the cycle [0, W).
        sigma: circular std dev in bin units. None -> linear interp (2 nearest bins).
        W: number of bins on the cycle.

    Returns:
        (..., W) normalized weights for depositing mass into each bin.
    """
    if sigma is None:
        pos = positions.double() % W
        lo = pos.long() % W
        hi = (lo + 1) % W
        frac = pos - pos.floor()
        weights = torch.zeros(*positions.shape, W, dtype=torch.float64)
        weights.scatter_(-1, lo.unsqueeze(-1), (1 - frac).unsqueeze(-1))
        weights.scatter_(-1, hi.unsqueeze(-1), frac.unsqueeze(-1))
        return weights

    # Von Mises kernel: weight[k] ~ exp(kappa * cos(2*pi*(k - center)/W))
    sigma_angular = sigma * (2 * math.pi / W)
    kappa = 1.0 / (sigma_angular**2)

    bins = torch.arange(W, dtype=torch.float64)  # (W,)
    diff = bins - positions.unsqueeze(-1).double()  # (..., W)
    angle = 2 * math.pi * diff / W
    log_weights = kappa * torch.cos(angle)
    log_weights -= log_weights.max(dim=-1, keepdim=True).values
    weights = torch.exp(log_weights)
    return weights / weights.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Distance metrics: simplex-based
# ---------------------------------------------------------------------------


def fisher_rao(p: Tensor, q: Tensor) -> Tensor:
    """Fisher-Rao geodesic distance on the probability simplex.

    d(p, q) = 2 * arccos(sum(sqrt(p_i * q_i))).
    p, q: (..., K) probability simplices.
    """
    bc = (p * q).clamp(min=0).sqrt().sum(-1)
    return 2.0 * torch.acos(bc.clamp(max=1.0))


def hellinger(p: Tensor, q: Tensor) -> Tensor:
    """Hellinger distance on the probability simplex.

    H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2.
    p, q: (..., K) probability simplices.
    """
    return (p.sqrt() - q.sqrt()).pow(2).sum(-1).clamp(min=0.0).sqrt() / (2.0**0.5)


# ---------------------------------------------------------------------------
# Non-cyclic (linear) cost matrices
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _get_noncyclic_costs(n: int) -> tuple[Tensor, Tensor]:
    """Compute and cache linear cost matrices for an n-category ordinal domain.

    Returns:
        (cost_sq, cost) tensors of shape (n, n).
        cost[i,j]    = |i - j|
        cost_sq[i,j] = |i - j|^2
    """
    idx = torch.arange(n)
    cost = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()
    cost_sq = cost**2
    return cost_sq, cost


# ---------------------------------------------------------------------------
# Distance metrics: non-cyclic Wasserstein
# ---------------------------------------------------------------------------


def wasserstein2_noncyclic(p: Tensor, q: Tensor) -> Tensor:
    """W2 distance with linear (non-cyclic) ground metric. Infers size from p.shape[-1].

    Uses exact OT via POT library. Handles (..., W) batched inputs.
    """
    W = p.shape[-1]
    cost_sq, _ = _get_noncyclic_costs(W)
    batch_shape = p.shape[:-1]
    p_flat = p.reshape(-1, W)
    q_flat = q.reshape(-1, W)
    cost_np = cost_sq.numpy()

    results = []
    for i in range(p_flat.shape[0]):
        p_i = p_flat[i].detach().double().numpy()
        q_i = q_flat[i].detach().double().numpy()
        p_i = p_i / p_i.sum()
        q_i = q_i / q_i.sum()
        w2_sq = ot.emd2(p_i, q_i, cost_np)
        results.append(max(w2_sq, 0.0) ** 0.5)

    return torch.tensor(results, dtype=p.dtype).reshape(batch_shape)


def wasserstein1_noncyclic(p: Tensor, q: Tensor) -> Tensor:
    """W1 (Earth Mover's) distance with linear (non-cyclic) ground metric. Infers W from p.shape[-1]."""
    W = p.shape[-1]
    _, cost = _get_noncyclic_costs(W)
    batch_shape = p.shape[:-1]
    p_flat = p.reshape(-1, W)
    q_flat = q.reshape(-1, W)
    cost_np = cost.numpy()

    results = []
    for i in range(p_flat.shape[0]):
        p_i = p_flat[i].detach().double().numpy()
        q_i = q_flat[i].detach().double().numpy()
        p_i = p_i / p_i.sum()
        q_i = q_i / q_i.sum()
        w1 = ot.emd2(p_i, q_i, cost_np)
        results.append(max(w1, 0.0))
    return torch.tensor(results, dtype=p.dtype).reshape(batch_shape)


# ---------------------------------------------------------------------------
# Distance metrics: cyclic Wasserstein
# ---------------------------------------------------------------------------


def wasserstein2_cyclic(p: Tensor, q: Tensor) -> Tensor:
    """W2 distance with cyclic ground metric. Infers cycle length from p.shape[-1].

    Uses exact OT via POT library. Handles (..., W) batched inputs.
    """
    W = p.shape[-1]
    cost_sq, _, _ = _get_cyclic_costs(W)
    batch_shape = p.shape[:-1]
    p_flat = p.reshape(-1, W)
    q_flat = q.reshape(-1, W)
    cost_np = cost_sq.numpy()

    results = []
    for i in range(p_flat.shape[0]):
        p_i = p_flat[i].detach().double().numpy()
        q_i = q_flat[i].detach().double().numpy()
        p_i = p_i / p_i.sum()
        q_i = q_i / q_i.sum()
        w2_sq = ot.emd2(p_i, q_i, cost_np)
        results.append(max(w2_sq, 0.0) ** 0.5)

    return torch.tensor(results, dtype=p.dtype).reshape(batch_shape)


def wasserstein1_cyclic(p: Tensor, q: Tensor) -> Tensor:
    """W1 (Earth Mover's) distance with cyclic ground metric. Infers W from p.shape[-1]."""
    W = p.shape[-1]
    _, cost, _ = _get_cyclic_costs(W)
    batch_shape = p.shape[:-1]
    p_flat = p.reshape(-1, W)
    q_flat = q.reshape(-1, W)
    cost_np = cost.numpy()

    results = []
    for i in range(p_flat.shape[0]):
        p_i = p_flat[i].detach().double().numpy()
        q_i = q_flat[i].detach().double().numpy()
        p_i = p_i / p_i.sum()
        q_i = q_i / q_i.sum()
        w1 = ot.emd2(p_i, q_i, cost_np)
        results.append(max(w1, 0.0))
    return torch.tensor(results, dtype=p.dtype).reshape(batch_shape)


# ---------------------------------------------------------------------------
# Distance metrics: log-prob and Gaussian
# ---------------------------------------------------------------------------


def euclidean_log_prob(
    p: Tensor,
    q: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    """Euclidean distance in log-probability space.

    d(p, q) = || log(p) - log(q) ||_2

    Natural metric if the model operates in log-space internally.
    Works for batches: p, q can be (N, K) and returns (N,).
    """
    return (p.clamp(min=eps).log() - q.clamp(min=eps).log()).norm(dim=-1)


def _fit_gaussian_params(
    distributions: Tensor,
    bin_positions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fit Gaussian (mu, sigma) from discrete probability vectors.

    Args:
        distributions: (N, K) probability vectors.
        bin_positions: (K,) positions for each bin.  Defaults to 0, 1, ..., K-1.

    Returns:
        (mu, sigma) each of shape (N,).
    """
    K = distributions.shape[-1]
    if bin_positions is None:
        bin_positions = torch.arange(
            K,
            dtype=distributions.dtype,
            device=distributions.device,
        )
    mu = (distributions * bin_positions).sum(dim=-1)
    var = (distributions * (bin_positions - mu.unsqueeze(-1)) ** 2).sum(dim=-1)
    sigma = (var + 1e-8).sqrt()
    return mu, sigma


def fisher_rao_gaussian(
    mu1: Tensor,
    sigma1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
) -> Tensor:
    """Fisher-Rao geodesic distance on the 1D Gaussian manifold.

    The Fisher information metric for N(mu, sigma^2) parameterized by
    (mu, sigma) is  ds^2 = dmu^2/sigma^2 + 2*dsigma^2/sigma^2.

    The closed-form geodesic distance is:

        d = sqrt(2) * arccosh(1 + (dmu^2 + 2*dsigma^2) / (4*sigma1*sigma2))

    Args:
        mu1, sigma1: Parameters of first Gaussian(s).
        mu2, sigma2: Parameters of second Gaussian(s).
            All tensors should be broadcastable.

    Returns:
        Geodesic distance(s), same shape as inputs after broadcasting.
    """
    dmu = mu1 - mu2
    dsigma = sigma1 - sigma2
    arg = 1.0 + (dmu**2 + 2.0 * dsigma**2) / (4.0 * sigma1 * sigma2 + 1e-12)
    return (2.0**0.5) * torch.acosh(arg.clamp(min=1.0))


def dissimilarity_from_confusion(P: Tensor, eps: float = 1e-10) -> Tensor:
    """Convert confusion matrix to dissimilarity cost matrix.

    Recipe:
    1. Geometric mean symmetrization: S[i,j] = sqrt(P[i,j] * P[j,i])
       (accounts for response bias)
    2. Zero diagonal, renormalize rows to sum to 1
       (conditional confusion: "given you're wrong, how are errors distributed?")
    3. Negative log: D[i,j] = -log(S_norm[i,j])
       (Shepard's law: similarity → distance)

    The result is a symmetric non-negative matrix suitable as an OT cost
    matrix. It is NOT guaranteed to satisfy the triangle inequality.

    Args:
        P: (N, N) confusion matrix where P[i,j] = P(output=j | correct=i).
        eps: Floor to avoid log(0).

    Returns:
        (N, N) symmetric dissimilarity matrix with zero diagonal.
    """
    # Zero diagonal and renormalize rows (conditional error distribution)
    P = P.clone()
    P.fill_diagonal_(0.0)
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=eps)
    # Geometric mean symmetrization (last structural step → result is symmetric)
    S = (P * P.T).sqrt().clamp(min=eps)
    # Negative log
    D = -S.log()
    D.fill_diagonal_(0.0)
    return D


def compute_and_plot_mds(
    cost_matrix: Tensor,
    labels: list[str],
    output_dir: str,
    colormap: str = "tab10",
    figure_format: str = "pdf",
) -> Tensor | None:
    """Run MDS on a dissimilarity matrix and save embedding + plot.

    Args:
        cost_matrix: (N, N) symmetric cost/dissimilarity matrix.
        labels: Length-N labels for each point.
        output_dir: Directory to save the MDS embedding (as
            ``mds_embedding.safetensors`` + ``mds_embedding.meta.json``) and
            the static MDS figure.
        colormap: Matplotlib colormap for coloring points by index.
        figure_format: ``png`` or ``pdf`` for the MDS scatter figure.

    Returns:
        (N, 2) embedding tensor, or None on failure.
    """
    import os

    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt

    from causalab.io.artifacts import save_tensors_with_meta
    from causalab.io.plots.figure_format import path_with_figure_format

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    embedding = mds.fit_transform(cost_matrix.numpy())
    embedding_t = torch.from_numpy(embedding)
    save_tensors_with_meta({"value": embedding_t}, {}, output_dir, "mds_embedding")

    n = len(labels)
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=60, c=colors, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (embedding[i, 0], embedding[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color=colors[i],
        )
    ax.set_title(f"MDS of cost matrix (stress={mds.stress_:.4f})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = path_with_figure_format(
        os.path.join(output_dir, "mds_embedding.pdf"),
        figure_format,
    )
    fig.savefig(out)
    plt.close(fig)

    return embedding_t


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DISTANCE_FUNCTIONS: dict[str, DistanceFn] = {
    "log_prob": euclidean_log_prob,
    "fisher_rao": fisher_rao,
    "hellinger": hellinger,
    "wasserstein1_cyclic": wasserstein1_cyclic,
    "wasserstein2_cyclic": wasserstein2_cyclic,
    "wasserstein1_noncyclic": wasserstein1_noncyclic,
    "wasserstein2_noncyclic": wasserstein2_noncyclic,
}

ALL_DISTANCE_FUNCTION_NAMES = (
    "log_prob",
    "fisher_rao",
    "fisher_rao_gaussian",
    "hellinger",
    "wasserstein1_cyclic",
    "wasserstein2_cyclic",
    "wasserstein1_noncyclic",
    "wasserstein2_noncyclic",
)

DISTANCE_LABELS = {
    "log_prob": "Euclidean log-prob",
    "fisher_rao": "Fisher-Rao",
    "fisher_rao_gaussian": "Fisher-Rao (Gaussian)",
    "hellinger": "Hellinger",
    "wasserstein1_cyclic": "Wasserstein-1 (cyclic)",
    "wasserstein2_cyclic": "Wasserstein-2 (cyclic)",
    "wasserstein1_noncyclic": "Wasserstein-1 (linear)",
    "wasserstein2_noncyclic": "Wasserstein-2 (linear)",
}


# ---------------------------------------------------------------------------
# Pairwise distance matrix
# ---------------------------------------------------------------------------


def pairwise_output_distance(
    distributions: Tensor,
    metric: str = "log_prob",
    bin_positions: Tensor | None = None,
) -> np.ndarray:
    """Compute pairwise distance matrix between output distributions.

    Args:
        distributions: (M, K) probability vectors.
        metric: One of ALL_DISTANCE_FUNCTION_NAMES.
        bin_positions: (K,) bin positions, needed for
            ``"fisher_rao_gaussian"`` (defaults to 0..K-1).

    Returns:
        (M, M) numpy distance matrix.
    """
    if metric not in ALL_DISTANCE_FUNCTION_NAMES:
        raise ValueError(
            f"Unknown distance function {metric!r}, "
            f"choose from {ALL_DISTANCE_FUNCTION_NAMES}"
        )

    if metric == "fisher_rao_gaussian":
        mu, sigma = _fit_gaussian_params(distributions, bin_positions)
        n = distributions.shape[0]
        D = torch.zeros(n, n)
        for i in range(n):
            D[i] = fisher_rao_gaussian(
                mu[i],
                sigma[i],
                mu,
                sigma,
            )
        return D.numpy()

    _pairwise_fns: dict[str, DistanceFn] = {
        "log_prob": euclidean_log_prob,
        "fisher_rao": fisher_rao,
        "hellinger": hellinger,
        "wasserstein1_cyclic": wasserstein1_cyclic,
        "wasserstein2_cyclic": wasserstein2_cyclic,
        "wasserstein1_noncyclic": wasserstein1_noncyclic,
        "wasserstein2_noncyclic": wasserstein2_noncyclic,
    }

    dist_fn = _pairwise_fns[metric]

    n = distributions.shape[0]
    D = torch.zeros(n, n)
    for i in range(n):
        D[i] = dist_fn(
            distributions[i].unsqueeze(0).expand(n, -1),
            distributions,
        )
    return D.numpy()


# ---------------------------------------------------------------------------
# Geodesic target distributions (displacement interpolation)
# ---------------------------------------------------------------------------


def hellinger_geodesic(p_start_W: Tensor, p_end_W: Tensor, alphas: Tensor) -> Tensor:
    """Geodesic on probability simplex: sqrt(P(a)) ~ (1-a)*sqrt(P_start) + a*sqrt(P_end).

    Returns (A, W) target distributions along the geodesic.
    """
    sqrt_start = p_start_W.sqrt()
    sqrt_end = p_end_W.sqrt()
    sqrt_interp = (1 - alphas[:, None]) * sqrt_start[None, :] + alphas[
        :, None
    ] * sqrt_end[None, :]
    sqrt_interp = sqrt_interp.clamp(min=0.0)
    p_target = sqrt_interp**2
    return p_target / p_target.sum(-1, keepdim=True)


def wasserstein2_cyclic_geodesic(
    p_start_W: Tensor,
    p_end_W: Tensor,
    alphas: Tensor,
    sigma: float | None = None,
) -> Tensor:
    """Displacement interpolation under W2 with cyclic ground metric."""
    W = p_start_W.shape[0]
    cost_sq, _, displacement = _get_cyclic_costs(W)
    cost_np = cost_sq.numpy()

    # Normalize to exact probability distributions (ot.emd requires equal mass)
    a = p_start_W.detach().double()
    b = p_end_W.detach().double()
    a = a / a.sum()
    b = b / b.sum()
    T = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), cost_np))
    sources = torch.arange(W).float().unsqueeze(1)

    result = torch.zeros(len(alphas), W, dtype=p_start_W.dtype)
    for a_idx, alpha in enumerate(alphas):
        pos = (sources + alpha * displacement) % W
        weights = cyclic_displacement_weights(pos, sigma, W)
        bins = (T.unsqueeze(-1) * weights).reshape(-1, W).sum(0)
        result[a_idx] = bins.to(dtype=p_start_W.dtype)

    return result / result.sum(-1, keepdim=True)


def wasserstein1_cyclic_geodesic(
    p_start_W: Tensor,
    p_end_W: Tensor,
    alphas: Tensor,
    sigma: float | None = None,
) -> Tensor:
    """Displacement interpolation under W1 with cyclic ground metric."""
    W = p_start_W.shape[0]
    _, cost, displacement = _get_cyclic_costs(W)
    cost_np = cost.numpy()

    # Normalize to exact probability distributions (ot.emd requires equal mass)
    a = p_start_W.detach().double()
    b = p_end_W.detach().double()
    a = a / a.sum()
    b = b / b.sum()
    T = torch.from_numpy(ot.emd(a.numpy(), b.numpy(), cost_np))
    sources = torch.arange(W).float().unsqueeze(1)

    result = torch.zeros(len(alphas), W, dtype=p_start_W.dtype)
    for a_idx, alpha in enumerate(alphas):
        pos = (sources + alpha * displacement) % W
        weights = cyclic_displacement_weights(pos, sigma, W)
        bins = (T.unsqueeze(-1) * weights).reshape(-1, W).sum(0)
        result[a_idx] = bins.to(dtype=p_start_W.dtype)

    return result / result.sum(-1, keepdim=True)


def sinkhorn_cyclic_geodesic(
    p_start_W: Tensor,
    p_end_W: Tensor,
    alphas: Tensor,
    sigma: float | None = None,
    reg: float = 0.1,
) -> Tensor:
    """Displacement interpolation under Sinkhorn-regularized OT with cyclic ground metric.

    Uses entropy-regularized transport plan (ot.sinkhorn) instead of exact OT,
    producing smoother/more diffused intermediate distributions.

    Args:
        p_start_W: (W,) start distribution.
        p_end_W: (W,) end distribution.
        alphas: (A,) interpolation parameters in [0, 1].
        sigma: circular std dev for deposit weights (None = linear interp to 2 nearest bins).
        reg: entropy regularization parameter for Sinkhorn.
    """
    W = p_start_W.shape[0]
    cost_sq, _, displacement = _get_cyclic_costs(W)
    cost_np = cost_sq.numpy()

    a = p_start_W.detach().double()
    b = p_end_W.detach().double()
    a = a / a.sum()
    b = b / b.sum()
    T = torch.from_numpy(ot.sinkhorn(a.numpy(), b.numpy(), cost_np, reg=reg))
    sources = torch.arange(W).float().unsqueeze(1)

    result = torch.zeros(len(alphas), W, dtype=p_start_W.dtype)
    for a_idx, alpha in enumerate(alphas):
        pos = (sources + alpha * displacement) % W
        weights = cyclic_displacement_weights(pos, sigma, W)
        bins = (T.unsqueeze(-1) * weights).reshape(-1, W).sum(0)
        result[a_idx] = bins.to(dtype=p_start_W.dtype)

    return result / result.sum(-1, keepdim=True)


def _fisher_rao_differentiable(p: Tensor, q: Tensor, eps: float = 1e-7) -> Tensor:
    """Fisher-Rao distance safe for backpropagation.

    Same formula as ``fisher_rao`` but with tighter clamps to avoid infinite
    gradients at simplex boundaries (sqrt at 0, arccos at 1).
    """
    q = q.to(p.device)
    bc = (p * q).clamp(min=eps).sqrt().sum(-1)
    return 2.0 * torch.acos(bc.clamp(max=1.0 - eps))


@dataclasses.dataclass
class ConformalGeodesicResult:
    """Diagnostics from a conformal geodesic optimization run."""

    path: Tensor  # (A, W+1)
    loss_history: list[float]
    n_steps_run: int
    converged: bool


def make_knn_cost_fn(
    natural_dists: Tensor,
    k: int,
    eps: float = 1e-7,
) -> ConformalCostFn:
    """Create a k-NN Fisher-Rao conformal cost function."""

    def cost_fn(path: Tensor) -> Tensor:
        p_exp = path.unsqueeze(-2)  # (K, 1, W+1)
        d_nat = _fisher_rao_differentiable(p_exp, natural_dists, eps)  # (K, M)
        k_actual = min(k, d_nat.shape[-1])
        return d_nat.topk(k_actual, dim=-1, largest=False).values.mean(dim=-1)

    return cost_fn


def hellinger_distance_to_manifold(
    distributions: Tensor,
    belief_manifold: Any,
    eps: float = 1e-8,
) -> Tensor:
    """Hellinger distance from each distribution row to the belief manifold.

    Args:
        distributions: (..., W+1) probability vectors on the simplex.
        belief_manifold: SplineManifold fit in Hellinger space. Must be built
            with ``sphere_project=True`` so ``decode(u)`` is on the unit sphere.
        eps: Tolerance for sqrt and norm clamps.

    Returns:
        (...,) tensor of Hellinger distances, in [0, 1].

    Differentiable through ``distributions`` (the nearest u is computed under
    no_grad and detached, so the gradient flows only through h = sqrt(p)).
    """
    mf_device = belief_manifold.centroids.device
    dtype = belief_manifold.centroids.dtype
    h = torch.sqrt(distributions.to(device=mf_device, dtype=dtype).clamp(min=eps))
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=eps)
    with torch.no_grad():
        u, _ = belief_manifold.encode_to_nearest_point(h.detach())
        h_proj = belief_manifold.decode(u)  # on sphere (sphere_project=True)
    return (h - h_proj.to(h.dtype)).norm(dim=-1) / (2.0**0.5)


def make_manifold_cost_fn(
    belief_manifold: Any,
    eps: float = 1e-8,
) -> ConformalCostFn:
    """Conformal cost function: Hellinger distance from path points to the
    belief manifold. Differentiable through the path; nearest u is detached.
    """

    def cost_fn(path: Tensor) -> Tensor:
        orig_shape = path.shape[:-1]
        flat = path.reshape(-1, path.shape[-1])
        cost = hellinger_distance_to_manifold(flat, belief_manifold, eps=eps)
        return cost.to(path.device).reshape(orig_shape)

    return cost_fn


def _build_cost_fn(
    natural_dists: Tensor | None,
    k: int,
    belief_manifold: Any | None,
    eps: float = 1e-7,
) -> ConformalCostFn:
    """Build the appropriate conformal cost function."""
    if belief_manifold is not None:
        return make_manifold_cost_fn(belief_manifold, eps)
    assert natural_dists is not None, "Need natural_dists when no belief_manifold"
    return make_knn_cost_fn(natural_dists, k, eps)


def _hellinger_differentiable(p: Tensor, q: Tensor, eps: float = 1e-7) -> Tensor:
    """Differentiable Hellinger distance. Same as hellinger() but with eps."""
    q = q.to(p.device)
    return (p.clamp(min=eps).sqrt() - q.clamp(min=eps).sqrt()).pow(2).sum(-1).clamp(
        min=0
    ).sqrt() / (2.0**0.5)


# Registry for differentiable base metrics used in conformal path length
_DIFFERENTIABLE_METRICS = {
    "fisher_rao": _fisher_rao_differentiable,
    "hellinger": _hellinger_differentiable,
}


def _conformal_path_length(
    path: Tensor,
    cost_fn: ConformalCostFn,
    alpha_conf: float,
    eps: float = 1e-7,
    elastic_weight: float = 0.0,
    equidist_weight: float = 0.0,
    base_metric: str = "hellinger",
) -> Tensor:
    """Differentiable conformal path length for optimization.

    Args:
        path: (K, W+1) distributions along the path (including endpoints).
        cost_fn: Maps (K, D) path to (K,) non-negative cost per point.
        alpha_conf: Conformal exponent in exp(alpha * c).
        eps: Numerical stability epsilon.
        elastic_weight: Discrete curvature penalty weight. 0 = off.
        equidist_weight: Step-size uniformity penalty weight. 0 = off.
        base_metric: "hellinger" or "fisher_rao" for step-wise distances.

    Returns:
        Scalar tensor: conformal path length L_c (+ regularization penalties).
    """
    metric_fn = _DIFFERENTIABLE_METRICS[base_metric]
    d_steps = metric_fn(path[:-1], path[1:], eps)  # (K-1,)
    c_vals = cost_fn(path)  # (K,)

    weights = torch.exp(alpha_conf * c_vals)
    trap_weights = (weights[:-1] + weights[1:]) / 2.0

    # Normalize by number of segments so gradient scale is independent of n_steps
    loss = (trap_weights * d_steps).sum() / max(d_steps.shape[0], 1)

    if elastic_weight > 0.0 and path.shape[0] >= 3:
        q = torch.sqrt(path.clamp(min=eps))
        curvature = q[2:] - 2.0 * q[1:-1] + q[:-2]
        elastic_penalty = (curvature**2).sum(dim=-1).mean()
        loss = loss + elastic_weight * elastic_penalty

    if equidist_weight > 0.0 and d_steps.shape[0] >= 2:
        mean_step = d_steps.mean()
        equidist_penalty = ((d_steps - mean_step) ** 2).mean() / (mean_step**2 + eps)
        loss = loss + equidist_weight * equidist_penalty

    return loss


def conformal_geodesic(
    p_start_W1: Tensor,
    p_end_W1: Tensor,
    alphas: Tensor,
    natural_dists: Tensor | None = None,
    alpha_conf: float = 1.0,
    k: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    lr: float = 1.0,
    init: str | Tensor = "hellinger",
    device: torch.device | str = "cpu",
    inner_max_iter: int = 20,
    base_metric: str = "hellinger",
    convergence_window: int = 1,
    return_diagnostics: bool = False,
    elastic_weight: float = 0.0,
    equidist_weight: float = 0.0,
    belief_manifold: Any = None,
) -> Tensor | ConformalGeodesicResult:
    """Path on the (W+1)-simplex minimizing conformal path length L_c.

    Uses L-BFGS optimization with softmax-parameterized interior points
    over the full (W+1)-dimensional simplex (concept probs + 'other' bin).

    Args:
        p_start_W1: (W+1,) start distribution (with 'other' bin).
        p_end_W1: (W+1,) end distribution (with 'other' bin).
        alphas: (A,) interpolation parameters in [0, 1].
        natural_dists: (M, W+1) natural output distributions with 'other' bin.
        alpha_conf: Conformal exponent.  Higher values penalize off-manifold
            regions more aggressively.
        k: Number of nearest neighbors for conformal cost.
        max_iter: Maximum L-BFGS outer iterations.
        tol: Convergence tolerance on relative change in L_c.
        lr: L-BFGS learning rate.
        inner_max_iter: Max line-search steps per L-BFGS outer step.
        convergence_window: Number of consecutive steps that must all
            show relative change < tol before early stopping.
        init: Initialization for interior path points.  ``"hellinger"``
            (default) uses the Hellinger geodesic.  A tensor of shape
            ``(n_interior, W+1)`` is used directly as the initial path.
        return_diagnostics: If True, return a ConformalGeodesicResult with
            loss_history and convergence info instead of a plain Tensor.

    Returns:
        (A, W+1) distributions along the conformal-optimal path, or
        ConformalGeodesicResult if return_diagnostics is True.
    """
    import logging

    logger = logging.getLogger(__name__)

    A = len(alphas)
    D = p_start_W1.shape[0]  # W+1

    # Separate interior alphas (those strictly between 0 and 1)
    interior_mask = (alphas > 0) & (alphas < 1)
    n_interior = interior_mask.sum().item()

    if n_interior == 0:
        result = torch.zeros(A, D, dtype=p_start_W1.dtype)
        for i, a in enumerate(alphas):
            result[i] = p_start_W1 if a <= 0 else p_end_W1
        return result

    interior_alphas = alphas[interior_mask]

    # Initialize interior path points
    if isinstance(init, Tensor):
        init_path = init
    else:
        init_path = hellinger_geodesic(
            p_start_W1, p_end_W1, interior_alphas
        )  # (n_interior, W+1)

    dev = torch.device(device)
    nat = natural_dists.detach().double().to(dev) if natural_dists is not None else None
    p_start_d = p_start_W1.detach().double().to(dev)
    p_end_d = p_end_W1.detach().double().to(dev)
    eps = 1e-8

    cost_fn = _build_cost_fn(nat, k, belief_manifold, eps)

    # Parameterize as logits -> softmax over full (W+1)-simplex
    logits = torch.nn.Parameter(init_path.double().clamp(min=eps).log().to(dev))
    optimizer = torch.optim.LBFGS(
        [logits],
        lr=lr,
        max_iter=inner_max_iter,
        line_search_fn="strong_wolfe",
    )

    loss_history: list[float] = []
    converged = False
    for step in range(max_iter):

        def closure() -> Tensor:
            optimizer.zero_grad()
            interior = torch.softmax(logits, dim=-1)
            full_path = torch.cat(
                [
                    p_start_d.unsqueeze(0),
                    interior,
                    p_end_d.unsqueeze(0),
                ]
            )
            loss = _conformal_path_length(
                full_path,
                cost_fn,
                alpha_conf,
                eps,
                elastic_weight=elastic_weight,
                equidist_weight=equidist_weight,
                base_metric=base_metric,
            )
            loss.backward()
            return loss

        loss_t = optimizer.step(closure)
        loss_val = (
            float(loss_t.detach()) if isinstance(loss_t, Tensor) else float(loss_t)
        )

        if math.isnan(loss_val):
            logger.warning(
                "conformal_geodesic: NaN loss at step %d, using Hellinger init", step
            )
            break

        loss_history.append(loss_val)
        if len(loss_history) >= convergence_window + 1:
            recent = loss_history[-(convergence_window + 1) :]
            max_rel_change = max(
                abs(recent[i + 1] - recent[i]) / max(abs(recent[i]), 1.0)
                for i in range(convergence_window)
            )
            if max_rel_change < tol:
                converged = True
                break

    n_steps_run = len(loss_history)
    if converged:
        logger.info(
            "conformal_geodesic: converged at step %d/%d (window=%d, tol=%g)",
            n_steps_run,
            max_iter,
            convergence_window,
            tol,
        )
    else:
        logger.info(
            "conformal_geodesic: reached max_iter=%d without converging (last loss=%.6g)",
            max_iter,
            loss_history[-1] if loss_history else float("nan"),
        )

    # Extract optimized path (fall back to init on NaN)
    with torch.no_grad():
        optimized = torch.softmax(logits, dim=-1)
        if optimized.isnan().any():
            logger.warning(
                "conformal_geodesic: NaN in result, falling back to Hellinger"
            )
            optimized = init_path.double()

    # Assemble full result at requested alphas
    out_dtype = p_start_W1.dtype
    result = torch.zeros(A, D, dtype=out_dtype)
    interior_idx = 0
    for i, a in enumerate(alphas):
        if a <= 0:
            result[i] = p_start_W1
        elif a >= 1:
            result[i] = p_end_W1
        else:
            result[i] = optimized[interior_idx].to(dtype=out_dtype)
            interior_idx += 1

    if return_diagnostics:
        return ConformalGeodesicResult(
            path=result,
            loss_history=loss_history,
            n_steps_run=n_steps_run,
            converged=converged,
        )
    return result


def conformal_geodesic_basin_hopping(
    p_start_W1: Tensor,
    p_end_W1: Tensor,
    alphas: Tensor,
    natural_dists: Tensor | None = None,
    alpha_conf: float = 1.0,
    k: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    lr: float = 1.0,
    init: str | Tensor = "hellinger",
    device: torch.device | str = "cpu",
    inner_max_iter: int = 20,
    convergence_window: int = 1,
    return_diagnostics: bool = False,
    n_hops: int = 50,
    step_size: float = 0.5,
    temperature: float = 1.0,
    n_candidates: int = 5,
    elastic_weight: float = 0.0,
    equidist_weight: float = 0.0,
    belief_manifold: Any = None,
    base_metric: str = "hellinger",
) -> Tensor | ConformalGeodesicResult:
    """Hybrid basin hopping for conformal geodesic on the (W+1)-simplex.

    Phase 1: Cheap grad-free search — perturb logits, evaluate conformal path
    length without gradients, Metropolis accept/reject.
    Phase 2: Run full L-BFGS (via conformal_geodesic) from top-N candidates.

    Accepts all arguments of conformal_geodesic plus basin hopping parameters.
    """
    import logging

    logger = logging.getLogger(__name__)

    A = len(alphas)
    D = p_start_W1.shape[0]
    interior_mask = (alphas > 0) & (alphas < 1)
    n_interior = interior_mask.sum().item()

    if n_interior == 0:
        return conformal_geodesic(
            p_start_W1,
            p_end_W1,
            alphas,
            natural_dists,
            alpha_conf=alpha_conf,
            k=k,
            init=init,
            device=device,
            return_diagnostics=return_diagnostics,
            belief_manifold=belief_manifold,
            base_metric=base_metric,
        )

    interior_alphas = alphas[interior_mask]

    eps = 1e-8
    if isinstance(init, Tensor):
        init_path = init
    else:
        init_path = hellinger_geodesic(p_start_W1, p_end_W1, interior_alphas)

    dev = torch.device(device)
    nat = natural_dists.detach().double().to(dev) if natural_dists is not None else None
    p_start_d = p_start_W1.detach().double().to(dev)
    p_end_d = p_end_W1.detach().double().to(dev)
    init_logits = init_path.double().clamp(min=eps).log().to(dev)
    init_scale = init_logits.std().item() + 1e-6

    cost_fn = _build_cost_fn(nat, k, belief_manifold, eps)

    def eval_loss(logits: Tensor) -> float:
        interior = torch.softmax(logits, dim=-1)
        full_path = torch.cat([p_start_d.unsqueeze(0), interior, p_end_d.unsqueeze(0)])
        return _conformal_path_length(
            full_path,
            cost_fn,
            alpha_conf,
            eps,
            elastic_weight=elastic_weight,
            equidist_weight=equidist_weight,
            base_metric=base_metric,
        ).item()

    # --- Phase 1: cheap candidate search (no gradients) ---
    logger.info("  Belief basin hopping phase 1: %d hops (grad-free)", n_hops)
    current_logits = init_logits.clone()
    with torch.no_grad():
        current_loss = eval_loss(current_logits)
    candidates: list[tuple[Tensor, float]] = [(current_logits.clone(), current_loss)]

    for hop in range(n_hops):
        perturbation = torch.randn_like(current_logits) * step_size * init_scale
        candidate_logits = current_logits + perturbation

        with torch.no_grad():
            candidate_loss = eval_loss(candidate_logits)
        candidates.append((candidate_logits.clone(), candidate_loss))

        delta = candidate_loss - current_loss
        if delta < 0 or _random.random() < math.exp(-delta / (temperature + 1e-12)):
            current_logits = candidate_logits
            current_loss = candidate_loss

    candidates.sort(key=lambda x: x[1])
    n_cand = min(n_candidates, len(candidates))
    logger.info(
        "  Phase 1 done: best = %.6f, refining top %d", candidates[0][1], n_cand
    )

    # --- Phase 2: L-BFGS refinement via conformal_geodesic ---
    best_result = None
    best_loss = float("inf")
    for i, (cand_logits, cand_search_loss) in enumerate(candidates[:n_cand]):
        logger.info(
            "  Phase 2: refining candidate %d/%d (search loss = %.6f)",
            i + 1,
            n_cand,
            cand_search_loss,
        )
        # Convert logits back to simplex for init
        cand_init = torch.softmax(cand_logits, dim=-1).float()
        result = conformal_geodesic(
            p_start_W1,
            p_end_W1,
            alphas,
            natural_dists,
            alpha_conf=alpha_conf,
            k=k,
            max_iter=max_iter,
            tol=tol,
            lr=lr,
            init=cand_init,
            device=device,
            inner_max_iter=inner_max_iter,
            convergence_window=convergence_window,
            return_diagnostics=True,
            elastic_weight=elastic_weight,
            equidist_weight=equidist_weight,
            belief_manifold=belief_manifold,
            base_metric=base_metric,
        )
        final_loss = result.loss_history[-1] if result.loss_history else float("inf")
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = result

    logger.info("  Belief basin hopping done: best loss = %.6f", best_loss)

    if return_diagnostics:
        return best_result
    return best_result.path


def _lerp_on_sphere(h_a: Tensor, h_b: Tensor, t: Tensor, eps: float = 1e-8) -> Tensor:
    """Slerp between two points on the unit sphere in √p space.

    Falls back to linear interpolation (+ renormalization) when points
    are nearly coincident, avoiding sin(0)/0 singularity.
    """
    dot = (h_a * h_b).sum().clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Smooth blend: use slerp when omega is large, lerp when small
    use_slerp = (sin_omega.abs() > 1e-6).float()
    safe_sin = sin_omega.clamp(min=1e-6)
    w_a_slerp = torch.sin((1 - t) * omega) / safe_sin
    w_b_slerp = torch.sin(t * omega) / safe_sin
    w_a = use_slerp * w_a_slerp + (1 - use_slerp) * (1 - t)
    w_b = use_slerp * w_b_slerp + (1 - use_slerp) * t
    result = w_a.unsqueeze(-1) * h_a + w_b.unsqueeze(-1) * h_b
    # Renormalize to stay on sphere
    return result / result.norm(dim=-1, keepdim=True).clamp(min=eps)


def _resample_arc_length(points: Tensor, n_out: int, eps: float = 1e-8) -> Tensor:
    """Resample a curve at equal arc-length intervals.

    Args:
        points: (M, D) densely sampled curve.
        n_out: Number of output points (including endpoints).

    Returns:
        (n_out, D) resampled points at uniform arc-length spacing.
    """
    diffs = points[1:] - points[:-1]
    seg_lengths = diffs.norm(dim=-1)  # (M-1,)
    cum_length = torch.cat(
        [
            torch.zeros(1, device=points.device, dtype=points.dtype),
            seg_lengths.cumsum(0),
        ]
    )
    total_val = cum_length[-1].detach().item()
    if total_val < eps:
        # Degenerate: all points collapsed. Return uniform copies.
        idx = torch.linspace(0, points.shape[0] - 1, n_out).long()
        return points[idx]

    target = torch.linspace(
        0, total_val, n_out, device=points.device, dtype=points.dtype
    )
    # Find bracketing indices for each target arc length
    # Use detached cum_length for searchsorted (index-only op), keep grads in interpolation
    idx = torch.searchsorted(cum_length.detach(), target).clamp(1, len(cum_length) - 1)
    t_lo = cum_length[idx - 1]
    t_hi = cum_length[idx]
    frac = ((target - t_lo) / (t_hi - t_lo).clamp(min=eps)).unsqueeze(-1)
    return points[idx - 1] * (1 - frac) + points[idx] * frac


def _init_ctrl_from_belief_manifold(
    p_start_W1: Tensor,
    p_end_W1: Tensor,
    belief_manifold: Any,
    n_ctrl: int,
    eps: float = 1e-8,
) -> Tensor:
    """Initialize control points by sampling the belief manifold arc.

    Encodes endpoints onto the manifold, interpolates in intrinsic space
    (shortest arc for periodic dims), decodes, and samples n_ctrl points.
    """
    dev = belief_manifold.centroids.device
    dtype = belief_manifold.centroids.dtype
    h_start = torch.sqrt(
        p_start_W1.to(device=dev, dtype=dtype).clamp(min=eps)
    ).unsqueeze(0)
    h_end = torch.sqrt(p_end_W1.to(device=dev, dtype=dtype).clamp(min=eps)).unsqueeze(0)

    u_start, _ = belief_manifold.encode_to_nearest_point(h_start)
    u_end, _ = belief_manifold.encode_to_nearest_point(h_end)
    u_start, u_end = u_start.squeeze(0), u_end.squeeze(0)

    delta = u_end - u_start
    if belief_manifold.periodic_dims:
        for pd, per in zip(belief_manifold.periodic_dims, belief_manifold.periods):
            if abs(float(delta[pd])) > per / 2:
                delta[pd] -= torch.sign(delta[pd]) * per

    t = torch.linspace(0, 1, n_ctrl, dtype=u_start.dtype, device=u_start.device)
    u_path = u_start.unsqueeze(0) + t.unsqueeze(1) * delta.unsqueeze(0)
    with torch.no_grad():
        h_path = belief_manifold.decode(u_path)
    # Project to unit sphere
    h_path = h_path.clamp(min=0).double()
    h_path = h_path / h_path.norm(dim=-1, keepdim=True).clamp(min=eps)
    return h_path


def conformal_geodesic_tps(
    p_start_W1: Tensor,
    p_end_W1: Tensor,
    alphas: Tensor,
    natural_dists: Tensor | None = None,
    alpha_conf: float = 1.0,
    k: int = 1,
    max_iter: int = 200,
    tol: float = 1e-8,
    lr: float = 1.0,
    device: torch.device | str = "cpu",
    inner_max_iter: int = 20,
    convergence_window: int = 5,
    return_diagnostics: bool = False,
    n_control_points: int = 5,
    n_dense: int = 100,
    belief_manifold: Any = None,
    base_metric: str = "hellinger",
    **_kwargs,
) -> Tensor | ConformalGeodesicResult:
    """Conformal geodesic with shape/spacing separation.

    Optimizes path shape via a small number of control points in √p space
    (on the unit sphere). At each optimizer step:
    1. Slerp between consecutive control points to get a dense curve
    2. Resample at equal arc-length intervals
    3. Project to simplex, compute L_c

    Spacing is always uniform by construction — no equidistribution penalty
    needed. The optimizer only controls the curve's shape.

    Always initializes from the Hellinger geodesic (chord), which is the
    shortest path at α=0.  The optimizer bends it toward low-cost regions
    as α increases.

    Args:
        n_control_points: Control points including endpoints (≥3).
        n_dense: Dense samples per segment for arc-length resampling.
        Other args: same as conformal_geodesic.
    """
    import logging

    logger = logging.getLogger(__name__)

    A = len(alphas)
    D = p_start_W1.shape[0]  # W+1
    eps = 1e-8

    interior_mask = (alphas > 0) & (alphas < 1)
    n_interior = interior_mask.sum().item()

    if n_interior == 0:
        result = torch.zeros(A, D, dtype=p_start_W1.dtype)
        for i, a in enumerate(alphas):
            result[i] = p_start_W1 if a <= 0 else p_end_W1
        if return_diagnostics:
            return ConformalGeodesicResult(
                path=result, loss_history=[], n_steps_run=0, converged=True
            )
        return result

    dev = torch.device(device)
    nat = natural_dists.detach().double().to(dev) if natural_dists is not None else None
    p_start_d = p_start_W1.detach().double().to(dev)
    p_end_d = p_end_W1.detach().double().to(dev)
    n_ctrl = max(n_control_points, 3)
    n_eval = n_interior + 2  # total path points for loss (interior + 2 endpoints)

    cost_fn = _build_cost_fn(nat, k, belief_manifold, eps)

    # Build candidate initializations
    h_start = torch.sqrt(p_start_d.clamp(min=eps))
    h_start = h_start / h_start.norm().clamp(min=eps)
    h_end = torch.sqrt(p_end_d.clamp(min=eps))
    h_end = h_end / h_end.norm().clamp(min=eps)
    t_ctrl = torch.linspace(0, 1, n_ctrl, dtype=torch.double, device=dev)

    inits = {"chord": _lerp_on_sphere(h_start, h_end, t_ctrl, eps)}
    if belief_manifold is not None:
        inits["manifold"] = _init_ctrl_from_belief_manifold(
            p_start_W1,
            p_end_W1,
            belief_manifold,
            n_ctrl,
            eps,
        ).to(dev)

    def _build_path(ctrl_start, ctrl_free, ctrl_end) -> Tensor:
        """Build arc-length-resampled path from control points."""
        all_ctrl = torch.cat([ctrl_start, ctrl_free, ctrl_end])
        all_ctrl = all_ctrl.clamp(min=0)
        all_ctrl = all_ctrl / all_ctrl.norm(dim=-1, keepdim=True).clamp(min=eps)

        dense_parts = []
        for seg in range(n_ctrl - 1):
            t_seg = torch.linspace(0, 1, n_dense, dtype=torch.double, device=dev)
            if seg > 0:
                t_seg = t_seg[1:]
            seg_pts = _lerp_on_sphere(all_ctrl[seg], all_ctrl[seg + 1], t_seg, eps)
            dense_parts.append(seg_pts)
        dense_curve = torch.cat(dense_parts)

        resampled = _resample_arc_length(dense_curve, n_eval, eps)
        return resampled**2

    def _run_one_init(ctrl_init):
        ctrl_start = ctrl_init[:1].detach()
        ctrl_end = ctrl_init[-1:].detach()
        ctrl_free = torch.nn.Parameter(ctrl_init[1:-1].clone())
        opt = torch.optim.LBFGS(
            [ctrl_free],
            lr=lr,
            max_iter=inner_max_iter,
            line_search_fn="strong_wolfe",
        )
        hist = []
        conv = False
        for step in range(max_iter):

            def closure():
                opt.zero_grad()
                path = _build_path(ctrl_start, ctrl_free, ctrl_end)
                loss = _conformal_path_length(
                    path, cost_fn, alpha_conf, eps, base_metric=base_metric
                )
                loss.backward()
                return loss

            loss_t = opt.step(closure)
            loss_val = (
                float(loss_t.detach()) if isinstance(loss_t, Tensor) else float(loss_t)
            )
            if math.isnan(loss_val):
                break
            hist.append(loss_val)
            min_steps = min(30, max_iter)  # run at least 30 steps before early stopping
            if step >= min_steps and len(hist) >= convergence_window + 1:
                recent = hist[-(convergence_window + 1) :]
                max_rel = max(
                    abs(recent[i + 1] - recent[i]) / max(abs(recent[i]), 1.0)
                    for i in range(convergence_window)
                )
                if max_rel < tol:
                    conv = True
                    break
        with torch.no_grad():
            final_path = _build_path(ctrl_start, ctrl_free, ctrl_end)
        return final_path, hist, conv

    # Try all initializations, keep the best
    best_path = None
    best_loss = float("inf")
    best_history = []
    best_converged = False
    for init_name, ctrl_init in inits.items():
        path, hist, conv = _run_one_init(ctrl_init)
        final_loss = hist[-1] if hist else float("inf")
        logger.info(
            "  %s init: final loss=%.6f (%d steps, converged=%s)",
            init_name,
            final_loss,
            len(hist),
            conv,
        )
        if final_loss < best_loss:
            best_loss = final_loss
            best_path = path
            best_history = hist
            best_converged = conv

    loss_history = best_history
    converged = best_converged
    n_steps_run = len(loss_history)

    if converged:
        logger.info(
            "conformal_geodesic_tps: best loss=%.6f (%d steps)", best_loss, n_steps_run
        )
    else:
        logger.info(
            "conformal_geodesic_tps: best loss=%.6f (max_iter=%d)", best_loss, max_iter
        )

    # Extract final path at requested alphas
    with torch.no_grad():
        full_path = best_path
        if full_path is None or full_path.isnan().any():
            logger.warning("conformal_geodesic_tps: NaN, falling back to Hellinger")
            interior_alphas = alphas[interior_mask]
            full_path = torch.cat(
                [
                    p_start_d.unsqueeze(0),
                    hellinger_geodesic(p_start_W1, p_end_W1, interior_alphas)
                    .double()
                    .to(dev),
                    p_end_d.unsqueeze(0),
                ]
            )

    # The resampled path has n_eval = n_interior + 2 points
    # Map to the requested alphas
    out_dtype = p_start_W1.dtype
    result = torch.zeros(A, D, dtype=out_dtype)
    interior_idx = 0
    for i, a in enumerate(alphas):
        if a <= 0:
            result[i] = p_start_W1
        elif a >= 1:
            result[i] = p_end_W1
        else:
            result[i] = full_path[1 + interior_idx].cpu().to(dtype=out_dtype)
            interior_idx += 1

    if return_diagnostics:
        return ConformalGeodesicResult(
            path=result,
            loss_history=loss_history,
            n_steps_run=n_steps_run,
            converged=converged,
        )
    return result


def _neb_project_gradients(
    grad: Tensor,
    path: Tensor,
    spring_k: float,
    eps: float = 1e-8,
) -> Tensor:
    """Apply NEB force decomposition to interior-point gradients.

    For each interior waypoint i the raw gradient (from the conformal path
    length) is decomposed into parallel and perpendicular components w.r.t.
    the local path tangent.  The returned "nudged" gradient combines:

      * **Perpendicular potential force** — gradient of conformal cost with
        the tangent component removed.  This shapes the path without sliding
        points along it.
      * **Parallel spring force** — penalises difference in neighbouring
        step sizes, projected onto the tangent.  This spaces points without
        distorting the path shape.

    Operates in logit-space (the parameterization used by the optimizer).

    Args:
        grad: (n_interior, D) raw gradient of the loss w.r.t. interior logits.
        path: (K, D) full path probabilities (start + interior + end), used
            to compute tangent directions.
        spring_k: Spring constant for the parallel spacing force.
        eps: Numerical stability.

    Returns:
        (n_interior, D) NEB-projected gradient to replace the raw gradient.
    """
    K = path.shape[0]
    n_interior = K - 2  # exclude fixed endpoints

    # Tangent at each interior point: central difference on the probability path
    # τ_i = (p_{i+1} - p_{i-1}) / ||p_{i+1} - p_{i-1}||
    tangents = path[2:] - path[:-2]  # (n_interior, D)
    tangent_norms = tangents.norm(dim=-1, keepdim=True).clamp(min=eps)
    tau = tangents / tangent_norms  # (n_interior, D) unit tangents

    # 1. Perpendicular potential: remove parallel component of raw gradient
    grad_par = (grad * tau).sum(dim=-1, keepdim=True) * tau
    grad_perp = grad - grad_par  # perpendicular component (shapes the path)

    # 2. Parallel spring force: penalise unequal step sizes
    # Compute step sizes in probability space
    d_steps = (path[1:] - path[:-1]).norm(dim=-1)  # (K-1,)
    # Spring force magnitude at interior point i: k * (|d_{i+1}| - |d_i|)
    # d_steps[i] is the step from point i to i+1, so for interior index j
    # (0-based within interior), the surrounding steps are d_steps[j] and d_steps[j+1]
    spring_mag = spring_k * (d_steps[1:] - d_steps[:-1])  # (n_interior,)
    # Project onto tangent: this is a gradient (pointing in descent direction),
    # so positive spring_mag means step i+1 > step i, and the spring should
    # push the point forward (along +tau) to equalise.  We negate because
    # optimizers minimise: grad = -force.
    spring_grad = -spring_mag.unsqueeze(-1) * tau  # (n_interior, D)

    return grad_perp + spring_grad


def conformal_geodesic_neb(
    p_start_W1: Tensor,
    p_end_W1: Tensor,
    alphas: Tensor,
    natural_dists: Tensor | None = None,
    alpha_conf: float = 1.0,
    k: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    lr: float = 0.01,
    init: str | Tensor = "hellinger",
    device: torch.device | str = "cpu",
    convergence_window: int = 5,
    return_diagnostics: bool = False,
    elastic_weight: float = 0.0,
    spring_k: float = 1.0,
    belief_manifold: Any = None,
    base_metric: str = "hellinger",
    **_kwargs,
) -> Tensor | ConformalGeodesicResult:
    """Conformal geodesic with Nudged Elastic Band (NEB) force projection.

    Instead of adding a soft equidistribution penalty, NEB decomposes the
    gradient at each waypoint into parallel (along path tangent) and
    perpendicular components:

      * Only the **perpendicular** component of the conformal cost gradient
        is kept — it shapes the path without sliding points along it.
      * Only the **parallel** component of the spring force is kept — it
        spaces points evenly without distorting the path shape.

    This cleanly decouples "where the path goes" from "how points are
    distributed along it."

    Uses Adam (not L-BFGS) because the gradient projection breaks the
    curvature assumptions that L-BFGS relies on.

    Args:
        spring_k: Spring constant for the NEB parallel spacing force.
            Higher values enforce more uniform spacing.
        Other args: same as conformal_geodesic (inner_max_iter is ignored).
    """
    import logging

    logger = logging.getLogger(__name__)

    A = len(alphas)
    D = p_start_W1.shape[0]

    interior_mask = (alphas > 0) & (alphas < 1)
    n_interior = interior_mask.sum().item()

    if n_interior == 0:
        result = torch.zeros(A, D, dtype=p_start_W1.dtype)
        for i, a in enumerate(alphas):
            result[i] = p_start_W1 if a <= 0 else p_end_W1
        return result

    interior_alphas = alphas[interior_mask]

    if isinstance(init, Tensor):
        init_path = init
    else:
        init_path = hellinger_geodesic(p_start_W1, p_end_W1, interior_alphas)

    dev = torch.device(device)
    nat = natural_dists.detach().double().to(dev) if natural_dists is not None else None
    p_start_d = p_start_W1.detach().double().to(dev)
    p_end_d = p_end_W1.detach().double().to(dev)
    eps = 1e-8

    cost_fn = _build_cost_fn(nat, k, belief_manifold, eps)

    logits = torch.nn.Parameter(init_path.double().clamp(min=eps).log().to(dev))
    optimizer = torch.optim.Adam([logits], lr=lr)

    loss_history: list[float] = []
    converged = False

    for step in range(max_iter):
        optimizer.zero_grad()

        interior = torch.softmax(logits, dim=-1)
        full_path = torch.cat(
            [
                p_start_d.unsqueeze(0),
                interior,
                p_end_d.unsqueeze(0),
            ]
        )
        loss = _conformal_path_length(
            full_path,
            cost_fn,
            alpha_conf,
            eps,
            elastic_weight=elastic_weight,
            base_metric=base_metric,
        )
        loss.backward()

        loss_val = loss.item()
        if math.isnan(loss_val):
            logger.warning("conformal_geodesic_neb: NaN loss at step %d", step)
            break

        # NEB projection: replace raw gradient with nudged gradient
        with torch.no_grad():
            logits.grad.copy_(
                _neb_project_gradients(
                    logits.grad,
                    full_path.detach(),
                    spring_k,
                    eps,
                )
            )

        optimizer.step()

        loss_history.append(loss_val)
        if len(loss_history) >= convergence_window + 1:
            recent = loss_history[-(convergence_window + 1) :]
            max_rel_change = max(
                abs(recent[i + 1] - recent[i]) / max(abs(recent[i]), 1.0)
                for i in range(convergence_window)
            )
            if max_rel_change < tol:
                converged = True
                break

    n_steps_run = len(loss_history)
    if converged:
        logger.info(
            "conformal_geodesic_neb: converged at step %d/%d", n_steps_run, max_iter
        )
    else:
        logger.info(
            "conformal_geodesic_neb: reached max_iter=%d (last loss=%.6g)",
            max_iter,
            loss_history[-1] if loss_history else float("nan"),
        )

    with torch.no_grad():
        optimized = torch.softmax(logits, dim=-1)
        if optimized.isnan().any():
            logger.warning(
                "conformal_geodesic_neb: NaN in result, falling back to Hellinger"
            )
            optimized = init_path.double()

    out_dtype = p_start_W1.dtype
    result = torch.zeros(A, D, dtype=out_dtype)
    interior_idx = 0
    for i, a in enumerate(alphas):
        if a <= 0:
            result[i] = p_start_W1
        elif a >= 1:
            result[i] = p_end_W1
        else:
            result[i] = optimized[interior_idx].to(dtype=out_dtype)
            interior_idx += 1

    if return_diagnostics:
        return ConformalGeodesicResult(
            path=result,
            loss_history=loss_history,
            n_steps_run=n_steps_run,
            converged=converged,
        )
    return result


GEODESIC_FNS = {
    "hellinger": hellinger_geodesic,
    "wasserstein1_cyclic": wasserstein1_cyclic_geodesic,
    "wasserstein2_cyclic": wasserstein2_cyclic_geodesic,
    "sinkhorn_cyclic": sinkhorn_cyclic_geodesic,
    "conformal": conformal_geodesic,
    "conformal_basin_hopping": conformal_geodesic_basin_hopping,
    "conformal_tps": conformal_geodesic_tps,
    "conformal_neb": conformal_geodesic_neb,
}


def compute_geodesic(
    metric: str,
    p_start_W: Tensor,
    p_end_W: Tensor,
    alphas: Tensor,
    sigma: float | None = None,
    sinkhorn_reg: float = 0.1,
    **kwargs,
) -> Tensor:
    """Single-segment geodesic between two distributions."""
    fn = GEODESIC_FNS[metric]
    if metric == "hellinger":
        return fn(p_start_W, p_end_W, alphas)
    if metric == "sinkhorn_cyclic":
        return fn(p_start_W, p_end_W, alphas, sigma=sigma, reg=sinkhorn_reg)
    if metric in (
        "conformal",
        "conformal_basin_hopping",
        "conformal_tps",
        "conformal_neb",
    ):
        return fn(p_start_W, p_end_W, alphas, **kwargs)  # device passed via kwargs
    return fn(p_start_W, p_end_W, alphas, sigma=sigma)


def compute_chained_geodesic(
    metric: str,
    all_var_probs_WW1: Tensor,
    base_idx: int,
    increment: int,
    alphas: Tensor,
    sigma: float | None = None,
    cyclic: bool = True,
) -> Tensor:
    """Chained geodesic through consecutive values.

    Uses all W values as supports, chaining geodesics between consecutive
    values. When ``cyclic=True``, position wraps mod W.

    Args:
        all_var_probs_WW1: (W, D) per-class centroid distributions where
            D is W+1 (with 'other' bin) or W (concept-only).
    """
    W = all_var_probs_WW1.shape[0]
    D = all_var_probs_WW1.shape[1]
    result = torch.zeros(len(alphas), D)

    for i, a in enumerate(alphas):
        pos = base_idx + a.item() * increment
        if cyclic:
            pos = pos % W
            val_lo = int(pos) % W
            val_hi = (val_lo + 1) % W
        else:
            pos = max(0.0, min(pos, W - 1))
            val_lo = min(int(pos), W - 2)
            val_hi = val_lo + 1
        t = pos - int(pos)
        p_lo = all_var_probs_WW1[val_lo]
        p_hi = all_var_probs_WW1[val_hi]
        t_tensor = torch.tensor([t])
        result[i] = compute_geodesic(metric, p_lo, p_hi, t_tensor, sigma=sigma)[0]

    return result
