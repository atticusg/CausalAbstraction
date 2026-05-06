"""Builder functions for constructing spline manifolds from data."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import torch
from torch import Tensor

from causalab.causal.counterfactual_dataset import CounterfactualExample
from .manifold import SplineManifold

logger = logging.getLogger(__name__)

EXCLUDED_VARS = {
    "probs",
    "sequence",
    "raw_input",
    "raw_output",
    "true_probs",
    "observations",
    "context_length",
}


def extract_parameters_from_dataset(
    dataset: list[CounterfactualExample],
    excluded_vars: set[str] | None = None,
    embeddings: dict[str, Callable[[Any], list[float]]] | None = None,
    causal_model: Any | None = None,
) -> dict[str, Tensor]:
    """Extract parameter values from input traces of a counterfactual dataset.

    Iterates over examples and extracts causal parameter values from the input
    trace only (not counterfactual traces). This gives one value per example,
    aligned with features from collect_features().

    When an embedding function is provided for a variable, it is used to map
    the value to one or more floats (e.g. cyclic day -> [cos, sin]).
    Multi-dimensional embeddings produce keys like ``var_0``, ``var_1``, etc.
    When no embedding is provided, the value is converted via ``float()``.

    Tuple-valued parameters (without an embedding) are expanded into separate
    dimensions (mu_0, mu_1).

    Args:
        dataset: List of CounterfactualExample dicts.
        excluded_vars: Set of variable names to skip. Uses EXCLUDED_VARS if None.
        embeddings: Optional dict mapping variable names to embedding functions.
            Each function takes a variable value and returns a list of floats.
        causal_model: Optional CausalModel. If provided and ``embeddings`` is
            None, embeddings are read from ``causal_model.embeddings``.

    Returns:
        Dict mapping parameter names to tensors of shape (n_examples,).
    """
    if excluded_vars is None:
        excluded_vars = EXCLUDED_VARS
    if embeddings is None:
        embeddings = getattr(causal_model, "embeddings", None) or {}

    param_values: dict[str, list[float]] = defaultdict(list)
    tuple_params: dict[str, int] = {}

    def _extract_from_trace(trace):
        for var in trace._values:
            if var in excluded_vars:
                continue
            val = trace[var]
            if val is None:
                continue
            if var in embeddings:
                coords = embeddings[var](val)
                if len(coords) == 1:
                    param_values[var].append(coords[0])
                else:
                    for j, c in enumerate(coords):
                        param_values[f"{var}_{j}"].append(c)
            elif isinstance(val, (tuple, list)):
                if var not in tuple_params:
                    tuple_params[var] = len(val)
                for j, v in enumerate(val):
                    param_values[f"{var}_{j}"].append(float(v))
            else:
                param_values[var].append(float(val))

    for ex in dataset:
        _extract_from_trace(ex["input"])

    return {k: torch.tensor(v) for k, v in param_values.items()}


def compute_centroids(
    features: Tensor,
    param_tensors: dict[str, Tensor],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Group features by unique parameter combinations and compute mean centroids.

    **Ordering warning**: The returned centroids are sorted by torch.unique's
    lexicographic order on the parameter combinations, which may NOT match the
    task's class index order. For example, 2D grid coordinates (angle, height)
    get sorted by (angle, height) while class indices may enumerate by
    (height, angle). Do not assume centroid[i] corresponds to class i.
    Use manifold.encode(class_ordered_centroids) to get intrinsic coordinates
    in class order.

    Args:
        features: Feature tensor (n_samples, ambient_dim).
        param_tensors: Dict mapping parameter names to tensors of shape (n_samples,).

    Returns:
        control_points: Unique parameter combinations (n_centroids, n_params).
        centroids: Mean features per group (n_centroids, ambient_dim).
        metadata: Dict with parameter_names and counts.
    """
    n = features.shape[0]
    param_names = sorted(param_tensors.keys())
    param_matrix = torch.stack([param_tensors[name] for name in param_names], dim=1)

    # Find unique parameter combinations
    unique_params, inverse_indices = torch.unique(
        param_matrix, dim=0, return_inverse=True
    )
    n_centroids = unique_params.shape[0]

    # Compute mean features per group
    centroids = torch.zeros(n_centroids, features.shape[1], dtype=features.dtype)
    counts = torch.zeros(n_centroids, dtype=torch.long)
    for i in range(n):
        idx = inverse_indices[i]
        centroids[idx] += features[i]
        counts[idx] += 1

    centroids = centroids / counts.unsqueeze(1).float()

    metadata = {
        "parameter_names": param_names,
        "n_centroids": n_centroids,
        "counts": counts.tolist(),
    }

    logger.info(
        f"Computed {n_centroids} centroids from {n} samples (params: {param_names})"
    )

    return unique_params, centroids, metadata


def build_spline_manifold(
    control_points: Tensor,
    centroids: Tensor,
    intrinsic_dim: int | None = None,
    ambient_dim: int | None = None,
    smoothness: float = 0.0,
    device: str | torch.device = "cpu",
    periodic_dims: tuple[bool, ...] | None = None,
    periods: list[float] | None = None,
    spline_method: str = "auto",
    sphere_project: bool = False,
) -> SplineManifold:
    """Build a SplineManifold from control points and centroids.

    Args:
        control_points: Parameter combinations (n_centroids, n_params).
        centroids: Mean features per group (n_centroids, ambient_dim).
        intrinsic_dim: Intrinsic dimension. Defaults to control_points.shape[1].
        ambient_dim: Ambient dimension. Defaults to centroids.shape[1].
        smoothness: Smoothness parameter (TPS regularizer or cubic Reinsch λ).
        device: Device to place the manifold on.
        periodic_dims: Which dimensions of control_points are periodic.
        periods: Period for each periodic dimension.
        spline_method: Backend selector. ``"auto"`` picks a natural cubic
            spline for 1D non-cyclic data and TPS otherwise. ``"tps"`` and
            ``"cubic"`` force the corresponding backend.
        sphere_project: When True, decode() projects the ambient spline
            value onto the unit L2 sphere. Use for the belief manifold
            (Hellinger space) so every decoded point is a valid sqrt(p).

    Returns:
        SplineManifold instance.
    """
    if intrinsic_dim is None:
        intrinsic_dim = control_points.shape[1]
    if ambient_dim is None:
        ambient_dim = centroids.shape[1]

    device = torch.device(device)
    control_points = control_points.to(device)
    centroids = centroids.to(device)

    manifold = SplineManifold(
        control_points=control_points,
        target_points=centroids,
        intrinsic_dim=intrinsic_dim,
        ambient_dim=ambient_dim,
        smoothness=smoothness,
        periodic_dims=periodic_dims,
        periods=periods,
        spline_method=spline_method,
        sphere_project=sphere_project,
    )

    return manifold


# ─────────────────────────────────────────────────────────────────────
# Periodic dimension detection
# ─────────────────────────────────────────────────────────────────────


def detect_periodic_dims(
    control_points: Tensor,
    eigenvalues: Tensor,
    eigenvalue_tol: float = 0.45,
    min_variance_fraction: float = 0.1,
) -> list[tuple[int, int]]:
    """Detect periodic dimension pairs from near-degenerate eigenvalues.

    A pair (i, j) is periodic if:
    1. Both eigenvalues are significant (each ≥ min_variance_fraction of total)
    2. Eigenvalues are near-degenerate: |λ_i - λ_j| / max(λ_i, λ_j) < eigenvalue_tol

    Near-degenerate eigenvalues signal a closed loop (circle/ellipse).

    Args:
        control_points: (n_centroids, n_components) coordinates.
        eigenvalues: (n_components,) variance per dimension.
        eigenvalue_tol: Max relative eigenvalue difference for pairing.
            0.5 allows aspect ratios up to 2:1 (elliptical loops).
        min_variance_fraction: Minimum fraction of total eigenvalue sum
            that each dimension must explain to be considered.

    Returns:
        List of (i, j) periodic dimension pairs.
    """
    n_comp = control_points.shape[1]
    total_var = eigenvalues.sum().item()
    min_eigenvalue = min_variance_fraction * total_var

    used = set()
    pairs = []

    for i in range(n_comp):
        if i in used:
            continue
        li = eigenvalues[i].item()
        if li < min_eigenvalue:
            continue
        for j in range(i + 1, n_comp):
            if j in used:
                continue
            lj = eigenvalues[j].item()
            if lj < min_eigenvalue:
                continue

            ratio = abs(li - lj) / max(li, lj)
            if ratio < eigenvalue_tol:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break  # Move to next i

    return pairs


def remap_periodic_to_angle(
    control_points: Tensor,
    periodic_pairs: list[tuple[int, int]],
    eigenvalues: Tensor | None = None,
) -> tuple[Tensor, list[int], list[float]]:
    """Collapse periodic dimension pairs into angular columns.

    Each periodic pair (i, j) -> one column θ with period 2π. When eigenvalues
    are provided, coordinates are normalized by sqrt(eigenvalue) before atan2,
    which maps an elliptical embedding back to a circle for uniform angular
    spacing (equivalent to arc-length parameterization).

    Non-periodic dimensions pass through unchanged.

    Args:
        control_points: (n_centroids, n_components) in ISOMAP coordinates.
        periodic_pairs: List of (i, j) pairs from detect_periodic_dims.
        eigenvalues: (n_components,) ISOMAP eigenvalues for normalization.

    Returns:
        new_points: (n_centroids, new_n_components) with collapsed dims.
        periodic_dim_indices: Indices of periodic columns in new_points.
        periods: Period for each periodic column (always 2π).
    """
    import math

    n = control_points.shape[0]
    n_comp = control_points.shape[1]

    paired_dims = set()
    for i, j in periodic_pairs:
        paired_dims.add(i)
        paired_dims.add(j)

    columns = []
    periodic_dim_indices = []
    periods = []

    # Add angular columns for each periodic pair
    for i, j in periodic_pairs:
        col_i = control_points[:, i] - control_points[:, i].mean()
        col_j = control_points[:, j] - control_points[:, j].mean()
        # Normalize by sqrt(eigenvalue) to map ellipse -> circle
        if eigenvalues is not None:
            col_i = col_i / eigenvalues[i].sqrt()
            col_j = col_j / eigenvalues[j].sqrt()
        angle = torch.atan2(col_j, col_i)
        periodic_dim_indices.append(len(columns))
        periods.append(2 * math.pi)
        columns.append(angle)

    # Add non-periodic columns
    for d in range(n_comp):
        if d not in paired_dims:
            columns.append(control_points[:, d])

    new_points = torch.stack(columns, dim=1)
    return new_points, periodic_dim_indices, periods
