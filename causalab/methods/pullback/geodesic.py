"""Centroid belief computation and manifold-trace path construction."""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor

from causalab.io.artifacts import load_tensor_results

logger = logging.getLogger(__name__)


def load_natural_distributions(
    experiment_root: str,
    W: int,
    class_assignments: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Load natural distributions and derive per-class centroids.

    Args:
        experiment_root: Experiment root containing the output_manifold dir.
        W: Number of concept classes.
        class_assignments: (M,) int tensor mapping each row of natural_dists to
            its TRUE class index. Required for the centroid computation; pass a
            row-aligned array of ``task.intervention_value_index(ex)`` values.

    Returns:
        natural_dists: (M, W+1) raw natural distributions
        all_class_dists_WW1: (W, W+1) per-class centroid distributions (with 'other' bin)
    """
    bm_dir = os.path.join(experiment_root, "output_manifold")
    nat_path = os.path.join(bm_dir, "per_example_output_dists.safetensors")
    if not os.path.exists(nat_path):
        raise FileNotFoundError(
            f"per_example_output_dists.safetensors not found at {nat_path}. "
            "Run output_manifold first to generate it."
        )
    natural_dists = load_tensor_results(bm_dir, "per_example_output_dists.safetensors")[
        "dists"
    ]
    logger.info("Loaded natural distributions: %s", natural_dists.shape)

    if class_assignments is None:
        raise ValueError(
            "class_assignments is required: pass an (M,) tensor of true class "
            "indices row-aligned with natural_dists "
            "(e.g. [task.intervention_value_index(ex) for ex in train_dataset])."
        )
    class_assignments = torch.as_tensor(class_assignments).long()
    if class_assignments.shape[0] != natural_dists.shape[0]:
        raise ValueError(
            f"class_assignments length ({class_assignments.shape[0]}) does not "
            f"match natural_dists rows ({natural_dists.shape[0]})."
        )

    all_class_dists_WW1 = torch.zeros(W, W + 1)
    for c in range(W):
        mask = class_assignments == c
        if mask.any():
            all_class_dists_WW1[c] = natural_dists[mask].mean(dim=0)
            all_class_dists_WW1[c] /= all_class_dists_WW1[c].sum()

    return natural_dists, all_class_dists_WW1


def _manifold_trace_path(
    ci: int,
    cj: int,
    belief_manifold,
    t_values: Tensor,
) -> Tensor:
    """Trace the belief manifold between two class centroids.

    Endpoints are obtained by projecting each class centroid onto the
    manifold via Gauss-Newton (``encode_to_nearest_point``). At smoothness=0
    this is exact — the manifold passes through every centroid, so the
    projection collapses to the corresponding ``control_points`` row. At
    smoothness>0, this returns the closest manifold point to the (slightly
    off-manifold) centroid, which is what we want.

    Intermediate points come from linearly interpolating u between the two
    projected endpoints (periodic shortest-arc) and decoding. The whole path
    lies on the manifold by construction; the simplex view is obtained at
    the boundary via ``hellinger_to_simplex``.

    Args:
        ci: Start class index.
        cj: End class index.
        belief_manifold: Pre-loaded SplineManifold in Hellinger space.
        t_values: (A,) query positions in [0, 1].

    Returns:
        (A, W+1) distributions along the manifold trace.
    """
    from causalab.methods.spline.belief_fit import hellinger_to_simplex

    dev = belief_manifold.centroids.device
    dtype = belief_manifold.centroids.dtype

    h_start = belief_manifold.centroids[ci].to(device=dev, dtype=dtype).unsqueeze(0)
    h_end = belief_manifold.centroids[cj].to(device=dev, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        u_start, _ = belief_manifold.encode_to_nearest_point(h_start)
        u_end, _ = belief_manifold.encode_to_nearest_point(h_end)
    u_start = u_start[0]
    u_end = u_end[0]

    delta = u_end - u_start
    if belief_manifold.periodic_dims:
        for pd, per in zip(belief_manifold.periodic_dims, belief_manifold.periods):
            if abs(float(delta[pd])) > per / 2:
                delta[pd] = delta[pd] - torch.sign(delta[pd]) * per

    t = t_values.to(device=dev, dtype=dtype)
    u_path = u_start.unsqueeze(0) + t.unsqueeze(1) * delta.unsqueeze(0)
    if belief_manifold.periodic_dims:
        for pd, per in zip(belief_manifold.periodic_dims, belief_manifold.periods):
            u_path[:, pd] = u_path[:, pd] % per

    with torch.no_grad():
        h_path = belief_manifold.decode(u_path)
    return hellinger_to_simplex(h_path)


def compute_manifold_trace_paths(
    selected_pair_indices: list[tuple[int, int]],
    belief_manifold,
    n_steps: int,
) -> tuple[dict[tuple[int, int], Tensor], Tensor]:
    """Build a manifold-trace belief path for each pair.

    The path is the manifold's geodesic in u-space: linearly interpolate
    between the two classes' control-point coords (with periodic shortest-arc
    wrap), decode to Hellinger, project to simplex.

    Returns:
        paths: dict (ci, cj) -> (n_steps, W+1) distributions on the manifold.
        t_values: (n_steps,) linspace from 0 to 1.
    """
    t_values = torch.linspace(0, 1, n_steps)
    paths = {
        (ci, cj): _manifold_trace_path(ci, cj, belief_manifold, t_values)
        for ci, cj in selected_pair_indices
    }
    logger.info(
        "Computed manifold-trace paths for %d pairs (n_steps=%d)",
        len(paths),
        n_steps,
    )
    return paths, t_values
