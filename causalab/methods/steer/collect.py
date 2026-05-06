"""Manifold steering infrastructure — distribution collection and grid generation.

Functions for steering a model along manifold coordinates and collecting
the resulting output distributions.  Used by distortion scores (coherence,
conformal, distance_from_behavior_manifold) and by the manifold fitting
pipeline's reconstruction test.

Includes:
  - collect_all_variable_distributions (featurizer-independent variant collection)
  - collect_grid_distributions (fixed grid points, averaged over samples)
  - _scores_to_joint_probs (internal helper)
  - make_intrinsic_steering_grid, _make_1d_coords (grid generation)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from causalab.methods.metric import (
    _normalize_var_indices,
    class_probabilities,
    scores_to_joint_probs as _scores_to_joint_probs,
)
from causalab.neural.activations.intervenable_model import device_for_layer


# ---------------------------------------------------------------------------
# Grid generation (moved from steer_manifold.py)
# ---------------------------------------------------------------------------


def _make_1d_coords(lo: float, hi: float, n: int) -> Tensor:
    """Make 1D coordinates, using log spacing when lo > 0 and the range spans
    more than one order of magnitude."""
    if lo > 0 and hi / lo > 10:
        return torch.logspace(math.log10(lo), math.log10(hi), n)
    return torch.linspace(lo, hi, n)


def make_intrinsic_steering_grid(
    intrinsic_dim: int,
    n_points_per_dim: int = 11,
    range_min: float = -3.0,
    range_max: float = 3.0,
    ranges: tuple[tuple[float, float], ...] | None = None,
) -> Tensor:
    """
    Generate steering grid in intrinsic manifold space.

    Uses log spacing for dimensions where min > 0 and max/min > 10
    (i.e. the range spans more than one order of magnitude).

    Args:
        intrinsic_dim: Dimensionality d of manifold intrinsic space.
        n_points_per_dim: Points per dimension (for d<=2).
        range_min, range_max: Global range for all coordinates (default: [-3, 3]).
                             Ignored if `ranges` is provided.
        ranges: Per-dimension ranges as ((min0, max0), (min1, max1), ...).
               If provided, overrides range_min/range_max and allows different
               ranges per intrinsic dimension.

    Returns:
        Grid tensor of shape:
        - d=1: (n_points_per_dim, 1)
        - d=2: (n_points_per_dim^2, 2) from meshgrid
        - d>2: (n^d, d) full meshgrid, n = min(n_points_per_dim, 512^(1/d))
    """
    # Build per-dimension ranges
    if ranges is not None:
        if len(ranges) != intrinsic_dim:
            raise ValueError(
                f"ranges has {len(ranges)} entries but intrinsic_dim is {intrinsic_dim}"
            )
        dim_ranges = ranges
    else:
        # Use global range for all dimensions (backwards compatibility)
        dim_ranges = tuple((range_min, range_max) for _ in range(intrinsic_dim))

    if intrinsic_dim == 1:
        coords = _make_1d_coords(dim_ranges[0][0], dim_ranges[0][1], n_points_per_dim)
        return coords.unsqueeze(-1)  # (n, 1)

    elif intrinsic_dim == 2:
        # Full meshgrid with per-dimension ranges
        coords0 = _make_1d_coords(dim_ranges[0][0], dim_ranges[0][1], n_points_per_dim)
        coords1 = _make_1d_coords(dim_ranges[1][0], dim_ranges[1][1], n_points_per_dim)
        u1, u2 = torch.meshgrid(coords0, coords1, indexing="ij")
        return torch.stack([u1.flatten(), u2.flatten()], dim=-1)  # (n^2, 2)

    else:
        # Full meshgrid with coarser resolution to keep point count manageable
        max_per_dim = max(2, int(round(512 ** (1.0 / intrinsic_dim))))
        n = min(n_points_per_dim, max_per_dim)
        per_dim = [
            _make_1d_coords(dim_ranges[d][0], dim_ranges[d][1], n)
            for d in range(intrinsic_dim)
        ]
        grids = torch.meshgrid(*per_dim, indexing="ij")
        return torch.stack([g.flatten() for g in grids], dim=-1)  # (n^d, d)


# ---------------------------------------------------------------------------
# Distribution collection
# ---------------------------------------------------------------------------


def collect_all_variable_distributions(
    pipeline,
    filtered_samples: list[dict],
    steered_variable: str,
    steered_values: list[str],
    var_indices: torch.Tensor | list[list[int]],
    batch_size: int = 32,
) -> torch.Tensor:
    """Collect output distributions for all value-variants of each sample.

    Featurizer-independent — uses raw model generation (no interventions).

    For each sample, substitutes every ``steered_value`` into the raw input
    template and runs the model to collect output logits, then computes joint
    probabilities.

    Returns:
        ``(N, W_variants, W_cats)`` joint probability tensor.
    """
    var_token_seqs = _normalize_var_indices(var_indices)
    W_cats = len(var_token_seqs)
    W_variants = len(steered_values)

    all_joint = []  # list of (W_variants, W_cats) per sample

    for sample in filtered_samples:
        base_input = sample["input"]
        base_val = base_input[steered_variable]
        base_raw = base_input["raw_input"]

        # Build W variants of this sample
        base_dict = (
            base_input.to_dict() if hasattr(base_input, "to_dict") else dict(base_input)
        )
        variants = []
        for val in steered_values:
            v = dict(base_dict)
            v[steered_variable] = val
            v["raw_input"] = base_raw.replace(base_val, val)
            variants.append(v)

        # Collect logits for all variants in batches
        step_batches: list[list[torch.Tensor]] = []
        for start in range(0, len(variants), batch_size):
            batch = variants[start : start + batch_size]
            out = pipeline.generate(batch, output_scores=True)
            for k, scores_k in enumerate(out["scores"]):
                if k >= len(step_batches):
                    step_batches.append([])
                step_batches[k].append(scores_k)

        # Concatenate batches per step → list of (W_variants, V) tensors
        per_step = [torch.cat(batches, dim=0) for batches in step_batches]

        # Compute joint probabilities (W_variants, W_cats)
        joint_WW = torch.ones(W_variants, W_cats)
        for k, logits_WV in enumerate(per_step):
            active = [
                (w, seq[k]) for w, seq in enumerate(var_token_seqs) if k < len(seq)
            ]
            if active:
                step_ids = [t for _, t in active]
                probs = class_probabilities(logits_WV, step_ids)
                w_idx = torch.tensor([w for w, _ in active])
                joint_WW[:, w_idx] *= probs.cpu()

        joint_WW = joint_WW / joint_WW.sum(dim=-1, keepdim=True)
        all_joint.append(joint_WW)

    return torch.stack(all_joint, dim=0)  # (N, W_variants, W_cats)


def collect_grid_distributions(
    pipeline,
    grid_points: torch.Tensor,
    interchange_target,
    filtered_samples: list[dict],
    var_indices: torch.Tensor | list[list[int]],
    batch_size: int = 32,
    n_base_samples: int = 5,
    average: bool = True,
    full_vocab_softmax: bool = False,
) -> torch.Tensor:
    """Collect output distributions by steering the model at each grid point.

    For each grid point, passes intrinsic coordinates directly to the
    intervention pipeline (which handles manifold decoding internally via
    inverse_featurizer), runs a forward pass, and extracts variable-token
    probabilities. Averages over a few base samples for robustness.

    Args:
        grid_points: (G, d) intrinsic coordinates — passed directly to
            replace_fn, NOT decoded here. The intervention pipeline's
            FeatureInterpolateIntervention already applies inverse_featurizer
            which includes manifold decode + un-standardization.
        average: If True (default), return (G, W_cats) averaged over samples.
            If False, return (G, N, W_cats) per-sample distributions.
        full_vocab_softmax: If True, softmax over full vocabulary before
            extracting class token probabilities. Probabilities won't sum to 1.

    Returns:
        (G, W_cats) if average=True, (G, N, W_cats) if average=False.
    """
    from causalab.neural.activations.interpolate import (
        run_interpolation_interventions,
    )

    base_samples = filtered_samples[:n_base_samples]
    if not base_samples:
        raise ValueError("No base samples available for grid distribution collection")

    # Create dummy counterfactual examples (self-pairs)
    dummy_cf = [
        {"input": s["input"], "counterfactual_inputs": [s["input"]]}
        if "counterfactual_inputs" not in s
        else s
        for s in base_samples
    ]

    # Tokenize for joint_probs computation
    if isinstance(var_indices, torch.Tensor):
        var_token_seqs = [[idx.item()] for idx in var_indices]
    else:
        var_token_seqs = var_indices
    W_cats = len(var_token_seqs)
    N = len(base_samples)

    all_distributions = []
    for g_idx in range(grid_points.shape[0]):
        intrinsic_point = grid_points[g_idx]  # (d,)

        def replace_fn(
            f_base: torch.Tensor,
            f_src: torch.Tensor,
            target: torch.Tensor = intrinsic_point,
            **_kwargs,
        ) -> torch.Tensor:
            target = target.to(f_base.device)
            B, k_full = f_base.shape[0], f_base.shape[-1]
            k_t = target.shape[-1]
            if k_t < k_full:
                # Partial-dim target: hold dropped dims at each sample's base values.
                opt = target.unsqueeze(0).expand(B, -1)
                return torch.cat([opt, f_base[:, k_t:]], dim=-1)
            return target.unsqueeze(0).expand(B, -1)

        results = run_interpolation_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dummy_cf,
            interchange_target=interchange_target,
            fn=replace_fn,
            params={},
            batch_size=batch_size,
            output_scores=True,
        )

        # Restore featurizer modules to the GPU of the layer they hook into.
        # delete_intervenable_model moves shared sub-modules to CPU as a side-effect;
        # for sharded models we route each unit to its own layer's device.
        for group in interchange_target:
            for unit in group:
                unit_device = device_for_layer(pipeline, unit.layer)
                unit.featurizer.featurizer.to(unit_device)
                unit.featurizer.inverse_featurizer.to(unit_device)

        joint_NW = _scores_to_joint_probs(
            results["scores"], var_indices, full_vocab_softmax=full_vocab_softmax
        )
        if joint_NW is None:
            if average:
                all_distributions.append(torch.ones(W_cats) / W_cats)
            else:
                all_distributions.append(torch.ones(N, W_cats) / W_cats)
            continue

        if average:
            all_distributions.append(joint_NW.mean(dim=0))  # (W_cats,)
        else:
            all_distributions.append(joint_NW)  # (N, W_cats)

    return torch.stack(all_distributions, dim=0)  # (G, W_cats) or (G, N, W_cats)


# ---------------------------------------------------------------------------
# Steering vector construction (moved from steer_manifold.py)
# ---------------------------------------------------------------------------


def construct_steering_vectors(
    steering_grid: Tensor,
    intrinsic_dim: int,
    total_dim: int,
) -> Tensor:
    """
    Construct k-dim steering vectors from d-dim intrinsic coordinates.

    z = [u, 0...0] where u is the intrinsic coordinate and residual is zeroed.

    Args:
        steering_grid: Intrinsic coordinates (n_points, d)
        intrinsic_dim: Dimensionality d of intrinsic space
        total_dim: Total latent dimension k (featurizer.n_features)

    Returns:
        Steering vectors of shape (n_points, k)
    """
    n_points = steering_grid.shape[0]
    residual_dim = total_dim - intrinsic_dim

    # Concatenate intrinsic coords with zeros
    residual = torch.zeros(n_points, residual_dim, dtype=steering_grid.dtype)
    return torch.cat([steering_grid, residual], dim=-1)  # (n_points, k)
