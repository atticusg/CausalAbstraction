"""Distance from Behavior Manifold — cumulative Bhattacharyya distance from
each observed output distribution along a steering path to a reference on the
belief manifold.

Per step, we compute the Bhattacharyya distance ``D_B = -log(BC)``, where
``BC = sum_i sqrt(p_i * q_i)`` is the Bhattacharyya coefficient between the
observed distribution and the reference. ``D_B`` is the canonical
``-log p``-flavoured cousin of Hellinger (related by ``D_B = -log(1 − d_H²)``)
and stays inside the Hellinger geometry the manifold was fit in. Cumulative
``D_B`` along the path is interpretable as integrated log-density "energy"
between the path and the reference — matching the energy-based naturalness
framing in §3.4 of the writeup.

Two reference choices:

  - manifold  — distance to the *nearest* point on the continuous manifold
                (1-NN over an infinite reference set).
  - geodesic  — distance to the *matched-fraction* point on the geodesic
                between the two endpoint centroids (single-point reference at
                fraction t along the centroid-pair geodesic). Only computed
                for centroid pairs (where (i, j) class indices are known).

Both variants **sum** ``D_B`` along the path per (pair, prompt), then average
across prompts to produce one per-pair scalar. Per-pair tensors are saved
separately so paired t-tests fire across modes for each independently.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from torch import Tensor

from causalab.io.artifacts import save_tensors_with_meta
from causalab.methods.distances import hellinger_distance_to_manifold

logger = logging.getLogger(__name__)

SUPPORTS_PATH_MODES = True

NAN_RESULT: dict[str, float] = {
    "mean": float("nan"),
    "se": float("nan"),
    "geodesic_mean": float("nan"),
    "geodesic_se": float("nan"),
}

_BC_EPS = 1e-7


def _hellinger_to_bhattacharyya(d_h: Tensor) -> Tensor:
    """Bhattacharyya distance from Hellinger: ``D_B = -log(1 - d_H²)``."""
    return -torch.log((1.0 - d_h.pow(2)).clamp(min=_BC_EPS))


def compute_score_single_path(
    distributions: Tensor,
    belief_manifold: Any,
) -> float:
    """Mean Bhattacharyya distance from a path to the manifold (legacy single-pair API)."""
    if distributions.shape[0] < 1:
        return float("nan")
    distributions = _pad_to_simplex(distributions, belief_manifold.centroids.shape[-1])
    with torch.no_grad():
        d_h = hellinger_distance_to_manifold(distributions, belief_manifold)
    return float(_hellinger_to_bhattacharyya(d_h).mean())


def compute_score(
    pair_distributions: Tensor,
    belief_manifold: Any = None,
    path_belief_endpoints: Tensor | None = None,
    output_dir: str | None = None,
    path_mode_label: str | None = None,
    tol: float = 1e-4,
    **kwargs: Any,
) -> dict[str, float]:
    """Cumulative Bhattacharyya distance from intervened paths to two references.

    Args:
        pair_distributions: ``(n_pairs, num_steps, n_prompts, W)`` tensor of
            *concept-slice* probabilities collected with
            ``full_vocab_softmax=True``. The "other" bin is appended internally
            so the (W+1)-dim padded vector is a valid probability distribution
            on the simplex; it is then sqrt-projected onto the unit Hellinger
            sphere before computing distances.
        belief_manifold: SplineManifold fit in Hellinger space (sphere_project=True).
        path_belief_endpoints: Optional ``(n_pairs, 2, intrinsic_dim_belief)``
            tensor giving each path's two endpoints in belief-manifold
            u-space. When given, the geodesic variant is computed for ALL
            n_pairs paths by building the belief geodesic between each pair's
            endpoints. Centroid-pair endpoints are class-centroid u-coords
            (``belief_manifold.control_points[i]``); extras endpoints are the
            belief-space images of the activation-side interior vertices
            (linear interp at fraction f along (i, j) in belief u-space).
            ``None`` skips the geodesic variant.
        output_dir, path_mode_label: artifact destination.
        tol: Tolerance for numerical violations of the sum<=1 invariant.

    Returns dict with keys: mean, se (manifold), geodesic_mean, geodesic_se.
    ``se`` is the standard error of the mean across pairs (std / √n_pairs).
    """
    if belief_manifold is None:
        logger.warning("No belief manifold provided — returning NaN")
        return dict(NAN_RESULT)

    n_pairs, num_steps, n_prompts, W = pair_distributions.shape
    W_amb = belief_manifold.centroids.shape[-1]

    # Validate that pair_distributions is a sub-probability distribution
    # (slice of a full-vocab softmax). The "other"-bin padding below relies on
    # sum-over-W ≤ 1; otherwise the padded simplex would not sum to 1 and the
    # subsequent Hellinger / Bhattacharyya computation would be on something
    # that isn't a valid probability distribution.
    on = pair_distributions.sum(dim=-1)
    assert (on <= 1.0 + tol).all(), (
        f"pair_distributions sum exceeds 1 (max={float(on.max()):.6f}); "
        "must come from full_vocab_softmax over the concept slice."
    )

    # Pad to W+1 simplex with the "other" bin so dim matches manifold ambient.
    # After this, p_obs is a valid probability distribution (sums to 1) — the
    # subsequent sqrt + sphere-normalize in hellinger_distance_to_manifold then
    # places it on the unit Hellinger sphere.
    p_obs = _pad_to_simplex(
        pair_distributions, W_amb
    )  # (n_pairs, num_steps, n_prompts, W+1)

    # ── Variant 1: distance to nearest point on the continuous manifold ──
    flat = p_obs.reshape(-1, W_amb)
    with torch.no_grad():
        d_h_manifold_flat = hellinger_distance_to_manifold(flat, belief_manifold)
    d_h_manifold = d_h_manifold_flat.reshape(n_pairs, num_steps, n_prompts).cpu()
    d_b_manifold = _hellinger_to_bhattacharyya(d_h_manifold)

    # ── Variant 2: distance to matched-fraction point on the geodesic between
    #             each path's two belief-space endpoints (centroid pairs and
    #             extras both, given the per-pair endpoints) ──
    d_b_geodesic = None
    if path_belief_endpoints is not None:
        n_with_endpoints = path_belief_endpoints.shape[0]
        d_h_geodesic = _hellinger_to_geodesic_at_t(
            p_obs[:n_with_endpoints],
            belief_manifold,
            path_belief_endpoints,
            num_steps,
        ).cpu()  # (n_with_endpoints, num_steps, n_prompts)
        d_b_geodesic = _hellinger_to_bhattacharyya(d_h_geodesic)

    # Sum along path per (pair, prompt) — cumulative log-energy along the
    # path. Then mean across prompts → one scalar per pair.
    def _reduce(d: Tensor) -> Tensor:
        # d: (n_pairs, num_steps, n_prompts)
        return d.sum(dim=1).mean(dim=1)

    per_pair_manifold = _reduce(d_b_manifold)
    per_pair_geodesic = _reduce(d_b_geodesic) if d_b_geodesic is not None else None

    if per_pair_manifold.numel() == 0:
        return dict(NAN_RESULT)

    out = dict(NAN_RESULT)
    _fill(out, "", per_pair_manifold)
    if per_pair_geodesic is not None:
        _fill(out, "geodesic_", per_pair_geodesic)

    logger.info(
        "  distance_from_behavior_manifold (cumulative Bhattacharyya along path)  "
        "manifold=%.4f±%.4f  geodesic=%.4f±%.4f  (±SE)",
        out["mean"],
        out["se"],
        out["geodesic_mean"],
        out["geodesic_se"],
    )

    if output_dir is not None:
        _save_artifacts(
            out,
            per_pair_manifold=per_pair_manifold,
            per_pair_geodesic=per_pair_geodesic,
            output_dir=output_dir,
            path_mode_label=path_mode_label,
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_simplex(p: Tensor, target_dim: int) -> Tensor:
    """Append the `1 − sum` "other" bin so p lives on the (target_dim − 1)-simplex."""
    if p.shape[-1] >= target_dim:
        return p
    other = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0)
    return torch.cat([p, other], dim=-1)


def _hellinger_to_geodesic_at_t(
    p_obs: Tensor,
    belief_manifold: Any,
    endpoints: Tensor,
    num_steps: int,
) -> Tensor:
    """Hellinger distance from each (pair, step, prompt) to the matched-fraction
    point on the belief-manifold geodesic between each path's two endpoints.

    Args:
        p_obs: (n_pairs, num_steps, n_prompts, W+1).
        belief_manifold: SplineManifold (sphere_project=True).
        endpoints: ``(n_pairs, 2, intrinsic_dim_belief)`` tensor of per-pair
            (start, end) endpoints in belief u-space. For centroid pairs the
            endpoints are class centroids; for extras they are the belief-space
            images of the activation-side interior vertices.
        num_steps: number of points along each path.

    Returns:
        (n_pairs, num_steps, n_prompts) Hellinger distances in [0, 1].
        Caller transforms to Bhattacharyya via ``_hellinger_to_bhattacharyya``.
    """
    from causalab.analyses.path_steering.path_mode import _build_geodesic_path

    device = belief_manifold.centroids.device
    endpoints = endpoints.to(device)

    # Per-pair belief geodesic at num_steps fractions, decoded to sqrt-space
    # (already on the unit sphere because sphere_project=True).
    h_refs = []
    with torch.no_grad():
        for k in range(endpoints.shape[0]):
            u_path = _build_geodesic_path(
                endpoints[k, 0],
                endpoints[k, 1],
                num_steps,
                belief_manifold,
            )
            h_path = belief_manifold.decode(u_path.to(device))  # (num_steps, W+1)
            h_refs.append(h_path)
    h_ref = torch.stack(h_refs)  # (n_pairs, num_steps, W+1)

    # h_obs on sphere.
    h_obs = p_obs.to(device=device).clamp(min=0).sqrt()
    h_obs = h_obs / h_obs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Broadcast h_ref over n_prompts.
    h_ref_b = h_ref.unsqueeze(2)  # (n_pairs, num_steps, 1, W+1)
    return (h_obs - h_ref_b).norm(dim=-1) / (2.0**0.5)


def _fill(out: dict[str, float], prefix: str, per_pair: Tensor) -> None:
    """Write {prefix}mean / {prefix}se into ``out`` (skipping NaNs).

    ``se`` is the standard error of the mean = std / √n.
    """
    valid = per_pair[~torch.isnan(per_pair)]
    n = valid.numel()
    if n == 0:
        return
    out[f"{prefix}mean"] = float(valid.mean())
    out[f"{prefix}se"] = (
        float(valid.std(unbiased=True) / (n**0.5)) if n > 1 else float("nan")
    )


def _save_artifacts(
    metrics: dict[str, float],
    *,
    per_pair_manifold: Tensor,
    per_pair_geodesic: Tensor | None,
    output_dir: str,
    path_mode_label: str | None,
) -> None:
    parts = ["distance_from_behavior_manifold"]
    if path_mode_label is not None:
        parts.append(path_mode_label)
    artifact_dir = os.path.join(output_dir, *parts)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_tensors_with_meta(
        {"value": per_pair_manifold}, {}, artifact_dir, "per_pair_scores"
    )
    if per_pair_geodesic is not None:
        save_tensors_with_meta(
            {"value": per_pair_geodesic},
            {},
            artifact_dir,
            "per_pair_scores_geodesic",
        )
    logger.info("Saved distance_from_behavior_manifold artifacts to %s", artifact_dir)
