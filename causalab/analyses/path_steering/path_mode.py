"""Path modes: how a steered path is built between two centroid endpoints.

A :class:`PathMode` carries the centroid space (``intrinsic`` / ``raw`` /
``pca``), the path kind (``geodesic`` / ``linear``), and any featurizer
override applied during steering. Callers dispatch through ``pm.build_path``
and ``pm.select_centroids`` rather than branching on ``pm.label``.

The low-level path primitives live here too so each mode owns the geometry
it prescribes. Import them directly only when you genuinely need both kinds
independent of any mode (e.g. visualization overlays, pullback comparison
passes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from causalab.neural.featurizer import ComposedFeaturizer, Featurizer

logger = logging.getLogger(__name__)


CentroidSpace = Literal["intrinsic", "raw", "pca"]
PathKind = Literal["geodesic", "linear"]


# ---------------------------------------------------------------------------
# Path-construction primitives
# ---------------------------------------------------------------------------


def _get_periodic_periods(manifold_obj: Any) -> dict[int, float]:
    """Per-dim periods from a manifold object (empty dict if none)."""
    periodic_dims = getattr(manifold_obj, "periodic_dims", None)
    periods = manifold_obj.periods if hasattr(manifold_obj, "periods") else None
    if not periodic_dims or not periods:
        return {}
    return dict(zip(periodic_dims, periods))


def _build_path_alphas(
    n_steps: int,
    oversteer_frac: float,
    oversteer_steps: int,
    device: torch.device,
) -> Tensor:
    """Alpha values: n_steps from [0,1], then oversteer_steps from (1, 1+frac]."""
    normal = torch.linspace(0, 1, n_steps, device=device)
    if oversteer_frac > 0 and oversteer_steps > 0:
        step_size = oversteer_frac / oversteer_steps
        overshoot = torch.linspace(
            1.0 + step_size,
            1.0 + oversteer_frac,
            oversteer_steps,
            device=device,
        )
        return torch.cat([normal, overshoot])
    return normal


def _build_geodesic_path(
    start_intrinsic: Tensor,
    end_intrinsic: Tensor,
    n_steps: int,
    manifold_obj: Any,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> Tensor:
    """Geodesic path in intrinsic space (linear interp, periodic shortest-arc per dim).

    When oversteer_frac > 0, oversteer_steps coarse points continue past the target,
    wrapping around periodic dimensions naturally.
    """
    alphas = _build_path_alphas(
        n_steps,
        oversteer_frac,
        oversteer_steps,
        start_intrinsic.device,
    )
    d = start_intrinsic.shape[0]
    periodic_periods = _get_periodic_periods(manifold_obj)

    # Clone before in-place wrap so autograd stays safe if inputs ever require grad.
    delta = (end_intrinsic - start_intrinsic).clone()
    for dim, period in periodic_periods.items():
        if dim < d:
            delta[dim] = ((delta[dim] + period / 2.0) % period) - period / 2.0

    return start_intrinsic.unsqueeze(0) + alphas.unsqueeze(1) * delta.unsqueeze(0)


def _build_linear_path_kd(
    start_features: Tensor,
    end_features: Tensor,
    n_steps: int,
    oversteer_frac: float = 0.0,
    oversteer_steps: int = 0,
) -> Tensor:
    """Straight-line path in feature space (raw activations or PCA k-dim).

    When oversteer_frac > 0, oversteer_steps coarse points extrapolate beyond
    the target, moving off-manifold and surfacing out-of-distribution mass.
    """
    alphas = _build_path_alphas(
        n_steps,
        oversteer_frac,
        oversteer_steps,
        start_features.device,
    )
    return start_features.unsqueeze(0) + alphas.unsqueeze(1) * (
        end_features - start_features
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# PathMode
# ---------------------------------------------------------------------------


@dataclass
class PathMode:
    """Resolved path mode: centroid space, path kind, and optional featurizer override."""

    label: str
    centroid_space: CentroidSpace
    path_kind: PathKind
    featurizer_override: Featurizer | None = None

    def select_centroids(
        self,
        *,
        intrinsic: Tensor,
        raw: Tensor,
        pca: Tensor,
    ) -> Tensor:
        """Return the centroid set this mode operates on."""
        return {"intrinsic": intrinsic, "raw": raw, "pca": pca}[self.centroid_space]

    def build_path(
        self,
        start: Tensor,
        end: Tensor,
        n_steps: int,
        *,
        manifold_obj: Any | None = None,
        oversteer_frac: float = 0.0,
        oversteer_steps: int = 0,
    ) -> Tensor:
        """Build the (n_steps [+ oversteer_steps], d) path between two endpoints.

        When ``path_kind="geodesic"`` but no manifold is available, falls back
        to a straight line — preserves the historical fallback behaviour.
        """
        if self.path_kind == "geodesic" and manifold_obj is not None:
            return _build_geodesic_path(
                start,
                end,
                n_steps,
                manifold_obj,
                oversteer_frac=oversteer_frac,
                oversteer_steps=oversteer_steps,
            )
        return _build_linear_path_kd(
            start,
            end,
            n_steps,
            oversteer_frac=oversteer_frac,
            oversteer_steps=oversteer_steps,
        )


def resolve_path_modes(
    path_modes_cfg: list,
    composed_featurizer: ComposedFeaturizer | None = None,
) -> list[PathMode]:
    """Resolve each path_mode config entry into a :class:`PathMode`.

    Supported modes:
    - ``"geometric"`` — geodesic path in intrinsic (u-space) coordinates.
    - ``"linear"`` — straight-line path in raw activation space (identity featurizer).
    - ``"linear_subspace"`` — straight-line path in PCA-reduced subspace.
    """
    results: list[PathMode] = []

    for mode in path_modes_cfg:
        if not isinstance(mode, str):
            raise ValueError(f"Invalid path_mode entry: {mode!r}")

        if mode == "geometric":
            results.append(
                PathMode(
                    label="geometric",
                    centroid_space="intrinsic",
                    path_kind="geodesic",
                )
            )
        elif mode == "linear":
            results.append(
                PathMode(
                    label="linear",
                    centroid_space="raw",
                    path_kind="linear",
                    featurizer_override=Featurizer(id="identity"),
                )
            )
        elif mode == "linear_subspace":
            pca_override = None
            if isinstance(composed_featurizer, ComposedFeaturizer):
                pca_override = composed_featurizer.stages[0]
            results.append(
                PathMode(
                    label="linear_subspace",
                    centroid_space="pca",
                    path_kind="linear",
                    featurizer_override=pca_override,
                )
            )
        else:
            raise ValueError(f"Unknown path_mode: {mode!r}")

    return results
