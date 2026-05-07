"""Visualization functions for the pullback analysis.

Two main visualizations:
1. Belief space: optimized paths in output space (line plots + Hellinger PCA 3D)
2. Activation space: optimized paths overlaid on TPS manifold (3D interactive)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Belief-space visualization
# ---------------------------------------------------------------------------


def plot_belief_paths(
    results_by_kopt: dict[str, dict[tuple[int, int], dict]]
    | dict[tuple[int, int], dict],
    concept_names: list[str],
    output_dir: str,
    colormap: str = "rainbow",
    figure_format: str = "pdf",
) -> None:
    """Plot belief trajectories for each path mode using the same plotting
    pipeline as the evaluate analysis (with error bars when per-sample data exists).

    ``results_by_kopt`` is the new nested form ``{k_opt_label: {pair: result_dict}}``;
    a flat ``{pair: result_dict}`` is also accepted (auto-lifted under label
    ``"full"``). Each k_opt label produces its own ``optimized_<label>/`` subdir;
    target/geometric/linear paths are k_opt-independent and rendered once.
    """
    from causalab.analyses.path_steering.path_visualization import (
        plot_saved_pair_distributions,
    )

    os.makedirs(output_dir, exist_ok=True)

    results_by_kopt = _ensure_nested_results(results_by_kopt)
    if not results_by_kopt:
        return

    # Comparison/target paths are identical across k_opt labels — render from
    # the first label that has data.
    first_label, first_results = next(iter(results_by_kopt.items()))

    shared_specs = [
        ("p_target_AW1", "target_path", False),  # (n_steps, W+1), no samples
        ("geo_probs_raw", "geometric", True),  # (n_steps, n_samples, W)
        ("lin_probs_raw", "linear", True),  # (n_steps, n_samples, W)
    ]
    per_kopt_specs = [
        ("opt_probs_raw", f"optimized_{label}", True, label)
        for label in results_by_kopt
    ]

    def _emit(
        spec_key: str, spec_label: str, has_samples: bool, label_results: dict
    ) -> None:
        pairs = list(label_results.keys())
        available = [p for p in pairs if spec_key in label_results[p]]
        eff_key = spec_key
        if not available:
            avg_key = spec_key.replace("_raw", "_AW1")
            available = [p for p in pairs if avg_key in label_results[p]]
            if not available:
                return
            eff_key = avg_key
            has_samples = False

        pair_dists_list = []
        for p in pairs:
            if eff_key not in label_results[p]:
                continue
            probs = label_results[p][eff_key]
            if probs.dim() == 2:
                W = len(concept_names)
                probs = probs[:, :W].unsqueeze(1)
            pair_dists_list.append(probs)

        if not pair_dists_list:
            return

        pair_distributions = torch.stack(pair_dists_list)
        plot_saved_pair_distributions(
            pair_distributions=pair_distributions,
            pairs=available,
            value_labels=concept_names,
            output_dir=os.path.join(output_dir, spec_label),
            path_mode_label=spec_label,
            colormap=colormap,
            full_vocab_softmax=True,
            figure_format=figure_format,
        )

    for key, label, has_samples in shared_specs:
        _emit(key, label, has_samples, first_results)

    for key, sub_label, has_samples, k_label in per_kopt_specs:
        _emit(key, sub_label, has_samples, results_by_kopt[k_label])

    logger.info("Saved belief path plots to %s", output_dir)


# Teal-family palette for overlaying multiple k_opt optimized trajectories on
# the same 3D plot. Cycles if the sweep is longer than the palette.
_OPTIMIZED_COLOR_PALETTE: list[str] = [
    "#019689",  # original optimized teal — keeps single-k_opt look identical
    "#04C2B0",
    "#015F58",
    "#02A697",
    "#7BD3CB",
]
# Backwards-compatible alias for any external import.
_OPTIMIZED_COLOR = _OPTIMIZED_COLOR_PALETTE[0]


def _optimized_color(idx: int) -> str:
    return _OPTIMIZED_COLOR_PALETTE[idx % len(_OPTIMIZED_COLOR_PALETTE)]


def _ensure_nested_results(
    results: dict[str, dict[tuple[int, int], dict]]
    | dict[tuple[int, int], dict]
    | None,
) -> dict[str, dict[tuple[int, int], dict]]:
    """Normalize ``results`` to nested ``{label: {pair: data}}``.

    Accepts the legacy flat form ``{pair: data}`` and lifts it under
    ``"full"``. Returns ``{}`` for None/empty input.
    """
    if not results:
        return {}
    sample_key = next(iter(results))
    if isinstance(sample_key, tuple):
        return {"full": dict(results)}  # type: ignore[arg-type]
    return {str(label): dict(per) for label, per in results.items()}


def plot_belief_paths_hellinger_pca(
    results_by_kopt: dict[str, dict[tuple[int, int], dict]]
    | dict[tuple[int, int], dict],
    natural_dists: Tensor,
    concept_names: list[str],
    output_dir: str,
    colormap: str | None = None,
    intervention_variable: str | None = None,
    belief_manifold: Any = None,
    train_dataset: list = None,  # required: row-aligned with natural_dists, supplies TRUE class labels
    belief_paths: dict[tuple[int, int], Tensor] | None = None,
) -> None:
    """Overlay optimized, geometric, and linear paths in Hellinger PCA space.

    ``results_by_kopt`` is the nested form ``{k_opt_label: {pair: result}}``;
    a flat ``{pair: result}`` is also accepted (auto-lifted under ``"full"``).
    One optimized trace per k_opt label is overlaid using teal-shaded colors
    from ``_OPTIMIZED_COLOR_PALETTE``; geometric/linear/belief-target traces
    are k_opt-independent and rendered once per pair from the first label
    that has them. The manifold mesh is drawn by ``plot_3d`` when
    ``belief_manifold`` is provided.

    When ``belief_paths`` is provided, the fitted belief-target trace is
    rendered in red. ``results_by_kopt`` may be empty in that case (e.g. when
    only the geodesic cache is populated).
    """
    from causalab.io.plots.plot_3d_interactive import plot_3d, PathTrace

    os.makedirs(output_dir, exist_ok=True)

    results_by_kopt = _ensure_nested_results(results_by_kopt)

    pair_set: set[tuple[int, int]] = set()
    for per in results_by_kopt.values():
        pair_set.update(per.keys())
    if pair_set:
        pairs = sorted(pair_set)
    elif belief_paths:
        pairs = list(belief_paths.keys())
    else:
        logger.warning("plot_belief_paths_hellinger_pca: nothing to plot")
        return

    # Convert probabilities to Hellinger (sqrt-probability) space
    def _to_hellinger(probs: Tensor) -> Tensor:
        """Convert (*, W) or (*, W+1) distributions to Hellinger coordinates."""
        p = probs.double()
        W1 = natural_dists.shape[-1]
        if p.shape[-1] < W1:
            other = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0)
            p = torch.cat([p, other], dim=-1)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return torch.sqrt(p.clamp(min=0)).float()

    # Background: natural outputs as per-example probabilities (passed to
    # plot_3d with feature_kind='hellinger'; the √ + drift-free centroid
    # math happens inside plot_3d).
    nat = natural_dists.double()
    nat = nat / nat.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    raw_nat = nat.float()
    n_classes = len(concept_names)
    iv = intervention_variable or "class"
    if train_dataset is None:
        raise ValueError(
            "train_dataset is required: row-aligned with natural_dists, used to "
            "derive TRUE class labels."
        )
    if len(train_dataset) < nat.shape[0]:
        raise ValueError(
            f"train_dataset length ({len(train_dataset)}) is shorter than "
            f"natural_dists rows ({nat.shape[0]})."
        )
    label_to_idx = {str(v): i for i, v in enumerate(concept_names)}

    def _iv_val(ex):
        inp = ex["input"]
        return inp[iv] if isinstance(inp, dict) else inp._values.get(iv)

    classes = np.array(
        [
            label_to_idx.get(str(_iv_val(ex)), -1)
            for ex in train_dataset[: nat.shape[0]]
        ],
        dtype=float,
    )
    if (classes < 0).any():
        bad = {
            str(_iv_val(ex))
            for ex in train_dataset[: nat.shape[0]]
            if label_to_idx.get(str(_iv_val(ex)), -1) < 0
        }
        raise ValueError(
            f"train_dataset has IV={iv} values not present in concept_names: "
            f"{sorted(bad)}. Either pass the right concept_names or fix the dataset."
        )
    param_dict = {iv: classes}

    # Manifold ranges for mesh rendering: use data-driven quantile range
    # (encode sqrt_nat through manifold, take 99.7% quantiles) instead of
    # control-points + 5% margin. The TPS spline extrapolates wildly outside
    # its control-point domain, producing visible "spikes" past the data
    # extent (especially for non-cyclic 1D manifolds like age, results 2-100).
    from causalab.analyses.activation_manifold.utils import (
        _compute_intrinsic_ranges,
    )

    ranges = None
    if belief_manifold is not None:
        sqrt_nat = torch.sqrt(raw_nat.clamp(min=0)).float()
        try:
            ranges = _compute_intrinsic_ranges(
                sqrt_nat,
                belief_manifold,
                torch.zeros(raw_nat.shape[1]),
                torch.ones(raw_nat.shape[1]),
            )
        except Exception:
            pass

    # Identity standardization (belief manifold operates in Hellinger space directly)
    D = raw_nat.shape[1]
    mean = torch.zeros(D)
    std = torch.ones(D)

    label_order = list(results_by_kopt.keys())
    multi_kopt = len(label_order) > 1

    for ci, cj in pairs:
        pair_label = f"{concept_names[ci]}_{concept_names[cj]}"
        path_traces = []

        # Fitted belief target — one trace per pair.
        if belief_paths is not None and (ci, cj) in belief_paths:
            path_traces.append(
                PathTrace(
                    points=_to_hellinger(belief_paths[(ci, cj)]),
                    name="belief target",
                    color="#D62728",
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )

        # Linear / geometric comparisons are k_opt-independent — pull from the
        # first label that has them.
        shared_res: dict = {}
        for label in label_order:
            cand = results_by_kopt[label].get((ci, cj), {})
            if "lin_probs_AW1" in cand or "geo_probs_AW1" in cand:
                shared_res = cand
                break

        if "lin_probs_AW1" in shared_res:
            path_traces.append(
                PathTrace(
                    points=_to_hellinger(shared_res["lin_probs_AW1"]),
                    name="linear",
                    color="darkgray",
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )
        if "geo_probs_AW1" in shared_res:
            path_traces.append(
                PathTrace(
                    points=_to_hellinger(shared_res["geo_probs_AW1"]),
                    name="geometric",
                    color="#000000",
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )

        # One optimized trace per k_opt label (teal-shaded palette cycle).
        for idx, label in enumerate(label_order):
            res = results_by_kopt[label].get((ci, cj), {})
            if "opt_probs_AW1" not in res:
                continue
            trace_name = f"optimized ({label})" if multi_kopt else "optimized"
            path_traces.append(
                PathTrace(
                    points=_to_hellinger(res["opt_probs_AW1"]),
                    name=trace_name,
                    color=_optimized_color(idx),
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )

        plot_3d(
            features=raw_nat,
            output_path=os.path.join(output_dir, f"belief_space_{pair_label}.html"),
            param_dict=param_dict,
            intervention_variable=iv,
            variable_values=concept_names,
            colormap=colormap,
            manifold_obj=belief_manifold,
            mean=mean,
            std=std,
            ranges=ranges,
            paths=path_traces,
            centroid_color="black",
            feature_kind="hellinger",
        )

    logger.info("Saved %d Hellinger PCA plots to %s", len(pairs), output_dir)


# ---------------------------------------------------------------------------
# 2. Activation-space visualization
# ---------------------------------------------------------------------------


def plot_activation_paths(
    results_by_kopt: dict[str, dict[tuple[int, int], dict]]
    | dict[tuple[int, int], dict],
    training_features: Tensor,
    concept_names: list[str],
    output_dir: str,
    manifold_obj: Any = None,
    manifold_mean: Tensor | None = None,
    manifold_std: Tensor | None = None,
    colormap: str | None = None,
    intervention_variable: str | None = None,
    param_dict: dict[str, np.ndarray] | None = None,
    variable_values: list[str] | None = None,
) -> None:
    """Plot optimized embedding paths in activation space.

    Linear/geometric traces are rendered once per pair; one optimized trace per
    k_opt label is overlaid using the teal-shaded palette cycle.
    """
    from causalab.io.plots.plot_3d_interactive import plot_3d, PathTrace

    os.makedirs(output_dir, exist_ok=True)

    results_by_kopt = _ensure_nested_results(results_by_kopt)
    if not results_by_kopt:
        return

    label_order = list(results_by_kopt.keys())
    multi_kopt = len(label_order) > 1

    pair_set: set[tuple[int, int]] = set()
    for per in results_by_kopt.values():
        pair_set.update(per.keys())
    pairs = sorted(pair_set)

    D = training_features.shape[1]
    mean = manifold_mean if manifold_mean is not None else torch.zeros(D)
    std = manifold_std if manifold_std is not None else torch.ones(D)

    # Manifold mesh ranges: use the data-driven 99.7% quantile range (encoded
    # through the manifold) instead of control-points + 5% margin. The TPS
    # spline extrapolates wildly outside its training extent, producing
    # spikey-end artefacts in the mesh — visible especially for non-cyclic
    # variables like age (entities 1-99).
    ranges = None
    if manifold_obj is not None:
        from causalab.analyses.activation_manifold.utils import (
            _compute_intrinsic_ranges,
        )

        try:
            ranges = _compute_intrinsic_ranges(
                training_features,
                manifold_obj,
                mean,
                std,
            )
        except Exception:
            pass

    for ci, cj in pairs:
        pair_label = f"{concept_names[ci]}_{concept_names[cj]}"
        path_traces = []

        # Comparison paths are k_opt-independent — pull from first available label.
        shared_res: dict = {}
        for label in label_order:
            cand = results_by_kopt[label].get((ci, cj), {})
            if "v_linear_k" in cand or "v_geometric_k" in cand:
                shared_res = cand
                break

        if "v_linear_k" in shared_res:
            path_traces.append(
                PathTrace(
                    points=shared_res["v_linear_k"],
                    name="linear",
                    color="darkgray",
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )
        if "v_geometric_k" in shared_res:
            path_traces.append(
                PathTrace(
                    points=shared_res["v_geometric_k"],
                    name="geometric",
                    color="#000000",
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )

        for idx, label in enumerate(label_order):
            res = results_by_kopt[label].get((ci, cj), {})
            if "v_init_k" in res:
                init_name = f"v_init ({label})" if multi_kopt else "v_init"
                path_traces.append(
                    PathTrace(
                        points=res["v_init_k"].float().detach().cpu(),
                        name=init_name,
                        color="#D62728",
                        width=3,
                        marker_size=4,
                        is_intrinsic=False,
                        mode="lines+markers",
                    )
                )
            if "v_optimized_k" not in res:
                continue
            trace_name = f"optimized ({label})" if multi_kopt else "optimized"
            path_traces.append(
                PathTrace(
                    points=res["v_optimized_k"].float().detach().cpu(),
                    name=trace_name,
                    color=_optimized_color(idx),
                    width=3,
                    marker_size=4,
                    is_intrinsic=False,
                    mode="lines+markers",
                )
            )

        plot_3d(
            features=training_features,
            output_path=os.path.join(output_dir, f"activation_{pair_label}.html"),
            param_dict=param_dict or {},
            intervention_variable=intervention_variable or "class",
            variable_values=variable_values or concept_names,
            colormap=colormap,
            manifold_obj=manifold_obj,
            mean=mean,
            std=std,
            ranges=ranges,
            paths=path_traces,
            centroid_color="black",
        )

    logger.info("Saved %d activation-space plots to %s", len(pairs), output_dir)
