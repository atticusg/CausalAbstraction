"""Phase-1 reproduction gate.

Projects the phase-1 dataset onto the subspace and runs an adaptively-
selected metric set against the user-supplied significance description.
Pass → the analysis continues to webtext + judge. Fail → the analysis
writes a structured ``more_info_needed.json`` and stops.

Adaptive selection follows the populated fields of :class:`Significance`:

- ``hypothesis_text`` present → variance + projection-margin metrics.
- ``topology_description`` present → adds an intrinsic-dimensionality
  estimate (TwoNN). Connected-component / persistence metrics are
  deferred to a follow-up; the gate logs which metrics were skipped.
- ``figure_path`` present → noted in the diagnostic but no figure metric
  runs in this first cut; the figure side-car schema is open in the plan.
- None populated → variance metric only, gate threshold relaxed; the
  reconcile step's LLM judge will act as tiebreaker downstream.

This module deliberately does *not* import :class:`Significance` fields
into :class:`Step1Summary`. ``Step1Summary`` is constructed from
projections and example spans only — the type-level half of the judge-
independence invariant.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch
from torch import Tensor

from causalab.analyses.characterize_subspace.schemas import (
    ProjectionStats,
    Significance,
    Span,
    Step1Summary,
)

logger = logging.getLogger(__name__)

_THRESHOLD_RECHECK_WARNED = False


def _warn_threshold_recheck_once() -> None:
    """Emit the max-token threshold caveat at most once per process."""
    global _THRESHOLD_RECHECK_WARNED
    if _THRESHOLD_RECHECK_WARNED:
        return
    _THRESHOLD_RECHECK_WARNED = True
    logger.warning(
        "Reproduction gate now runs on each document's peak-norm token (issue "
        "#210; the token with the largest Euclidean subspace norm, BOS "
        "excluded) — variance/margin on its leading-axis (dim-0) component, "
        "intrinsic-dim on the full k-dim vector. Thresholds were tuned for "
        "mean-pooled distributions and may need recalibration. See "
        "TODO(threshold-recheck) in _default_thresholds."
    )


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    threshold: float
    passed: bool
    notes: str = ""


@dataclass(frozen=True)
class ReproductionReport:
    """Outcome of the gate. ``passed`` controls downstream branching."""

    passed: bool
    metrics: list[MetricResult]
    step1_summary: Step1Summary
    skipped_metrics: list[str] = field(default_factory=list)
    diagnostic: str | None = None


MetricFn = Callable[[Tensor, dict[str, float]], MetricResult]


# --------------------------------------------------------------------------- #
# Individual metrics                                                          #
# --------------------------------------------------------------------------- #


def _variance_metric(
    projections_1d: Tensor, thresholds: dict[str, float]
) -> MetricResult:
    """Variance of phase-1 peak-token subspace-activation norms.

    Non-degenerate variance is necessary for any further claim about the
    subspace. Threshold is small (``variance_min``, default 0.01) — failure
    here means the rotation matrix probably doesn't apply to this dataset.
    """
    threshold = thresholds.get("variance_min", 0.01)
    if projections_1d.numel() < 2:
        return MetricResult(
            name="variance",
            value=0.0,
            threshold=threshold,
            passed=False,
            notes="Need at least 2 phase-1 samples.",
        )
    value = float(projections_1d.var(unbiased=True).item())
    return MetricResult(
        name="variance",
        value=value,
        threshold=threshold,
        passed=value >= threshold,
    )


def _projection_margin_metric(
    projections_1d: Tensor, thresholds: dict[str, float]
) -> MetricResult:
    """Standardised gap between the top- and bottom-quantile means.

    Computed as ``(mean(top_q) - mean(bot_q)) / std`` for top/bottom 25% of the
    subspace-activation norm. A meaningful subspace should pull phase-1 inputs
    apart in activation strength — otherwise the rotation is uninformative for
    this dataset.
    """
    threshold = thresholds.get("margin_min", 0.5)
    n = projections_1d.numel()
    if n < 8:
        return MetricResult(
            name="projection_margin",
            value=0.0,
            threshold=threshold,
            passed=False,
            notes=f"Need at least 8 samples (have {n}).",
        )
    sorted_vals, _ = torch.sort(projections_1d)
    q = max(1, n // 4)
    bot_mean = sorted_vals[:q].mean()
    top_mean = sorted_vals[-q:].mean()
    std = projections_1d.std(unbiased=True).clamp(min=1e-8)
    value = float(((top_mean - bot_mean) / std).item())
    return MetricResult(
        name="projection_margin",
        value=value,
        threshold=threshold,
        passed=value >= threshold,
    )


def _twonn_intrinsic_dim(points: Tensor, thresholds: dict[str, float]) -> MetricResult:
    """TwoNN intrinsic-dimension estimate.

    Implementation of Facco et al. (2017): ``d ≈ N / sum(log(r_2 / r_1))``
    where ``r_1`` and ``r_2`` are the first- and second-nearest-neighbour
    distances. Threshold is ``intrinsic_dim_min`` (default 1.0).
    """
    threshold = thresholds.get("intrinsic_dim_min", 1.0)
    n = points.shape[0]
    if n < 4 or points.ndim != 2 or points.shape[1] < 2:
        return MetricResult(
            name="intrinsic_dim",
            value=0.0,
            threshold=threshold,
            passed=False,
            notes="Need at least 4 samples in at least 2 dimensions.",
        )
    # Pairwise distances; ignore self-distance (column 0 after sort).
    diffs = points.unsqueeze(0) - points.unsqueeze(1)
    dists = diffs.norm(dim=-1)
    dists.fill_diagonal_(float("inf"))
    sorted_dists, _ = torch.sort(dists, dim=1)
    r1 = sorted_dists[:, 0].clamp(min=1e-12)
    r2 = sorted_dists[:, 1].clamp(min=1e-12)
    mu = r2 / r1
    valid = mu > 1.0
    if int(valid.sum().item()) < 3:
        return MetricResult(
            name="intrinsic_dim",
            value=0.0,
            threshold=threshold,
            passed=False,
            notes="Too many duplicate points for TwoNN.",
        )
    logs = torch.log(mu[valid])
    value = float(valid.sum().item() / logs.sum().clamp(min=1e-12).item())
    if not math.isfinite(value):
        value = 0.0
    return MetricResult(
        name="intrinsic_dim",
        value=value,
        threshold=threshold,
        passed=value >= threshold,
    )


# --------------------------------------------------------------------------- #
# Adaptive selection + gate                                                   #
# --------------------------------------------------------------------------- #


def select_metric_names(significance: Significance) -> tuple[list[str], list[str]]:
    """Return ``(selected, skipped)`` metric-name lists for a given Significance.

    ``selected`` are metrics that run. ``skipped`` are metrics conceptually
    appropriate to the populated fields but not implemented in this version;
    they appear in the diagnostic and in ``more_info_needed.json`` so the
    caller knows what's missing.
    """
    selected: list[str] = ["variance", "projection_margin"]
    skipped: list[str] = []

    if significance.topology_description:
        selected.append("intrinsic_dim")
        skipped.append("connected_component_count (not yet implemented)")

    if significance.figure_path:
        skipped.append(
            "procrustes_residual (needs sibling <figure>.points.json side-car; "
            "schema is open — see plan file)"
        )
        skipped.append("kde_js_divergence (not yet implemented)")

    return selected, skipped


_METRIC_REGISTRY: dict[str, MetricFn] = {
    "variance": _variance_metric,
    "projection_margin": _projection_margin_metric,
    # intrinsic_dim takes the full point cloud, not just dim-0; handled
    # specially in run_gate.
}


def _default_thresholds(significance: Significance) -> dict[str, float]:
    """Return thresholds, relaxed when no significance description is supplied.

    .. warning::
       These defaults were tuned against **mean-pooled** phase-1
       distributions. The gate now runs on each document's **peak-norm token**
       (issue #210; the token with the largest Euclidean subspace norm, BOS
       excluded), whose spread differs from the mean-pooled distribution. The
       values below are intentionally left unchanged here so the representation
       change is observable in isolation; recalibration is a separate follow-up.
    """
    if significance.non_empty_values():
        return {
            # TODO(threshold-recheck): retune for max-token distributions.
            "variance_min": 0.01,
            # TODO(threshold-recheck): retune for max-token distributions.
            "margin_min": 0.5,
            # TODO(threshold-recheck): retune for max-token distributions.
            "intrinsic_dim_min": 1.0,
        }
    # description-absent: only check non-degeneracy.
    return {
        # TODO(threshold-recheck): retune for max-token distributions.
        "variance_min": 0.001,
        "margin_min": 0.0,
        "intrinsic_dim_min": 0.0,
    }


def run_gate(
    *,
    projections: Tensor,
    example_spans: list[Span],
    significance: Significance,
    dataset_name: str,
    pass_threshold: float = 0.5,
    thresholds: dict[str, float] | None = None,
) -> ReproductionReport:
    """Run the adaptive metric set and decide whether to continue.

    Variance / projection-margin and the ``Step1Summary`` stats run on the
    per-document subspace-activation norm ``‖peak_kdim‖₂`` — the same
    non-negative scalar the spans and the histogram carry — so the judge never
    sees two different scalars. ``intrinsic_dim`` still uses the full k-dim
    point cloud.

    Args:
        projections: ``(n, k)`` phase-1 peak-token projections; the per-row
            Euclidean norm is the 1-D quantity scored.
        example_spans: small sample of phase-1 spans to include in the
            ``Step1Summary`` payload sent to the judge later.
        significance: only used to pick which metrics to run and which
            thresholds to apply — it is **not** stored in ``Step1Summary``.
        dataset_name: provenance label for the bundle.
        pass_threshold: fraction of metrics that must pass for the gate to
            pass overall.
        thresholds: per-metric thresholds; defaults pulled from
            :func:`_default_thresholds`.
    """
    if projections.ndim != 2:
        raise ValueError(
            f"Expected (n, k) projections; got shape {tuple(projections.shape)}."
        )

    _warn_threshold_recheck_once()

    selected, skipped = select_metric_names(significance)
    thresh = thresholds or _default_thresholds(significance)

    # Score the subspace-activation norm (non-negative), matching the span /
    # histogram scalar — not the signed dim-0 coordinate.
    proj_1d = projections.norm(dim=1)
    results: list[MetricResult] = []
    for name in selected:
        if name == "intrinsic_dim":
            results.append(_twonn_intrinsic_dim(projections, thresh))
        else:
            results.append(_METRIC_REGISTRY[name](proj_1d, thresh))

    n_pass = sum(1 for r in results if r.passed)
    passed = (n_pass / max(1, len(results))) >= pass_threshold

    n = proj_1d.numel()
    summary = Step1Summary(
        dataset_name=dataset_name,
        stats=ProjectionStats(
            n_samples=int(n),
            mean=float(proj_1d.mean().item()) if n else 0.0,
            std=float(proj_1d.std(unbiased=True).item()) if n > 1 else 0.0,
            min=float(proj_1d.min().item()) if n else 0.0,
            max=float(proj_1d.max().item()) if n else 0.0,
        ),
        example_spans=example_spans,
    )

    diagnostic = None
    if not passed:
        failing = [r for r in results if not r.passed]
        bits = [f"{r.name}={r.value:.4g} (<{r.threshold:.4g})" for r in failing]
        diagnostic = "Reproduction gate failed: " + "; ".join(bits)
        if skipped:
            diagnostic += f". Skipped metrics: {skipped}."

    return ReproductionReport(
        passed=passed,
        metrics=results,
        step1_summary=summary,
        skipped_metrics=skipped,
        diagnostic=diagnostic,
    )


# --------------------------------------------------------------------------- #
# Structured failure artifact                                                 #
# --------------------------------------------------------------------------- #


def write_more_info_needed(
    out_dir: str,
    *,
    report: ReproductionReport,
    significance: Significance,
) -> str:
    """Write ``more_info_needed.json`` for a failed gate.

    Schema:

    ```json
    {
      "status": "insufficient_handoff",
      "failed_metrics": [{"name", "value", "threshold", "axis": "variance|margin|topology"}],
      "missing_inputs": [{"field", "why"}],
      "actionable_requests": [str, ...]
    }
    ```
    """
    failed = [r for r in report.metrics if not r.passed]
    axis_map = {
        "variance": "variance",
        "projection_margin": "margin",
        "intrinsic_dim": "topology",
    }
    failed_metrics = [
        {
            "name": r.name,
            "value": r.value,
            "threshold": r.threshold,
            "axis": axis_map.get(r.name, "unknown"),
            "notes": r.notes,
        }
        for r in failed
    ]
    missing_inputs: list[dict[str, str]] = []
    actionable: list[str] = []
    if significance.figure_path and not significance.hypothesis_text:
        missing_inputs.append(
            {
                "field": "hypothesis_text",
                "why": (
                    "A natural-language hypothesis would let the variance / "
                    "projection-margin gate run against a clearer criterion."
                ),
            }
        )
    if not significance.non_empty_values():
        actionable.append(
            "Provide a hypothesis_text, figure_path, or topology_description "
            "so the gate can run against a concrete criterion."
        )
    if any(r.name == "variance" and not r.passed for r in report.metrics):
        actionable.append(
            "Phase-1 projections have near-zero variance. Confirm the "
            "subspace was actually fit on this model/layer/site combination."
        )
    if any(r.name == "projection_margin" and not r.passed for r in report.metrics):
        actionable.append(
            "Phase-1 projections show no separation along the subspace's "
            "leading direction. Re-examine the rotation orientation or the "
            "choice of layer/site."
        )

    payload = {
        "status": "insufficient_handoff",
        "failed_metrics": failed_metrics,
        "missing_inputs": missing_inputs,
        "actionable_requests": actionable,
        "diagnostic": report.diagnostic,
        "skipped_metrics": report.skipped_metrics,
        "step1_summary": asdict(report.step1_summary),
    }
    out_path = os.path.join(out_dir, "more_info_needed.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote insufficient-handoff artifact to %s", out_path)
    return out_path
