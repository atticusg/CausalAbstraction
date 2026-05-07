"""Coherence — fraction of mass on concept tokens (i.e. on-target probability)
along a steered path.

Two reductions per (pair, prompt) over the path's K+1 steps; both higher-is-better
in [0, 1]:
  - mean   — average P(concept) across the path
  - worst  — min   P(concept) across the path  (worst-case slip on this prompt)

Each reduction is then averaged over prompts to produce one per-pair scalar.
The two per-pair tensors are saved separately so paired t-tests can fire across
modes for each independently.

Pipeline/featurizer lifecycle is managed by the orchestrator.
"""

from __future__ import annotations

import json
import logging
import os

from torch import Tensor

from causalab.io.artifacts import save_tensors_with_meta

logger = logging.getLogger(__name__)

SUPPORTS_PATH_MODES = True

NAN_RESULT: dict[str, float] = {
    "mean": float("nan"),
    "se": float("nan"),
    "worst_mean": float("nan"),
    "worst_se": float("nan"),
}


def _se(t: Tensor) -> float:
    """Standard error of the mean = std / √n. NaN for n < 2."""
    n = t.numel()
    if n < 2:
        return float("nan")
    return float(t.std(unbiased=True) / (n**0.5))


def compute_score_single_path(distributions: Tensor, tol: float = 1e-4) -> float:
    """Mean P(concept) for a single (num_steps, W) path.

    Args:
        distributions: ``(num_steps, W)`` from full_vocab_softmax over the
            concept slice; the deficit (1 − sum) is the off-target mass.
        tol: Tolerance for numerical violations of the sum<=1 invariant.
    """
    on = distributions.sum(dim=-1)
    assert (on <= 1.0 + tol).all(), (
        f"P(concept) exceeds 1 (max={float(on.max()):.6f}); distributions must "
        "come from full_vocab_softmax over the concept slice."
    )
    return float(on.clamp(max=1.0).mean())


def compute_score(
    pair_distributions: Tensor,
    output_dir: str | None = None,
    path_mode_label: str | None = None,
    tol: float = 1e-4,
    **kwargs,
) -> dict[str, float]:
    """Compute coherence with mean and worst reductions.

    Args:
        pair_distributions: ``(n_pairs, num_steps, n_prompts, W)`` collected with
            full_vocab_softmax=True. P(concept) at each (pair, step, prompt) is
            the sum over W; the deficit (1 − sum) is the off-target mass.
        output_dir: If provided, saves artifacts.
        path_mode_label: Subdirectory label for artifacts.
        tol: Tolerance for numerical violations of the sum<=1 invariant.

    Returns dict with keys: mean, se, worst_mean, worst_se. ``se`` is the
    standard error of the mean across pairs (std / √n_pairs).
    """
    on = pair_distributions.sum(dim=-1)  # (n_pairs, num_steps, n_prompts)
    assert (on <= 1.0 + tol).all(), (
        f"P(concept) exceeds 1 (max={float(on.max()):.6f}); distributions must "
        "come from full_vocab_softmax over the concept slice."
    )
    on = on.clamp(max=1.0)

    # Reduce along path dim per (pair, prompt) → (n_pairs, n_prompts).
    on_mean_pn = on.mean(dim=1)
    on_worst_pn = on.min(dim=1).values

    # Mean over prompts → (n_pairs,) per reduction.
    per_pair_mean = on_mean_pn.mean(dim=1)
    per_pair_worst = on_worst_pn.mean(dim=1)

    if per_pair_mean.numel() == 0:
        return NAN_RESULT

    out = {
        "mean": float(per_pair_mean.mean()),
        "se": _se(per_pair_mean),
        "worst_mean": float(per_pair_worst.mean()),
        "worst_se": _se(per_pair_worst),
    }
    logger.info(
        "  coherence  mean=%.4f±%.4f  worst=%.4f±%.4f  (±SE)",
        out["mean"],
        out["se"],
        out["worst_mean"],
        out["worst_se"],
    )

    if output_dir is not None:
        _save_artifacts(out, per_pair_mean, per_pair_worst, output_dir, path_mode_label)
    return out


def _save_artifacts(
    metrics: dict[str, float],
    per_pair_mean: Tensor,
    per_pair_worst: Tensor,
    output_dir: str,
    path_mode_label: str | None,
) -> None:
    parts = ["coherence"]
    if path_mode_label is not None:
        parts.append(path_mode_label)
    artifact_dir = os.path.join(output_dir, *parts)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_tensors_with_meta(
        {"value": per_pair_mean}, {}, artifact_dir, "per_pair_scores"
    )
    save_tensors_with_meta(
        {"value": per_pair_worst}, {}, artifact_dir, "per_pair_scores_worst"
    )
    logger.info("Saved coherence artifacts to %s", artifact_dir)
