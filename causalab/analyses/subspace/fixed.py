"""Fixed (given) subspace threading.

Threads a **precomputed** rotation — e.g. SAE decoder directions, or any
imported `(d_model, k)` basis — into the manifold pipeline, instead of fitting
one with PCA/DAS/DBM. The on-disk bundle it writes is identical to the PCA
producer's (see :mod:`causalab.analyses.subspace.artifacts`), so
``activation_manifold`` / ``output_manifold`` / ``path_steering`` auto-discover
and consume it with **no** session-local code.

The rotation is resolved from one of three inputs (``subspace.fixed.*``):

- ``artifact`` — a ready ``.safetensors`` with tensor key ``rotation_matrix``.
- ``source`` — either an SAE-cluster spec (``sae_checkpoint``/``clusters_path``/
  ``cluster_id``) or a block/Grassmannian-SAE block (``block_sae_checkpoint``/
  ``block_id``), built via ``characterize_subspace.subspace_builder``.
- ``feature_ids`` (+ ``sae_checkpoint``) — explicit SAE feature ids, stacked
  into a basis via :func:`causalab.methods.sae.decoder_subspace`.

Trap encoded here (re-discovered 9+ times in the 2026-06 sweep): collect **RAW**
features first (no featurizer), *then* project ``raw @ rotation``. Setting a
``SubspaceFeaturizer`` before collection returns already-projected ``(N, k)``
features, and projecting again raises a shape error.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence, cast

from torch import Tensor

from causalab.analyses.subspace.artifacts import save_subspace_artifacts
from causalab.analyses.subspace._visualization import save_features_visualization
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.specs import SiteSpec
from causalab.neural.pipeline import LMPipeline
from causalab.neural.activations.collect import collect_features

logger = logging.getLogger(__name__)


def resolve_fixed_rotation(
    fixed_cfg: Any,
    *,
    out_dir: str,
) -> tuple[Tensor, dict[str, Any]]:
    """Resolve a ``(d_model, k)`` rotation from a ``subspace.fixed`` config block.

    Exactly one input is consulted, in priority order: ``feature_ids`` (with
    ``sae_checkpoint``) → ``artifact`` / ``source``. Returns the rotation and a
    JSON-serializable provenance dict for the run metadata.
    """
    from omegaconf import DictConfig, OmegaConf

    if isinstance(fixed_cfg, DictConfig):
        fixed = cast(dict[str, Any], OmegaConf.to_container(fixed_cfg, resolve=True))
    else:
        fixed = dict(fixed_cfg or {})

    feature_ids = fixed.get("feature_ids")
    artifact = fixed.get("artifact")
    source = fixed.get("source")
    # Exactly one input selects the rotation. resolve_subspace_artifact already
    # guards artifact-vs-source, but feature_ids would otherwise silently win
    # over both — so check all three here.
    provided = [
        name
        for name, val in (
            ("feature_ids", feature_ids),
            ("artifact", artifact),
            ("source", source),
        )
        if val
    ]
    if len(provided) > 1:
        raise ValueError(
            "subspace.fixed: provide exactly one of feature_ids / artifact / "
            f"source; got {provided}."
        )
    if not provided:
        raise ValueError(
            "subspace.fixed: one of feature_ids / artifact / source is required "
            "(the given rotation to thread)."
        )

    if feature_ids:
        sae_checkpoint = fixed.get("sae_checkpoint") or (fixed.get("source") or {}).get(
            "sae_checkpoint"
        )
        if not sae_checkpoint:
            raise ValueError(
                "subspace.fixed.feature_ids requires subspace.fixed.sae_checkpoint "
                "(the SAE whose decoder directions the ids index)."
            )
        from causalab.io.sae_checkpoints import read_sae_decoder
        from causalab.methods.sae import decoder_subspace

        decoder, d_model_hint = read_sae_decoder(str(sae_checkpoint))
        rotation = decoder_subspace(
            decoder,
            [int(f) for f in feature_ids],
            d_model=d_model_hint,
            orthonormalize=bool(fixed.get("orthonormalize", True)),
        )
        provenance = {
            "feature_ids": [int(f) for f in feature_ids],
            "sae_checkpoint": str(sae_checkpoint),
            "orthonormalize": bool(fixed.get("orthonormalize", True)),
        }
        return rotation, provenance

    from causalab.analyses.characterize_subspace.subspace_builder import (
        resolve_subspace_artifact,
    )
    from safetensors.torch import load_file

    path, build_provenance = resolve_subspace_artifact(
        artifact=artifact,
        source=source,
        out_dir=out_dir,
    )
    rotation = load_file(path)["rotation_matrix"]
    provenance = build_provenance or {"artifact": str(path)}
    return rotation, provenance


def orient_rotation(
    rotation: Tensor, *, d_model: int, k_features_hint: int | None = None
) -> Tensor:
    """Return ``rotation`` in canonical ``(d_model, k)`` orientation.

    The fixed-subspace pipeline always knows ``d_model`` (from the loaded
    model), so *non-square* orientation is resolved deterministically by
    matching an axis against it — unlike ``characterize_subspace``'s adaptive
    loader (:func:`causalab.analyses.characterize_subspace.loading._orient`),
    which falls back to heuristics when ``d_model`` is unknown.

    A **square** ``(n, n)`` matrix is ambiguous even with ``d_model`` known — it
    could be ``(d_model, k)`` or its transpose — so, mirroring ``_orient``, an
    integer ``k_features_hint`` is required to confirm ``k`` (it must equal the
    matrix dimension); the matrix is then returned as-is.

    - ``(d_model, k)`` non-square: returned unchanged.
    - ``(k, d_model)`` non-square: transposed to ``(d_model, k)``.
    - square ``(n, n)``: requires ``k_features_hint == n == d_model``; returned
      unchanged.
    - neither axis equals ``d_model``: :class:`ValueError` naming both axes.

    Idempotent: a second call (with the same ``k_features_hint``) is a no-op, so
    callers may orient defensively without double-transposing.
    """
    if rotation.ndim != 2:
        raise ValueError(
            "Fixed rotation must be 2D (d_model, k) or (k, d_model); got shape "
            f"{tuple(rotation.shape)}."
        )
    a, b = int(rotation.shape[0]), int(rotation.shape[1])
    if a == b:
        # Square: orientation is ambiguous even though d_model is known (the
        # matrix could be (d_model, k) or its transpose). Mirror _orient and
        # require an explicit k_features_hint to confirm k.
        if a != d_model:
            raise ValueError(
                f"Fixed rotation is square {(a, b)} but matches the model's "
                f"d_model={d_model} on neither axis; cannot project."
            )
        if not isinstance(k_features_hint, int):
            raise ValueError(
                f"Fixed rotation is square ({a}x{b}); orientation is ambiguous "
                "(could be (d_model, k) or its transpose). Pass k_features_hint "
                "(the value of k) to disambiguate."
            )
        if k_features_hint != a:
            raise ValueError(
                f"k_features_hint={k_features_hint} does not match the square "
                f"rotation's axis ({a})."
            )
        return rotation
    if a == d_model:
        return rotation
    if b == d_model:
        return rotation.t().contiguous()
    raise ValueError(
        f"Fixed rotation shape {(a, b)} matches the model's d_model={d_model} on "
        "neither axis; cannot determine orientation. Expected (d_model, k) or "
        "(k, d_model)."
    )


def _orthonormalize_rotation(rotation: Tensor) -> Tensor:
    """Return a rotation with orthonormal columns spanning the same subspace.

    ``SubspaceFeaturizer`` wraps the rotation in
    ``torch.nn.utils.parametrizations.orthogonal``, so a **non**-orthonormal
    basis would project through one frame here (``raw @ rotation``) while the
    featurizer — and every downstream consumer that rebuilds it — uses a
    *different* orthonormal frame. Orthonormalizing once (QR preserves the span)
    keeps the saved features, the manifold frame, and the path_steering frame in
    agreement. Already-orthonormal input is returned unchanged (the featurizer
    preserves it exactly), so PCA-style rotations are untouched.
    """
    import torch

    rotation = rotation.float()
    k = rotation.shape[1]
    gram = rotation.T @ rotation
    eye = torch.eye(k, dtype=gram.dtype, device=gram.device)
    if torch.allclose(gram, eye, atol=1e-4):
        return rotation
    q, _ = torch.linalg.qr(rotation)
    logger.warning(
        "Fixed rotation columns were not orthonormal; orthonormalized via QR "
        "(same span). The SubspaceFeaturizer orthogonalizes the basis anyway, so "
        "this keeps the saved features and the steering/manifold frames in sync."
    )
    return q


def find_fixed_subspace(
    sites: Sequence[Sequence[SiteSpec]],
    train_dataset: list,
    pipeline: LMPipeline,
    rotation: Tensor,
    batch_size: int = 32,
    output_dir: str = "",
    *,
    intervention_variable: str | None = None,
    embeddings: dict[str, Callable] | None = None,
    colormap: str | None = None,
    vis_dims: list[int] | None = None,
    variable_values: list[str] | None = None,
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
    figure_format: str = "png",
    k_features_hint: int | None = None,
) -> dict[str, Any]:
    """Thread a given ``(d_model, k)`` rotation and project features through it.

    Collects RAW activations (no featurizer), projects them through the given
    rotation, and writes the canonical bundle. Mirrors
    :func:`find_pca_subspace` minus the PCA fit. The frozen subspace
    featurizer is attached **functionally**: the returned ``spec`` carries it;
    the caller's input specs are unchanged.

    ``k_features_hint`` is only consulted to disambiguate a **square** rotation
    (see :func:`orient_rotation`); non-square input is oriented from ``d_model``.

    Returns:
        Dict with keys ``rotation``, ``features`` (``(N, k)``),
        ``k_features``, and ``spec`` (the updated :class:`SiteSpec`).
    """
    spec = sites[0][0]
    # Orient to (d_model, k) FIRST (idempotent), then orthonormalize. The order
    # matters: _orthonormalize_rotation assumes columns = k, and QR on a wide
    # (k, d_model) matrix would yield a wrong-shaped Q. The dispatch in main.py
    # already orients, but this protects direct callers of this public function.
    rotation = orient_rotation(
        rotation.detach(),
        d_model=int(pipeline.model.config.hidden_size),
        k_features_hint=k_features_hint,
    )
    rotation = _orthonormalize_rotation(rotation)
    k_features = int(rotation.shape[1])

    logger.info("Threading fixed subspace with k=%d...", k_features)
    # Collect RAW first — no featurizer set yet (see module docstring trap).
    # collect_output_logits=False (default) returns dict[str, Tensor].
    raw_features_dict = cast(
        dict[str, Tensor],
        collect_features(
            dataset=train_dataset,
            pipeline=pipeline,
            sites=[spec],
            batch_size=batch_size,
        ),
    )
    raw_features = raw_features_dict[spec.key].detach()
    rotation = rotation.to(raw_features.device)
    features = (raw_features.float() @ rotation.float()).detach()

    # Attach the featurizer now (after collection) so downstream decoding /
    # reconstruction sees the same projection — functionally: the returned
    # spec carries it, the input spec is unchanged.
    feat = SubspaceFeaturizer(rotation_subspace=rotation, trainable=False, id="PCA")
    spec = spec.with_featurizer(feat)

    if output_dir:
        save_subspace_artifacts(
            output_dir,
            train_dataset,
            rotation,
            raw_features,
            features,
        )

        vis_features = features[:, vis_dims] if vis_dims is not None else features
        save_features_visualization(
            vis_features,
            train_dataset,
            output_dir,
            intervention_variable,
            embeddings,
            colormap=colormap,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
        )

    return {
        "rotation": rotation,
        "features": features,
        "k_features": k_features,
        "spec": spec,
    }


def _score_single_cell(
    sites: Sequence[Sequence[SiteSpec]],
    cell_key: tuple[Any, ...],
    dataset: list,
    pipeline: LMPipeline,
    task: Any,
    batch_size: int,
    metric: Any,
    original_outputs: list | None,
) -> float:
    """Run one single-cell interchange scan and return its mean score.

    Thin wrapper over :func:`causalab.methods.interchange.run_layer_scan` for a
    one-cell spec grid, mirroring the pairwise branch of
    ``locate.run_interchange_scan`` (causalab/analyses/locate/run_interchange.py).
    When the cell's spec carries a frozen ``SubspaceFeaturizer`` the patch is
    restricted to that subspace; a featurizer-free spec gives the full-cell
    (unrestricted) control. The score's units are the task's
    ``intervention_metric`` (see :func:`score_fixed_subspace`).

    ``original_outputs`` is the base (un-intervened) forward pass, shared across
    cells by the caller (it depends only on ``dataset``/``pipeline``); pass ``None``
    for metrics that don't need it (``metric.needs_original_output is False``).
    """
    from causalab.methods.interchange import run_layer_scan

    scores = run_layer_scan(
        {cell_key: sites},
        dataset,
        pipeline,
        batch_size=batch_size,
        metric=metric,
        output_scores=metric.needs_scores,
        causal_model=task.causal_model,
        original_outputs=original_outputs,
    )
    return float(scores[cell_key])


def score_fixed_subspace(
    *,
    sites: Sequence[Sequence[SiteSpec]],
    cell_key: tuple[Any, ...],
    layer: int,
    position_name: str,
    pipeline: LMPipeline,
    task: Any,
    dataset: list,
    batch_size: int,
    intervention_metric: str,
) -> dict[str, Any]:
    """Score a threaded fixed subspace's causal mediation at one pinned cell.

    ``sites`` is the cell whose spec :func:`find_fixed_subspace` returned with
    the frozen ``SubspaceFeaturizer`` attached, so the **subspace-restricted**
    scan patches only within the given rotation. A fresh featurizer-free spec
    at the same ``(layer, position_name)`` gives the **full-cell**
    (unrestricted) control;
    ``score_ratio = subspace_score / full_cell_score`` is the fraction of the
    cell's mediation the subspace captures.

    The score's units are the task's ``intervention_metric``, exactly as in
    ``locate`` (which reports the same metric as ``scores_per_cell``):
    ``string_match`` / ``causal_label`` give the interchange-intervention accuracy
    (IIA, in [0, 1], higher = the subspace carries the variable); ``kl`` /
    ``output_shift*`` / ``hellinger`` give the base-vs-patched distribution shift
    (>= 0, higher = the variable is more strongly encoded there). Uses the same
    shipped scan path as ``locate`` pairwise mode (build the metric via
    ``resolve_interchange_metric``, scan via ``run_layer_scan``) — no bespoke
    mediation analysis. Retires the session-local ``fixed_subspace_mediation``
    pattern (#262).

    Returns ``{"intervention_metric", "subspace_score", "full_cell_score",
    "score_ratio"}``; the ratio is NaN when the full-cell score is zero or NaN
    (no meaningful denominator).
    """
    from causalab.runner.helpers import (
        build_targets_for_grid,
        get_output_token_ids,
        resolve_interchange_metric,
    )
    from causalab.methods.metric import compute_base_outputs

    score_token_ids, _n = get_output_token_ids(task, pipeline)
    metric = resolve_interchange_metric(
        intervention_metric,
        score_token_ids=score_token_ids,
        checker=task.checker,
    )
    # The base (un-intervened) forward pass depends only on (dataset, pipeline),
    # not on the target/featurizer, so compute it once and share it across both
    # the subspace-restricted scan and the full-cell control. Only distribution-
    # shift metrics (output_shift*/kl/hellinger) read it; causal-label metrics
    # pass None. Sharing avoids a redundant base forward for those metrics.
    original_outputs = (
        compute_base_outputs(dataset, pipeline, batch_size)
        if metric.needs_original_output
        else None
    )

    logger.info(
        "Scoring fixed subspace at (L%d, %s) with intervention_metric=%r ...",
        layer,
        position_name,
        intervention_metric,
    )
    subspace_score = _score_single_cell(
        sites, cell_key, dataset, pipeline, task, batch_size, metric, original_outputs
    )

    # Full-cell (unrestricted) control: rebuild the same cell with no featurizer.
    full_targets, _tp = build_targets_for_grid(pipeline, task, [layer], [position_name])
    full_key = next(iter(full_targets))
    full_cell_score = _score_single_cell(
        full_targets[full_key],
        full_key,
        dataset,
        pipeline,
        task,
        batch_size,
        metric,
        original_outputs,
    )

    # NaN guard: NaN != NaN rejects a NaN full-cell score; also reject 0.0.
    ratio = (
        float(subspace_score / full_cell_score)
        if full_cell_score == full_cell_score and full_cell_score != 0.0
        else float("nan")
    )
    logger.info(
        "Fixed subspace score (%s): subspace=%.6f full_cell=%.6f ratio=%.4f",
        intervention_metric,
        subspace_score,
        full_cell_score,
        ratio,
    )
    return {
        "intervention_metric": intervention_metric,
        "subspace_score": float(subspace_score),
        "full_cell_score": float(full_cell_score),
        "score_ratio": ratio,
    }
