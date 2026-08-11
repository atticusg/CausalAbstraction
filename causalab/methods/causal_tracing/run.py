"""Run causal tracing (sufficiency): corrupt the entry, restore one site, score.

Causal tracing inverts ablation. Ablation removes a site and asks whether the
behaviour breaks (necessity); tracing degrades the information where it enters,
establishing a broken floor, then **restores one clean site at a time** (or a
small window of layers at a site) and asks how much behaviour returns
(sufficiency / mediation).

The corruption and the restoration are applied together in a **single** forward
pass. For ``zero``/``mean`` corruption every site is a ``replace`` (one
homogeneous pass). For ``noise`` corruption the entry uses the dynamic, seeded
``noise`` intervention (independent per-token Gaussian, so it spans the whole
multi-token subject) while the restored site uses ``replace`` — a *mixed* run,
expressed through ``run_steering_interventions``'s ``type_by_key`` map. The
engine emits the edits in forward order, so the layer ``-1`` corruption is
written before any layer ``>= 0`` restore reads it, and the restored site's
clean value propagates through an already-corrupted network — no two-pass
collect/inject.

Entry points, mirroring :mod:`causalab.methods.ablation.run`:

* :func:`run_causal_trace` — one combined corrupt+restore pass, raw outputs.
* :func:`run_corrupted_floor` — corruption only (no restore), one score.
* :func:`run_causal_trace_scan` — sweep the restore grid, one score per cell.

Sites are :class:`~causalab.neural.specs.SiteSpec` values (WU4, #506); the
restore grid is the WU2 :data:`~causalab.neural.activations.site_grids.SiteGrid`
shape.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from torch import Tensor
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.activations.site_grids import SiteGrid
from causalab.neural.pipeline import GenerationResult
from causalab.neural.specs import SiteSpec


def _type_map(
    entry_sites: Sequence[SiteSpec],
    swept_sites: Sequence[SiteSpec],
    entry_type: str,
) -> dict[str, str] | None:
    """Per-site intervention types, or ``None`` when every site is ``replace``.

    ``zero``/``mean`` corruption is all-``replace`` (returns ``None`` → the
    homogeneous path). ``noise`` corruption marks the entry sites ``noise`` and
    the restored sites ``replace`` — the mixed model.
    """
    if entry_type == "replace":
        return None
    type_map = {spec.key: entry_type for spec in entry_sites}
    type_map.update({spec.key: "replace" for spec in swept_sites})
    return type_map


def run_causal_trace(
    pipeline: Any,
    dataset: list[CounterfactualExample],
    sites: Sequence[SiteSpec],
    vectors: dict[str, Tensor],
    *,
    type_by_key: dict[str, str] | None = None,
    noise_seed: int = 0,
    batch_size: int = 16,
    output_scores: bool | int = True,
    gen_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """Apply corruption + restoration in one pass; return the flat
    :class:`~causalab.neural.pipeline.GenerationResult` (EU5b, #487).

    ``sites`` is the flat entry (corruption) + restored (swept) site list;
    ``vectors`` maps every ``spec.key`` to its reference tensor (broadcast
    corruption vectors / noise scales, per-example clean restore values).
    ``type_by_key`` (from :func:`_type_map`) makes the entry ``noise`` and the
    restore ``replace`` for noise corruption; ``None`` runs everything as
    ``replace``.

    On the nnsight engine (PL3, #405) ragged entry spans batch natively, so
    the whole dataset runs as one stream in dataset order — no
    length-bucketing, no per-bucket vector reindexing, no pyvene model reuse.
    Each call draws its noise from one stream seeded ``noise_seed``, so a
    cell's run is reproducible end-to-end. ``gen_kwargs`` are extra HF
    ``generate`` kwargs forwarded through to the engine (e.g.
    ``{"min_new_tokens": N}`` — the escape hatch its ragged-scores refusal
    names).
    """
    return run_steering_interventions(
        pipeline,
        dataset,
        sites,
        vectors,
        batch_size=batch_size,
        output_scores=output_scores,
        mode="replace",
        type_by_key=type_by_key,
        noise_seed=noise_seed,
        gen_kwargs=gen_kwargs,
    )


def run_corrupted_floor(
    pipeline: Any,
    dataset: list[CounterfactualExample],
    entry_sites: Sequence[SiteSpec],
    corruption_vectors: dict[str, Tensor],
    *,
    metric: InterchangeMetric,
    entry_type: str = "replace",
    noise_seed: int = 0,
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> float:
    """Score the broken-behaviour floor: corruption applied, nothing restored."""
    entry_sites = list(entry_sites)
    vectors = {spec.key: corruption_vectors[spec.key] for spec in entry_sites}
    type_by_key = _type_map(entry_sites, [], entry_type)
    outputs = run_causal_trace(
        pipeline,
        dataset,
        entry_sites,
        vectors,
        type_by_key=type_by_key,
        noise_seed=noise_seed,
        batch_size=batch_size,
        output_scores=output_scores,
    )
    return score_intervention_outputs(
        results={("floor",): outputs},
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )[("floor",)]


def run_causal_trace_scan(
    swept_grid: SiteGrid,
    dataset: list[CounterfactualExample],
    pipeline: Any,
    *,
    entry_sites: Iterable[SiteSpec],
    corruption_vectors: dict[str, Tensor],
    clean_vectors: dict[str, Tensor],
    metric: InterchangeMetric,
    entry_type: str = "replace",
    noise_seed: int = 0,
    batch_size: int = 16,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[tuple[Any, ...], float]:
    """Restore each grid cell over the corrupted backdrop; score each cell.

    Each ``(key, swept groups)`` cell restores *every* site in that cell — a
    single site, or a window of consecutive layers at the site (the window is
    encoded by the analysis as multiple specs per cell) — to its clean value
    while the entry span stays corrupted, then scores the result. Returns
    ``{key: restored_score}``; the caller turns this into recovery against the
    corrupted floor. Each cell is scored as it is generated (only one cell's
    logits stay alive), so full-vocab ``output_scores=True`` is affordable.
    """
    entry_sites = list(entry_sites)
    entry_vectors = {spec.key: corruption_vectors[spec.key] for spec in entry_sites}
    entry_keys = {spec.key for spec in entry_sites}

    results: dict[tuple[Any, ...], float] = {}
    for key, groups in tqdm(
        swept_grid.items(), desc="Causal-trace scan", total=len(swept_grid)
    ):
        swept_sites = [spec for group in groups for spec in group]
        # A restored site that coincides with a corrupted entry site would share a
        # spec key: the clean value would silently overwrite the corruption (dict
        # merge + type map), mis-scoring the cell. Forbid it. (The default
        # corruption.layer=-1 corrupts block_input while residual restores target
        # block_output, so their keys differ and this never trips.)
        clash = entry_keys & {spec.key for spec in swept_sites}
        if clash:
            raise ValueError(
                f"Restore cell {key} overlaps the corrupted entry site(s) {sorted(clash)}; "
                "the restore would cancel the corruption. Choose a corruption layer/"
                "component distinct from the restored sites."
            )
        combined = entry_sites + swept_sites
        vectors = {
            **entry_vectors,
            **{spec.key: clean_vectors[spec.key] for spec in swept_sites},
        }
        type_by_key = _type_map(entry_sites, swept_sites, entry_type)
        outputs = run_causal_trace(
            pipeline,
            dataset,
            combined,
            vectors,
            type_by_key=type_by_key,
            noise_seed=noise_seed,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        results[key] = score_intervention_outputs(
            results={key: outputs},
            dataset=dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )[key]
    return results
