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
multi-token subject) while the restored site uses ``replace`` — a *mixed* model,
expressed through ``run_steering_interventions``'s ``type_by_unit`` map. Sites
are visited in forward order, so the layer ``-1`` corruption is written
before any layer ``>= 0`` restore reads it, and the restored site's clean value
propagates through an already-corrupted network — no two-pass collect/inject.

Entry points, mirroring :mod:`causalab.methods.ablation.run`:

* :func:`run_causal_trace` — one combined corrupt+restore pass, raw outputs.
* :func:`run_corrupted_floor` — corruption only (no restore), one score.
* :func:`run_causal_trace_scan` — sweep the restore grid, one score per cell.
"""

from __future__ import annotations

from typing import Any, Iterable

from torch import Tensor
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.steer.steer import run_steering_interventions
from causalab.neural.units import AtomicModelUnit, InterchangeTarget


def _type_map(
    entry_units: list[AtomicModelUnit],
    swept_units: list[AtomicModelUnit],
    entry_type: str,
) -> dict[str, str] | None:
    """Per-unit intervention types, or ``None`` when every site is ``replace``.

    ``zero``/``mean`` corruption is all-``replace`` (returns ``None`` → the
    homogeneous path). ``noise`` corruption marks the entry units ``noise`` and
    the restored units ``replace`` — the mixed model.
    """
    if entry_type == "replace":
        return None
    type_map = {unit.id: entry_type for unit in entry_units}
    type_map.update({unit.id: "replace" for unit in swept_units})
    return type_map


def run_causal_trace(
    pipeline: Any,
    dataset: list[CounterfactualExample],
    target: InterchangeTarget,
    vectors: dict[str, Tensor],
    *,
    type_by_unit: dict[str, str] | None = None,
    noise_seed: int = 0,
    batch_size: int = 16,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Apply corruption + restoration in one pass; return raw generation outputs.

    ``target`` groups the entry (corruption) units and the restored (swept)
    units; ``vectors`` maps every unit id to its reference tensor (broadcast
    corruption vectors / noise scales, per-example clean restore values).
    ``type_by_unit`` (from :func:`_type_map`) makes the entry ``noise`` and the
    restore ``replace`` for noise corruption; ``None`` runs everything as
    ``replace``.

    The whole dataset is one ``run_steering_interventions`` call, so a cell's
    noise stream starts from ``noise_seed`` and advances once across every batch.
    That is what makes grid cells comparable: each cell is its own call and so
    replays the same stream. An all-position entry span no longer needs the
    dataset regrouped by length, which also removes the surprise that two
    disjoint groups of examples used to be handed the *same* noise draws.
    """
    return run_steering_interventions(
        pipeline,
        dataset,
        target,
        vectors,
        batch_size=batch_size,
        output_scores=output_scores,
        mode="replace",
        type_by_unit=type_by_unit,
        noise_seed=noise_seed,
    )


def run_corrupted_floor(
    pipeline: Any,
    dataset: list[CounterfactualExample],
    entry_units: list[AtomicModelUnit],
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
    entry_units = list(entry_units)
    target = InterchangeTarget([entry_units])
    vectors = {unit.id: corruption_vectors[unit.id] for unit in entry_units}
    type_by_unit = _type_map(entry_units, [], entry_type)
    outputs = run_causal_trace(
        pipeline,
        dataset,
        target,
        vectors,
        type_by_unit=type_by_unit,
        noise_seed=noise_seed,
        batch_size=batch_size,
        output_scores=output_scores,
    )
    return score_intervention_outputs(
        raw_results={("floor",): outputs},
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )[("floor",)]


def run_causal_trace_scan(
    swept_targets: dict[tuple[Any, ...], InterchangeTarget],
    dataset: list[CounterfactualExample],
    pipeline: Any,
    *,
    entry_units: Iterable[AtomicModelUnit],
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

    Each ``(key, swept_target)`` restores *every* unit in that target — a single
    site, or a window of consecutive layers at the site (the window is encoded by
    the analysis as multiple units per cell) — to its clean value while the entry
    span stays corrupted, then scores the result. Returns ``{key: restored_score}``;
    the caller turns this into recovery against the corrupted floor. Each cell is
    scored as it is generated (only one cell's logits stay alive), so full-vocab
    ``output_scores=True`` is affordable.
    """
    entry_units = list(entry_units)
    entry_vectors = {unit.id: corruption_vectors[unit.id] for unit in entry_units}
    entry_ids = {unit.id for unit in entry_units}

    results: dict[tuple[Any, ...], float] = {}
    for key, swept_target in tqdm(
        swept_targets.items(), desc="Causal-trace scan", total=len(swept_targets)
    ):
        swept_units = swept_target.flatten()
        # A restored site that coincides with a corrupted entry site would share a
        # unit id: the clean value would silently overwrite the corruption (dict
        # merge + type map), mis-scoring the cell. Forbid it. (The default
        # corruption.layer=-1 corrupts block_input while residual restores target
        # block_output, so their ids differ and this never trips.)
        clash = entry_ids & {unit.id for unit in swept_units}
        if clash:
            raise ValueError(
                f"Restore cell {key} overlaps the corrupted entry site(s) {sorted(clash)}; "
                "the restore would cancel the corruption. Choose a corruption layer/"
                "component distinct from the restored sites."
            )
        combined = InterchangeTarget([entry_units, swept_units])
        vectors = {
            **entry_vectors,
            **{unit.id: clean_vectors[unit.id] for unit in swept_units},
        }
        type_by_unit = _type_map(entry_units, swept_units, entry_type)
        outputs = run_causal_trace(
            pipeline,
            dataset,
            combined,
            vectors,
            type_by_unit=type_by_unit,
            noise_seed=noise_seed,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        results[key] = score_intervention_outputs(
            raw_results={key: outputs},
            dataset=dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )[key]
    return results
