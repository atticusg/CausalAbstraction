"""
engine.py
=========
Run interventions through nnsight traces.

This is the execution core: it takes model units (*where*), their featurizers and
modes (*what*), and per-example token positions (*which tokens*), and runs the
model with those interventions applied.

Structure — and why it is two passes, not one
---------------------------------------------
Sources are read in their own forward, then written into the base's run. That
mirrors what the pyvene backbone did (each source was a plain collection forward;
the base was the generation prefill), so activations and logits match it exactly.
The alternative — putting base and sources in one batched trace and handing
values across with ``tracer.barrier`` — also works and was verified during the
migration spike, but it needs sites visited in forward order with *two* barrier
rounds each (all-read, sync, base-writes, sync) or a later source pushes the
model past a site the base has not written yet. Two passes needs no barriers at
all, and cross-model patching (sources from a *different* model) and steering
(no source pass at all) fall out of the same code path.

What this replaces
------------------
``prepare_intervenable_model`` / ``delete_intervenable_model`` and the pyvene
``IntervenableModel`` lifecycle. There is no model to build or tear down: an
intervention exists only for the duration of a trace, so the per-batch
build-hooks / move-to-CPU / ``gc.collect()`` / ``empty_cache()`` cycle is gone,
along with the sharded-model device bookkeeping that existed to place pyvene's
index tensors.

Ordering contract
-----------------
The interleaver serves each module location once, in forward order. Sites are
therefore visited in ``(layer, component rank)`` order, and units sharing a site
are applied together in one read/modify/write — reading the same location twice
would raise ``OutOfOrderError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from causalab.neural.components import forward_order, resolve_site
from causalab.neural.featurizer import Featurizer
from causalab.neural.interventions import FeatureIntervention, build_intervention
from causalab.neural.units import AtomicModelUnit

logger = logging.getLogger(__name__)

# Stateless, so one shared instance is safe. Used by `build_plans(raw=True)` to
# read an activation in its own space rather than a unit's feature space.
_IDENTITY_FEATURIZER = Featurizer()

__all__ = [
    "UnitPlan",
    "build_plans",
    "gather_positions",
    "scatter_positions",
    "collect_unit_activations",
    "generate_with_interventions",
    "forward_with_interventions",
    "model_inputs",
]

# The keys a causal LM's forward accepts. `pipeline.load` also returns
# offset_mapping / wrapped_text / content_char_offset for token-position
# resolution, which the model must not see.
_FORWARD_KEYS = ("input_ids", "attention_mask", "position_ids")


def model_inputs(encoding: dict[str, Any]) -> dict[str, Tensor]:
    """The subset of a ``pipeline.load`` encoding the model's forward accepts."""
    return {k: v for k, v in encoding.items() if k in _FORWARD_KEYS}


# --------------------------------------------------------------------------- #
#  Position indexing                                                           #
# --------------------------------------------------------------------------- #
def _as_position_tensor(positions: Sequence[Sequence[int]], device: Any) -> Tensor:
    """``[batch, n_positions]`` index tensor from per-example position lists.

    Requires a uniform width across the batch — every example must select the
    same number of tokens. A ragged selection cannot form a rectangular index
    tensor; :func:`causalab.neural.activations.interchange_mode.prepare_intervenable_inputs`
    rejects it up front with an actionable message.
    """
    return torch.as_tensor(positions, dtype=torch.long, device=device)


def gather_positions(
    activation: Tensor, positions: Tensor, head: int | None = None
) -> Tensor:
    """Select ``positions`` (and optionally one head) from a ``[batch, seq, ...]`` activation.

    Returns ``[batch, n_positions, width]``, where width is the hidden size or —
    for a per-head site, whose activation is ``[batch, seq, n_heads, head_dim]`` —
    the head dimension.
    """
    rows = torch.arange(activation.shape[0], device=activation.device).unsqueeze(1)
    selected = activation[rows, positions]
    if head is not None:
        selected = selected[:, :, head]
    return selected


def scatter_positions(
    activation: Tensor, positions: Tensor, values: Tensor, head: int | None = None
) -> Tensor:
    """Write ``values`` back at ``positions`` (and one head), returning a new tensor.

    Out-of-place: advanced indexing yields a copy, so the updated slice has to be
    written back into a clone. Returning a new tensor also keeps the caller from
    mutating the live activation before every unit at the site has been applied.
    """
    rows = torch.arange(activation.shape[0], device=activation.device).unsqueeze(1)
    updated = activation.clone()
    if head is None:
        updated[rows, positions] = values.to(updated.dtype)
        return updated
    patch = updated[rows, positions]  # [batch, n_positions, n_heads, head_dim]
    patch[:, :, head] = values.to(patch.dtype)
    updated[rows, positions] = patch
    return updated


# --------------------------------------------------------------------------- #
#  Plans                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class UnitPlan:
    """Everything needed to apply one unit's intervention to one batch.

    ``source`` is whatever the unit's mode consumes: another run's activation
    (interchange), a steering vector, a replacement value, a noise scale — or
    ``None`` for collection.
    """

    unit: AtomicModelUnit
    positions: Sequence[Sequence[int]]
    head: int | None
    intervention: FeatureIntervention
    feature_indices: list[int] | None = None
    source: Tensor | None = None

    @property
    def site_key(self) -> tuple[int, str]:
        """Units sharing this key read and write the same module location."""
        return (self.unit.layer, self.unit.component_type)

    @property
    def order(self) -> tuple[int, int]:
        return forward_order(self.unit.layer, self.unit.component_type)


def build_plans(
    units: Iterable[AtomicModelUnit],
    positions: Sequence[Sequence[Sequence[int]]],
    mode: str,
    *,
    sources: Sequence[Tensor | None] | None = None,
    feature_indices: Sequence[list[int] | None] | None = None,
    type_by_unit: dict[str, str] | None = None,
    noise_seed: int = 0,
    raw: bool = False,
) -> list[UnitPlan]:
    """Pair each unit with its positions, mode and source for one batch.

    ``type_by_unit`` overrides ``mode`` per unit, which is how a single pass mixes
    modes — causal tracing corrupts the entry site with ``noise`` while restoring
    another with ``replace``. Unlike the pyvene backbone, mixing costs nothing
    here: each unit simply carries a different intervention object.

    ``raw=True`` ignores each unit's featurizer and works in activation space.
    Use it when reading the *sources* of an interchange: the interchange
    intervention featurizes the source itself, so a featurized read would apply
    the featurizer twice — for a DAS rotation that is ``x @ Rᵀ @ Rᵀ``, which is
    both wrong and (for a rank-reducing subspace) a shape error. Collecting
    features for analysis is the opposite case and must keep the featurizer.
    """
    units = list(units)
    plans: list[UnitPlan] = []
    for index, unit in enumerate(units):
        unit_mode = type_by_unit[unit.id] if type_by_unit is not None else mode
        featurizer = _IDENTITY_FEATURIZER if raw else unit.featurizer
        plans.append(
            UnitPlan(
                unit=unit,
                positions=positions[index],
                head=unit.head_index(),
                intervention=build_intervention(
                    featurizer,
                    unit_mode,
                    seed=noise_seed,
                    tie_masks=featurizer.tie_masks,
                ),
                feature_indices=(
                    feature_indices[index] if feature_indices is not None else None
                ),
                source=sources[index] if sources is not None else None,
            )
        )
    return plans


def _grouped_in_forward_order(
    plans: Sequence[UnitPlan],
) -> list[tuple[tuple[int, str], list[UnitPlan]]]:
    """Plans bucketed by site, the buckets ordered as the forward reaches them."""
    buckets: dict[tuple[int, str], list[UnitPlan]] = {}
    for plan in plans:
        buckets.setdefault(plan.site_key, []).append(plan)
    return sorted(buckets.items(), key=lambda item: item[1][0].order)


# --------------------------------------------------------------------------- #
#  Reading                                                                     #
# --------------------------------------------------------------------------- #
def collect_unit_activations(
    pipeline: Any,
    encoding: dict[str, Any],
    plans: Sequence[UnitPlan],
    *,
    return_logits: bool = False,
) -> list[Tensor] | tuple[list[Tensor], Tensor]:
    """One forward; return each plan's featurized activation, in ``plans`` order.

    Each site is read once even when several units share it, and sites are read
    in forward order — both required by the interleaver, neither visible to
    callers, who get their results back in the order they asked.
    """
    pipeline.ensure_instrumented()
    device = pipeline.model.device
    results: list[Tensor | None] = [None] * len(plans)
    index_of = {id(plan): i for i, plan in enumerate(plans)}

    with pipeline.nnsight.trace(**model_inputs(encoding)):
        for _key, group in _grouped_in_forward_order(plans):
            site = resolve_site(
                pipeline.nnsight, group[0].unit.component_type, group[0].unit.layer
            )
            activation = site.read()
            for plan in group:
                selected = gather_positions(
                    activation,
                    _as_position_tensor(plan.positions, device),
                    plan.head,
                )
                results[index_of[id(plan)]] = plan.intervention(
                    selected, None, plan.feature_indices
                ).save()
        logits = pipeline.nnsight.output.logits.save() if return_logits else None

    collected = [r for r in results if r is not None]
    if len(collected) != len(plans):  # pragma: no cover - defensive
        raise RuntimeError(
            f"Collected {len(collected)} activations for {len(plans)} units."
        )
    if return_logits:
        assert logits is not None
        return collected, logits
    return collected


# --------------------------------------------------------------------------- #
#  Writing                                                                     #
# --------------------------------------------------------------------------- #
def _apply(pipeline: Any, plans: Sequence[UnitPlan], device: Any) -> None:
    """Apply every plan, inside an already-open trace."""
    for _key, group in _grouped_in_forward_order(plans):
        site = resolve_site(
            pipeline.nnsight, group[0].unit.component_type, group[0].unit.layer
        )
        activation = site.read()
        for plan in group:
            positions = _as_position_tensor(plan.positions, device)
            selected = gather_positions(activation, positions, plan.head)
            replacement = plan.intervention(selected, plan.source, plan.feature_indices)
            activation = scatter_positions(
                activation, positions, replacement, plan.head
            )
        site.write(activation)


def generate_with_interventions(
    pipeline: Any,
    base_encoding: dict[str, Any],
    plans: Sequence[UnitPlan],
    **gen_kwargs: Any,
) -> Any:
    """Generate from ``base_encoding`` with every plan applied to the prefill.

    Interventions land on each site's *first* occurrence — the prompt prefill —
    matching the ``intervene_on_prompt`` behaviour callers rely on. Reapplying at
    every decode step is a different experiment; it would wrap the body in
    ``tracer.iter``.
    """
    pipeline.ensure_instrumented()
    device = pipeline.model.device
    defaults: dict[str, Any] = {
        "max_new_tokens": pipeline.max_new_tokens,
        "pad_token_id": pipeline.tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": False,
        "use_cache": True,
    }
    defaults.update(gen_kwargs)
    with pipeline.nnsight.generate(**model_inputs(base_encoding), **defaults) as tracer:
        _apply(pipeline, plans, device)
        result = tracer.result.save()
    return result


def forward_with_interventions(
    pipeline: Any,
    base_encoding: dict[str, Any],
    plans: Sequence[UnitPlan],
) -> Any:
    """One forward with every plan applied; returns the model's output object.

    The differentiable path: nothing here detaches, so a loss built from the
    returned logits backpropagates into any learnable featurizer or mask the
    plans carry.
    """
    pipeline.ensure_instrumented()
    device = pipeline.model.device
    with pipeline.nnsight.trace(**model_inputs(base_encoding)):
        _apply(pipeline, plans, device)
        output = pipeline.nnsight.output.save()
    return output
