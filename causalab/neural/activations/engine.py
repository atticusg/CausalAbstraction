"""
engine.py
=========
Run interventions through nnsight traces.

This is the execution core: it takes model units (*where*), their featurizers and
modes (*what*), and per-example token positions (*which tokens*), and runs the
model with those interventions applied.

Structure — and why it is two passes, not one
---------------------------------------------
Sources are read in their own forward, then written into the base's run. The
alternative — putting base and sources in one batched trace and handing
values across with ``tracer.barrier`` — also works, but it needs one barrier
round per site with every source's read fenced behind the rounds for the sites
before it; hoisting a read drives the model past a site the base has not written
yet. Two passes needs no barriers at all, and cross-model patching (sources from
a *different* model) and steering (no source pass at all) fall out of the same
code path.

No intervention lifecycle
-------------------------
There is no intervened model to build or tear down: an intervention exists only
for the duration of a trace. Nothing here allocates per-batch hooks, moves a
model to CPU, or calls ``gc.collect()`` / ``empty_cache()`` between batches, and
index tensors follow the activation they index rather than needing their own
device bookkeeping on a sharded model.

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
from typing import Any, Iterable, NamedTuple, Sequence

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
    "build_interventions",
    "FlatIndex",
    "align_per_example",
    "flat_index",
    "gather_positions",
    "scatter_positions",
    "collect_unit_activations",
    "collect_unit_activations_under",
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
class FlatIndex(NamedTuple):
    """Every selected ``(example, position)`` pair, flattened into one axis.

    A *rectangular* ``[batch, n_positions]`` index cannot represent a span whose
    width varies across the batch — ``torch.as_tensor`` on a ragged nested list
    raises. Naming each pair individually removes that restriction: an example
    selecting three tokens simply contributes three pairs.

    ``counts`` records how many pairs each example contributed, which is what
    lets a *per-example* value (a steering vector, a replacement) be expanded to
    line up with the flattened rows — see :func:`align_per_example`.
    """

    rows: Tensor  #: ``[total_positions]`` — which example each pair belongs to
    cols: Tensor  #: ``[total_positions]`` — which token within that example
    counts: Tensor  #: ``[batch]`` — positions selected per example


def flat_index(positions: Sequence[Sequence[int]], device: Any) -> FlatIndex:
    """Flatten per-example position lists into ``(row, col)`` pairs.

    On a uniform span this selects exactly what a rectangular
    ``[batch, n_positions]`` index selected, in the same order, so it is a
    drop-in for the rectangular path rather than a different one.
    """
    rows = [i for i, per_example in enumerate(positions) for _ in per_example]
    cols = [p for per_example in positions for p in per_example]
    counts = [len(per_example) for per_example in positions]
    return FlatIndex(
        rows=torch.as_tensor(rows, dtype=torch.long, device=device),
        cols=torch.as_tensor(cols, dtype=torch.long, device=device),
        counts=torch.as_tensor(counts, dtype=torch.long, device=device),
    )


def align_per_example(value: Tensor, index: FlatIndex) -> Tensor:
    """Repeat a per-example ``value`` so it lines up with the flattened rows.

    A steering vector is one value per example, but the activation it is applied
    to is now one row per *position*. An example selecting three tokens needs its
    vector three times. Values already carrying one row per position (an
    interchange source, gathered through the same index) are returned unchanged.
    """
    total = index.rows.shape[0]
    if value.ndim == 3:
        # A source carrying its own position axis: either one row per selected
        # position already, or the singleton axis that used to broadcast.
        if value.shape[0] * value.shape[1] == total:
            return value.reshape(-1, value.shape[-1])
        if value.shape[1] == 1:
            value = value.squeeze(1)
    if value.shape[0] == total:
        return value
    return value.repeat_interleave(index.counts.to(value.device), dim=0)


def gather_positions(
    activation: Tensor, index: FlatIndex, head: int | None = None
) -> Tensor:
    """Select every ``(example, position)`` pair (and optionally one head).

    Returns ``[total_positions, width]``, where width is the hidden size or — for
    a per-head site, whose activation is ``[batch, seq, n_heads, head_dim]`` —
    the head dimension. One row per selected token, so a batch whose examples
    select different numbers of tokens is representable.
    """
    selected = activation[index.rows, index.cols]
    if head is not None:
        selected = selected[:, head]
    return selected


def scatter_positions(
    activation: Tensor, index: FlatIndex, values: Tensor, head: int | None = None
) -> None:
    """Write ``values`` in place at every selected pair (and one head).

    In place because ``activation`` is a *view* of the tensor the model is about
    to consume: every transform in :mod:`causalab.neural.components` returns an
    alias — a tuple element, a channel slice, an ``unflatten`` — so the write
    reaches the forward on its own, with nothing to assign back. That also means
    units sharing a site compose, each one seeing its predecessors' writes.

    Each pair is named once, so there are no duplicate targets and no ordering
    question about which write wins.
    """
    if head is None:
        activation[index.rows, index.cols] = values.to(activation.dtype)
    else:
        activation[index.rows, index.cols, head] = values.to(activation.dtype)


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


def build_interventions(
    units: Iterable[AtomicModelUnit],
    mode: str,
    *,
    type_by_unit: dict[str, str] | None = None,
    noise_seed: int = 0,
    raw: bool = False,
) -> list[FeatureIntervention]:
    """One intervention per unit, in ``units`` order.

    Build these **once per run** and pass them to every batch's
    :func:`build_plans`. Two modes carry state across batches and would be wrong
    if rebuilt per batch:

    * ``noise`` advances its RNG, so rebuilding would hand identically-shaped
      consecutive batches the same corruption — making results depend on
      ``batch_size``.
    * ``mask`` owns learnable parameters, which an optimizer must see as one set
      for the whole run.
    """
    interventions = []
    for unit in units:
        unit_mode = type_by_unit[unit.id] if type_by_unit is not None else mode
        featurizer = _IDENTITY_FEATURIZER if raw else unit.featurizer
        interventions.append(
            build_intervention(
                featurizer,
                unit_mode,
                seed=noise_seed,
                tie_masks=featurizer.tie_masks,
            )
        )
    return interventions


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
    interventions: Sequence[FeatureIntervention] | None = None,
) -> list[UnitPlan]:
    """Pair each unit with its positions, mode and source for one batch.

    ``type_by_unit`` overrides ``mode`` per unit, which is how a single pass mixes
    modes — causal tracing corrupts the entry site with ``noise`` while restoring
    another with ``replace``. Mixing costs nothing: each unit simply carries a
    different intervention object.

    ``raw=True`` ignores each unit's featurizer and works in activation space.
    Use it when reading the *sources* of an interchange: the interchange
    intervention featurizes the source itself, so a featurized read would apply
    the featurizer twice — for a DAS rotation that is ``x @ Rᵀ @ Rᵀ``, which is
    both wrong and (for a rank-reducing subspace) a shape error. Collecting
    features for analysis is the opposite case and must keep the featurizer.

    ``interventions`` reuses objects from :func:`build_interventions` instead of
    constructing fresh ones — required whenever the interventions carry state
    across batches (``noise``, ``mask``).
    """
    units = list(units)
    if interventions is None:
        interventions = build_interventions(
            units, mode, type_by_unit=type_by_unit, noise_seed=noise_seed, raw=raw
        )
    plans: list[UnitPlan] = []
    for index, unit in enumerate(units):
        plans.append(
            UnitPlan(
                unit=unit,
                positions=positions[index],
                head=unit.head_index(),
                intervention=interventions[index],
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
                    flat_index(plan.positions, device),
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
            index = flat_index(plan.positions, device)
            selected = gather_positions(activation, index, plan.head)
            source = (
                None if plan.source is None else align_per_example(plan.source, index)
            )
            replacement = plan.intervention(selected, source, plan.feature_indices)
            scatter_positions(activation, index, replacement, plan.head)


def collect_unit_activations_under(
    pipeline: Any,
    encoding: dict[str, Any],
    write_plans: Sequence[UnitPlan],
    read_plans: Sequence[UnitPlan],
) -> list[Tensor]:
    """One forward that *writes* some sites and *reads* others.

    Path patching's first pass: the sender and restorers write while the
    receivers read, so each receiver sees the value it would see when only the
    sender→receiver path is perturbed. Correctness rests entirely on ordering —
    a receiver must be read after every upstream restorer has written — which
    holds because reads and writes are merged into one forward-ordered sequence
    rather than run as two independent passes.

    Returns the read plans' activations in ``read_plans`` order.
    """
    device = pipeline.model.device
    results: list[Tensor | None] = [None] * len(read_plans)
    read_index = {id(plan): i for i, plan in enumerate(read_plans)}
    combined = [*write_plans, *read_plans]
    writes = {id(plan) for plan in write_plans}

    with pipeline.nnsight.trace(**model_inputs(encoding)):
        for _key, group in _grouped_in_forward_order(combined):
            site = resolve_site(
                pipeline.nnsight, group[0].unit.component_type, group[0].unit.layer
            )
            activation = site.read()
            # Writes first: a receiver sharing a site with a restorer must read
            # the restored value, which is the whole point of the single pass.
            # The writes land in place, so the reads below see them.
            for plan in group:
                if id(plan) not in writes:
                    continue
                index = flat_index(plan.positions, device)
                selected = gather_positions(activation, index, plan.head)
                source = (
                    None
                    if plan.source is None
                    else align_per_example(plan.source, index)
                )
                replacement = plan.intervention(selected, source, plan.feature_indices)
                scatter_positions(activation, index, replacement, plan.head)
            for plan in group:
                if id(plan) in writes:
                    continue
                selected = gather_positions(
                    activation, flat_index(plan.positions, device), plan.head
                )
                results[read_index[id(plan)]] = plan.intervention(
                    selected, None, plan.feature_indices
                ).save()

    collected = [r for r in results if r is not None]
    if len(collected) != len(read_plans):  # pragma: no cover - defensive
        raise RuntimeError(
            f"Collected {len(collected)} activations for {len(read_plans)} receivers."
        )
    return collected


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
    device = pipeline.model.device
    with pipeline.nnsight.trace(**model_inputs(base_encoding)):
        _apply(pipeline, plans, device)
        output = pipeline.nnsight.output.save()
    return output
