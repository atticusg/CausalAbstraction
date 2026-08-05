"""Run path patching for a sender→receiver edge and score the direct effect.

Two regimes, dispatched on the receiver:

* **``receiver = output``** (default) — the *direct effect on the logits* is the
  degenerate one-pass case: once the restorer set is frozen to the clean base, the
  only surviving sender→logits route is the one selected by ``restore`` (the bare
  residual stream, or residual + MLPs), so the metric is read straight from a
  single intervened forward via the fully-tested
  :func:`run_interchange_interventions`. The caller passes an ordinary
  ``(base, source)`` counterfactual dataset and the runner arranges the two
  source groups so the sender reads the source and the restorers read the base.

* **internal receiver** (``head_value_input`` / ``mlp_input`` / ``residual``) — the
  sender's perturbation also reaches the output through routes that *bypass* the
  receiver, so a single forward would not isolate the edge. The two-pass procedure
  (:func:`~causalab.neural.activations.interchange_mode.run_two_pass_path_patching`)
  collects the receiver's path-restricted input, then re-injects it on an otherwise
  clean base run — leaving exactly the sender→receiver edge effect.

The scan loops a grid of senders and scores each cell with an ``InterchangeMetric``
(reuse the logit-difference metric in
:mod:`causalab.methods.path_patching.metric`).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_receiver_unit,
    build_restorer_set,
    deepest_receiver,
    group_path_patching_target,
    sender_reaches_receiver,
)
from causalab.neural.activations.interchange_mode import (
    run_interchange_interventions,
    run_two_pass_path_patching,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.units import AtomicModelUnit

logger = logging.getLogger(__name__)


def _with_base_as_second_source(
    dataset: list[CounterfactualExample],
) -> list[CounterfactualExample]:
    """Re-present each example so the restorer group reads the base input.

    The path-patching target is ``[[sender], [restorers]]`` (group 0 reads the
    *source*, group 1 reads the *base*). The runner zips groups to
    ``counterfactual_inputs``, so each example must supply
    ``counterfactual_inputs = [source, base]`` — the original source for the
    sender, then the example's own base for the restorers. Only references are
    reused (interchange reads, never mutates). Patching the restorers from the
    base is **not** a no-op: in the intervened forward the sender is perturbed, so
    overwriting downstream components with their clean values cancels the indirect
    propagation, leaving only the sender's direct residual contribution.
    """
    return [
        {
            "input": ex["input"],
            "counterfactual_inputs": [ex["counterfactual_inputs"][0], ex["input"]],
        }
        for ex in dataset
    ]


def _two_pass_for_sender(
    pipeline: LMPipeline,
    presented_dataset: list[CounterfactualExample],
    sender: AtomicModelUnit,
    receiver_units: list[AtomicModelUnit],
    read_receiver: ReceiverSpec,
    *,
    restore: Iterable[str],
    batch_size: int,
    output_scores: bool | int,
) -> dict[str, list[Any]]:
    """Build ``sender``'s restorer set/target and run the two-pass collect/inject.

    Single source of the per-sender two-pass invocation, shared by the single-receiver
    runner (:func:`_run_for_sender`) and the receiver-set runners. ``read_receiver`` is
    the :class:`ReceiverSpec` the restorer *range* is built against — the receiver
    itself for one receiver, the deepest member for a set (its read point bounds the
    union restorer set, isolating the direct sender→R_k edge for every member at once).
    The presented ``[source, base]`` dataset feeds both groups (the receivers read the
    base); an empty-restorer target collapses to a single interchange group, which the
    two-pass runner handles by sending one source plus the collect getters.
    """
    restorers = build_restorer_set(pipeline, sender, read_receiver, restore=restore)
    target = group_path_patching_target(sender, restorers)
    return run_two_pass_path_patching(
        pipeline,
        presented_dataset,
        target,
        receiver_units,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def _score_cell(
    key: tuple[Any, ...],
    outputs: dict[str, list[Any]],
    *,
    dataset: list[CounterfactualExample],
    metric: InterchangeMetric,
    causal_model: CausalModel | None,
    original_outputs: list[dict[str, Any]] | None,
) -> float:
    """Score one swept cell's raw generation outputs with ``metric``.

    Shared by both scans so the (identical) scoring call cannot drift between them.
    """
    return score_intervention_outputs(
        raw_results={key: outputs},
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )[key]


def _run_for_sender(
    pipeline: LMPipeline,
    base_dataset: list[CounterfactualExample],
    presented_dataset: list[CounterfactualExample],
    sender: AtomicModelUnit,
    receiver: ReceiverSpec,
    receiver_unit: AtomicModelUnit | None,
    *,
    restore: Iterable[str],
    batch_size: int,
    output_scores: bool | int,
) -> dict[str, list[Any]]:
    """Path-patch one sender against ``receiver`` and return raw generation outputs.

    ``presented_dataset`` is ``base_dataset`` re-presented as ``[source, base]``
    (see :func:`_with_base_as_second_source`). It is sender-independent, so the
    scan builds it once and threads it in here, along with the (also
    sender-independent) ``receiver_unit``.

    ``receiver = output`` is the one-pass case: the target grouping routes through
    :func:`~causalab.methods.path_patching.targets.group_path_patching_target` (so
    it matches :func:`build_path_patching_target` exactly, including the
    empty-restorer collapse to a single-group, single-source interchange) and the
    metric is read off the single intervened forward. Any internal receiver runs
    the two-pass collect/inject instead.

    Raises ``ValueError`` for an internal receiver the sender cannot reach (sender at
    or downstream of the receiver's read point) — there is no forward path, so the run
    would only confirm a structural zero. ``output`` receivers are always reachable.
    """
    if not sender_reaches_receiver(pipeline, sender, receiver):
        raise ValueError(
            f"Sender {sender.id!r} is not upstream of receiver "
            f"{receiver.kind!r}@layer={receiver.layer}: it writes at or after the "
            f"receiver's read point, so no forward path connects them and the direct "
            f"effect is structurally zero. Pick a sender upstream of the receiver."
        )
    if receiver.kind == "output":
        # With restorers: group 1 reads the base, so each example needs
        # counterfactual_inputs = [source, base] (the re-presented dataset).
        # Without restorers the target is single-group, so the 1-source dataset.
        restorers = build_restorer_set(pipeline, sender, receiver, restore=restore)
        target = group_path_patching_target(sender, restorers)
        dataset = presented_dataset if restorers else base_dataset
        return run_interchange_interventions(
            pipeline,
            dataset,
            target,
            batch_size=batch_size,
            output_scores=output_scores,
        )
    assert receiver_unit is not None
    # Internal receiver: PASS 1 collects the receiver under the sender+restorer
    # interchange, PASS 2 injects it on a clean base run.
    return _two_pass_for_sender(
        pipeline,
        presented_dataset,
        sender,
        [receiver_unit],
        receiver,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    sender: AtomicModelUnit,
    receiver: ReceiverSpec = OUTPUT,
    *,
    restore: Iterable[str] = ("attention", "mlp"),
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Path-patch the edge ``sender → receiver`` and return raw generation outputs.

    ``receiver`` defaults to the output logits (the IOI Fig. 3 head-sweep case).
    ``restore`` selects which component families are frozen between the sender and
    the receiver (see
    :func:`~causalab.methods.path_patching.targets.build_restorer_set`). The
    outputs (``{"string", "sequences", "scores"}``) are scored by the caller
    (e.g. via :func:`run_path_patching_scan` or ``score_intervention_outputs``).
    """
    presented = _with_base_as_second_source(counterfactual_dataset)
    receiver_unit = build_receiver_unit(pipeline, receiver)
    return _run_for_sender(
        pipeline,
        counterfactual_dataset,
        presented,
        sender,
        receiver,
        receiver_unit,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching_scan(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    senders: dict[tuple[Any, ...], AtomicModelUnit],
    *,
    metric: InterchangeMetric,
    receiver: ReceiverSpec = OUTPUT,
    restore: Iterable[str] = ("attention", "mlp"),
    batch_size: int = 32,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[tuple[Any, ...], float]:
    """Sweep a grid of senders against ``receiver``; score each with ``metric``.

    ``senders`` maps a grid key (e.g. ``(layer, head)``) to the sender unit.
    Returns ``{key: score}`` ready for a layer×head heatmap. Each cell is
    generated then scored immediately, so only one cell's logits stay alive (the
    reason full-vocab ``output_scores`` is affordable).
    """
    restore = tuple(restore)  # materialize: reused per cell
    # The re-presented [source, base] dataset and the receiver unit are both
    # sender-independent, so build them once here rather than per cell.
    presented = _with_base_as_second_source(counterfactual_dataset)
    receiver_unit = build_receiver_unit(pipeline, receiver)
    scores: dict[tuple[Any, ...], float] = {}
    for key, sender in tqdm(
        senders.items(), desc="path-patch scan", total=len(senders)
    ):
        outputs = _run_for_sender(
            pipeline,
            counterfactual_dataset,
            presented,
            sender,
            receiver,
            receiver_unit,
            restore=restore,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        scores[key] = _score_cell(
            key,
            outputs,
            dataset=counterfactual_dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )
    return scores


def _validate_receiver_set(receiver_specs: list[ReceiverSpec]) -> None:
    """Guard the two preconditions a receiver *set* must satisfy.

    (1) Internal receivers only — ``output`` is the one-pass case handled by
    :func:`run_path_patching`, not a member of a two-pass set. (2) A single shared
    token position — :func:`build_restorer_set` freezes restorers at the receiver's
    ``token_position`` (a single position to stay one-token-per-position), so a set
    spanning positions has no single union restorer set. The IOI Fig. 4 / Fig. 5 sets
    satisfy both (END for name-mover queries, S2 for S-inhibition values).
    """
    if not receiver_specs:
        raise ValueError("receiver_specs is empty; pass at least one ReceiverSpec.")
    if any(rs.kind == "output" for rs in receiver_specs):
        raise ValueError(
            "Receiver sets are internal-receiver only; 'output' is the one-pass "
            "direct-effect case handled by run_path_patching. Drop 'output' from "
            "the set."
        )
    # Dedupe by the position's ``id`` (falling back to object identity for an
    # id-less position) rather than by object identity alone, so two value-equal
    # positions a direct caller built separately (e.g. two ``last_token``s) are not
    # misread as spanning distinct positions.
    position_ids = {
        getattr(rs.token_position, "id", None) or id(rs.token_position)
        for rs in receiver_specs
    }
    if len(position_ids) != 1:
        raise ValueError(
            f"All receivers in a set must share one token position — restorers freeze "
            f"at the receiver's token_position (a single position), so a set spanning "
            f"positions {position_ids} has no single union restorer set. Split the set "
            f"by position, or set the same token_position on every receiver."
        )


def _build_receiver_units(
    pipeline: LMPipeline, receiver_specs: list[ReceiverSpec]
) -> list[AtomicModelUnit]:
    """Build the set's receiver units, rejecting two specs that map to the same unit.

    Call only after :func:`_validate_receiver_set` (which rules out ``output``, so
    every unit here is non-``None``). On a grouped-query-attention model
    ``build_receiver_unit`` remaps a ``head_value_input`` query head to its KV group
    (``targets.build_receiver_unit``), so two distinct query heads in the same group
    collapse to the *same* value unit — patching the set would then place two
    collect/inject sites on one slot (a double-count). Reject
    that (and a plain duplicate spec) with an actionable error rather than misbehave
    silently. ``head_query_input`` has no remap, so query sets are unaffected.
    """
    units: list[AtomicModelUnit] = []
    by_id: dict[str, list[ReceiverSpec]] = {}
    for rs in receiver_specs:
        unit = build_receiver_unit(pipeline, rs)
        assert unit is not None  # guaranteed by _validate_receiver_set (no 'output')
        units.append(unit)
        by_id.setdefault(unit.id, []).append(rs)
    collisions = {uid: specs for uid, specs in by_id.items() if len(specs) > 1}
    if collisions:
        detail = "; ".join(
            f"{uid} <- " + ", ".join(f"L{rs.layer}H{rs.head}" for rs in specs)
            for uid, specs in collisions.items()
        )
        raise ValueError(
            f"Receiver-set members map to the same model unit ({detail}). On a "
            f"grouped-query-attention model, head_value_input query heads in the same "
            f"KV group share one value unit, so they collapse to a single "
            f"collect/inject site. Use one representative head per KV group, or "
            f"head_query_input (which has no KV remap)."
        )
    return units


def run_path_patching_set(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    sender: AtomicModelUnit,
    receiver_specs: list[ReceiverSpec],
    *,
    restore: Iterable[str] = ("attention",),
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """Path-patch one sender into a *set* of internal receivers simultaneously.

    The set is collected and injected in one shot (not summed over single receivers):
    the metric is nonlinear and the receivers interact downstream, so the joint patch
    is the faithful estimand (IOI Fig. 4 / Fig. 5). Restorers are built against the
    deepest receiver, which isolates the direct sender→R_k edge for every R_k at once
    (see :func:`~causalab.methods.path_patching.targets.deepest_receiver`).

    ``restore`` defaults to ``("attention",)`` — the Wang et al. (2022) §3.1 direct
    effect used for the IOI receiver-set figures. **Note this differs from the
    single-receiver :func:`run_path_patching` default ``("attention", "mlp")``**: a
    1-element set reproduces the single-receiver run only when both are passed the
    same ``restore`` (the estimands differ otherwise). Raises ``ValueError`` if the
    sender reaches no receiver in the set (a structural zero).
    """
    _validate_receiver_set(receiver_specs)
    deepest = deepest_receiver(pipeline, receiver_specs)
    if not sender_reaches_receiver(pipeline, sender, deepest):
        raise ValueError(
            f"Sender {sender.id!r} reaches no receiver in the set: it writes at or "
            f"after the deepest receiver's read point (layer {deepest.layer}, "
            f"{deepest.kind!r}), so no forward path connects them and the set's direct "
            f"effect is structurally zero. Pick a sender upstream of the set."
        )
    receiver_units = _build_receiver_units(pipeline, receiver_specs)
    presented = _with_base_as_second_source(counterfactual_dataset)
    return _two_pass_for_sender(
        pipeline,
        presented,
        sender,
        receiver_units,
        deepest,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching_set_scan(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    senders: dict[tuple[Any, ...], AtomicModelUnit],
    *,
    receiver_specs: list[ReceiverSpec],
    metric: InterchangeMetric,
    restore: Iterable[str] = ("attention",),
    batch_size: int = 32,
    output_scores: bool | int = True,
    causal_model: CausalModel | None = None,
    original_outputs: list[dict[str, Any]] | None = None,
) -> dict[tuple[Any, ...], float]:
    """Sweep a grid of senders against a *set* of receivers; score each with ``metric``.

    Like :func:`run_path_patching_scan` but for a simultaneous receiver set (see
    :func:`run_path_patching_set`, which also notes that ``restore`` defaults to
    ``("attention",)`` here vs ``("attention", "mlp")`` for the single-receiver
    scan). Unlike the single-receiver scan, a sender that reaches no receiver in the
    set scores ``nan`` (rather than raising), so whole structurally-disconnected
    layers drop out cleanly and the heatmap grid stays rectangular. (The analysis
    layer pre-filters unreachable senders via the same deepest-receiver gate, so the
    ``nan`` branch fires only for direct API callers sweeping an unfiltered grid.)
    """
    _validate_receiver_set(receiver_specs)
    restore = tuple(restore)  # materialize: reused per cell
    deepest = deepest_receiver(pipeline, receiver_specs)
    # The re-presented [source, base] dataset and the receiver units are both
    # sender-independent, so build them once here rather than per cell.
    receiver_units = _build_receiver_units(pipeline, receiver_specs)
    presented = _with_base_as_second_source(counterfactual_dataset)
    scores: dict[tuple[Any, ...], float] = {}
    for key, sender in tqdm(
        senders.items(), desc="path-patch set scan", total=len(senders)
    ):
        # Gate on the deepest receiver already in scope (== reaching at least one
        # member), avoiding a per-cell deepest_receiver recompute.
        if not sender_reaches_receiver(pipeline, sender, deepest):
            scores[key] = float("nan")
            continue
        outputs = _two_pass_for_sender(
            pipeline,
            presented,
            sender,
            receiver_units,
            deepest,
            restore=restore,
            batch_size=batch_size,
            output_scores=output_scores,
        )
        scores[key] = _score_cell(
            key,
            outputs,
            dataset=counterfactual_dataset,
            metric=metric,
            causal_model=causal_model,
            original_outputs=original_outputs,
        )
    return scores
