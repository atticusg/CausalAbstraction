"""Run path patching for a sender→receiver edge and score the direct effect.

Every edge lowers onto the Plan compiler
(:func:`causalab.neural.plan.run_plan` via
:func:`~causalab.methods.path_patching.plans.build_edge_plan`) — one Plan per
batch, whose named inputs and per-receiver ``ReadSource`` edges replace the
pyvene mixed model and its ``sorted_keys`` collect-order contract. Two regimes,
dispatched on the receiver:

* **``receiver = output``** (default) — the *direct effect on the logits* is the
  degenerate one-pass case: once the restorer set is frozen to the clean base,
  the only surviving sender→logits route is the one selected by ``restore`` (the
  bare residual stream, or residual + MLPs), so the metric is read straight off
  the ``base`` invoke's logits. With no restorers this is the canonical single
  fused-trace interchange.

* **internal receiver** (``head_value_input`` / ``head_query_input`` /
  ``mlp_input`` / ``residual``) — the sender's perturbation also reaches the
  output through routes that *bypass* the receiver, so a single forward would
  not isolate the edge. The plan carries the two-pass procedure explicitly: the
  ``base`` invoke collects the receiver's path-restricted activation under the
  sender+restorer patch, and the ``final`` invoke re-injects it on an otherwise
  clean run — leaving exactly the sender→receiver edge effect. The compiler
  stages the passes (see :mod:`causalab.methods.path_patching.plans`).

The scan loops a grid of senders and scores each cell with an
``InterchangeMetric`` (reuse the logit-difference metric in
:mod:`causalab.methods.metric`). Scoring reads single-step outputs
(``max_new_tokens == 1`` — refused loudly otherwise; see
:mod:`causalab.methods.path_patching.outputs`).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.methods.metric import InterchangeMetric, score_intervention_outputs
from causalab.methods.path_patching.outputs import check_single_step, plan_outputs
from causalab.methods.path_patching.plans import build_edge_plan
from causalab.methods.path_patching.targets import (
    OUTPUT,
    ReceiverSpec,
    build_receiver_site,
    deepest_receiver,
    sender_reaches_receiver,
)
from causalab.neural.head_view import HeadSite
from causalab.neural.pipeline import (
    GenerationResult,
    LMPipeline,
    compress_scores_top_k,
)
from causalab.neural.plan import run_plan
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec

logger = logging.getLogger(__name__)


def _run_edge(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    sender: SiteSpec,
    receiver_sites: Sequence[Site | HeadSite],
    range_receiver: ReceiverSpec,
    *,
    restore: Iterable[str],
    batch_size: int,
    output_scores: bool | int,
    restorer_sites: list[Site] | None = None,
) -> GenerationResult:
    """Run one sender's edge over the dataset in batches; return ONE flat result.

    The single source of edge execution, shared by the one-pass output case
    (``receiver_sites == []``) and the two-pass internal case — the plan builder
    dispatches on the receiver sites, so the two regimes cannot drift. The
    output is the flat :class:`~causalab.neural.pipeline.GenerationResult` the
    generate path produces (EU5b, #487), so scoring is shared. Concatenation
    across the internal batches is trivially regular here: every batch emits
    exactly one step (``check_single_step``), so the ragged-steps case the
    dataset engine refuses cannot arise.
    """
    batch_results: list[GenerationResult] = []
    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc="path patching",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        examples = dataset[start : start + batch_size]
        plan, logits_key = build_edge_plan(
            pipeline,
            examples,
            sender,
            receiver_sites,
            range_receiver,
            restore=restore,
            restorer_sites=restorer_sites,
        )
        with torch.no_grad():
            result = run_plan(pipeline.model, plan)
        batch_results.append(
            plan_outputs(
                pipeline, result.logits[logits_key], output_scores=output_scores
            )
        )
    flat = GenerationResult(
        sequences=torch.cat([r.sequences for r in batch_results], dim=0)
        if batch_results
        else torch.empty((0, pipeline.max_new_tokens), dtype=torch.long),
        strings=[s for r in batch_results for s in r.strings],
        scores=[torch.cat([r.scores[0] for r in batch_results], dim=0)]  # type: ignore[index]  # plan_outputs sets scores iff output_scores
        if output_scores and batch_results
        else ([] if output_scores else None),
    )
    if not isinstance(output_scores, bool) and output_scores > 0:
        flat = compress_scores_top_k(flat, pipeline, k=output_scores)
    return flat


def _score_cell(
    key: tuple[Any, ...],
    outputs: GenerationResult,
    *,
    dataset: list[CounterfactualExample],
    metric: InterchangeMetric,
    causal_model: CausalModel | None,
    original_outputs: list[dict[str, Any]] | None,
) -> float:
    """Score one swept cell's result with ``metric``.

    Shared by both scans so the (identical) scoring call cannot drift between them.
    """
    return score_intervention_outputs(
        results={key: outputs},
        dataset=dataset,
        metric=metric,
        causal_model=causal_model,
        original_outputs=original_outputs,
    )[key]


def _run_for_sender(
    pipeline: LMPipeline,
    dataset: list[CounterfactualExample],
    sender: SiteSpec,
    receiver: ReceiverSpec,
    receiver_site: Site | HeadSite | None,
    *,
    restore: Iterable[str],
    batch_size: int,
    output_scores: bool | int,
) -> GenerationResult:
    """Path-patch one sender against ``receiver``; return its flat result.

    ``receiver_site`` is the (sender-independent) resolved receiver location, so
    the scan builds it once and threads it in here.

    Raises ``ValueError`` for an internal receiver the sender cannot reach (sender
    at or downstream of the receiver's read point) — there is no forward path, so
    the run would only confirm a structural zero. ``output`` receivers are always
    reachable.
    """
    if not sender_reaches_receiver(pipeline, sender, receiver):
        raise ValueError(
            f"Sender {sender.key!r} is not upstream of receiver "
            f"{receiver.kind!r}@layer={receiver.layer}: it writes at or after the "
            f"receiver's read point, so no forward path connects them and the direct "
            f"effect is structurally zero. Pick a sender upstream of the receiver."
        )
    receiver_sites = [] if receiver_site is None else [receiver_site]
    return _run_edge(
        pipeline,
        dataset,
        sender,
        receiver_sites,
        receiver,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    sender: SiteSpec,
    receiver: ReceiverSpec = OUTPUT,
    *,
    restore: Iterable[str] = ("attention", "mlp"),
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> GenerationResult:
    """Path-patch the edge ``sender → receiver``; return its flat result.

    ``receiver`` defaults to the output logits (the IOI Fig. 3 head-sweep case).
    ``restore`` selects which component families are frozen between the sender and
    the receiver (see
    :func:`~causalab.methods.path_patching.targets.build_restorer_sites`). The
    flat :class:`~causalab.neural.pipeline.GenerationResult` (EU5b, #487) is
    scored by the caller (e.g. via :func:`run_path_patching_scan` or
    ``score_intervention_outputs``).
    """
    check_single_step(pipeline)
    receiver_site = build_receiver_site(pipeline, receiver)
    return _run_for_sender(
        pipeline,
        counterfactual_dataset,
        sender,
        receiver,
        receiver_site,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching_scan(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    senders: dict[tuple[Any, ...], SiteSpec],
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

    ``senders`` maps a grid key (e.g. ``(layer, head)``) to the sender spec.
    Returns ``{key: score}`` ready for a layer×head heatmap. Each cell is
    generated then scored immediately, so only one cell's logits stay alive (the
    reason full-vocab ``output_scores`` is affordable).
    """
    check_single_step(pipeline)
    restore = tuple(restore)  # materialize: reused per cell
    # The receiver site is sender-independent, so build it once here.
    receiver_site = build_receiver_site(pipeline, receiver)
    scores: dict[tuple[Any, ...], float] = {}
    for key, sender in tqdm(
        senders.items(), desc="path-patch scan", total=len(senders)
    ):
        outputs = _run_for_sender(
            pipeline,
            counterfactual_dataset,
            sender,
            receiver,
            receiver_site,
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
    token position — restorers and receivers freeze/read at the set's
    ``token_position`` (a single position to stay one-token-per-position), so a
    set spanning positions has no single union restorer set. The IOI Fig. 4 /
    Fig. 5 sets satisfy both (END for name-mover queries, S2 for S-inhibition
    values).
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


def _build_receiver_sites(
    pipeline: LMPipeline, receiver_specs: list[ReceiverSpec]
) -> list[Site | HeadSite]:
    """Resolve the set's receiver sites, rejecting two specs that map to the same
    location.

    Call only after :func:`_validate_receiver_set` (which rules out ``output``, so
    every site here is non-``None``). On a grouped-query-attention model
    ``build_receiver_site`` remaps a ``head_value_input`` query head to its KV
    group, so two distinct query heads in the same group collapse to the *same*
    value site — patching the set would then place two collect/inject edges on one
    slot (a double-write). Reject that (and a plain duplicate spec) with an
    actionable error rather than misbehave silently. ``head_query_input`` has no
    remap, so query sets are unaffected.
    """
    sites: list[Site | HeadSite] = []
    by_site: dict[Site | HeadSite, list[ReceiverSpec]] = {}
    for rs in receiver_specs:
        site = build_receiver_site(pipeline, rs)
        assert site is not None  # guaranteed by _validate_receiver_set (no 'output')
        sites.append(site)
        by_site.setdefault(site, []).append(rs)
    collisions = {site: specs for site, specs in by_site.items() if len(specs) > 1}
    if collisions:
        detail = "; ".join(
            f"{site!r} <- " + ", ".join(f"L{rs.layer}H{rs.head}" for rs in specs)
            for site, specs in collisions.items()
        )
        raise ValueError(
            f"Receiver-set members map to the same model unit ({detail}). On a "
            f"grouped-query-attention model, head_value_input query heads in the same "
            f"KV group share one value site, so they collapse to a single "
            f"collect/inject edge. Use one representative head per KV group, or "
            f"head_query_input (which has no KV remap)."
        )
    return sites


def run_path_patching_set(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    sender: SiteSpec,
    receiver_specs: list[ReceiverSpec],
    *,
    restore: Iterable[str] = ("attention",),
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> GenerationResult:
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
    check_single_step(pipeline)
    _validate_receiver_set(receiver_specs)
    deepest = deepest_receiver(pipeline, receiver_specs)
    if not sender_reaches_receiver(pipeline, sender, deepest):
        raise ValueError(
            f"Sender {sender.key!r} reaches no receiver in the set: it writes at or "
            f"after the deepest receiver's read point (layer {deepest.layer}, "
            f"{deepest.kind!r}), so no forward path connects them and the set's direct "
            f"effect is structurally zero. Pick a sender upstream of the set."
        )
    receiver_sites = _build_receiver_sites(pipeline, receiver_specs)
    return _run_edge(
        pipeline,
        counterfactual_dataset,
        sender,
        receiver_sites,
        deepest,
        restore=restore,
        batch_size=batch_size,
        output_scores=output_scores,
    )


def run_path_patching_set_scan(
    pipeline: LMPipeline,
    counterfactual_dataset: list[CounterfactualExample],
    senders: dict[tuple[Any, ...], SiteSpec],
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
    check_single_step(pipeline)
    _validate_receiver_set(receiver_specs)
    restore = tuple(restore)  # materialize: reused per cell
    deepest = deepest_receiver(pipeline, receiver_specs)
    # The receiver sites are sender-independent, so build them once here.
    receiver_sites = _build_receiver_sites(pipeline, receiver_specs)
    scores: dict[tuple[Any, ...], float] = {}
    for key, sender in tqdm(
        senders.items(), desc="path-patch set scan", total=len(senders)
    ):
        # Gate on the deepest receiver already in scope (== reaching at least one
        # member), avoiding a per-cell deepest_receiver recompute.
        if not sender_reaches_receiver(pipeline, sender, deepest):
            scores[key] = float("nan")
            continue
        outputs = _run_edge(
            pipeline,
            counterfactual_dataset,
            sender,
            receiver_sites,
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
