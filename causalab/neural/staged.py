"""The one scheduler + one executor for every plan — EU2 + EU3.

THE lowering of the Plan IR (:mod:`causalab.neural.plan`) — the
engine-unification consolidation, #480/#483/#484: every plan is scheduled by
:func:`lower_plan` into a :class:`StagedProgram` — stages of trace groups of
invokes, plus an optional terminal generate invoke — and executed by one
executor (:func:`run_plan_staged` / :func:`_run_trace_group`). A plan that
fits one fused forward schedules as the *degenerate* program (one stage, one
group) and runs exactly the single fused multi-invoke trace the retired
single-trace compiler (PL1's ``_run_single_trace``) emitted; a plan that
cannot pays stage boundaries — values saved in an earlier trace feed later
traces as concrete constants. ``run_plan(lowering="single")`` is a
strictness assertion on top of the same schedule: it refuses
(:class:`~causalab.neural.plan.StagingRequired`, from schedule facts —
:func:`_refuse_not_single`) iff the program needs more than one trace.
The scheduler handles exactly the five staging seams, with the semantics
each one wants:

* **Cross-input flow against forward order** (read the source at layer 10 to
  write the base at layer 2): the producing input runs in an earlier trace.
* **Chained cross-input flow** (A → B → C, any DAG): inputs are layered by
  longest path; each link either shares its producer's trace (forward-order
  edges — the canonical PL1 barrier recipe) or crosses a stage boundary as a
  saved value. Only a genuine **cycle** is refused (no trace order can
  deliver both directions; break it by declaring a duplicate input).
* **A same-input read firing after the written site** (the two-pass
  path-patching / self-grafting shape): the read reroutes to a hidden
  ``(input, "clean")`` pass — an extra trace over the same tensors — so the
  value is the input's *clean* activation, the causal-tracing convention.
  A read under interventions is expressed explicitly instead: declare the
  same tensors twice (``inputs={"pass1": x, "pass2": x}``) and put the
  stage-one edits on ``pass1``.
* **Mixed padded frames**: separate traces have separate frames, so nothing
  shifts and no alignment is required (each input's resolved positions stay
  in its own frame).
* **Cross-model patching** (PL4, #406): :attr:`~causalab.neural.plan.Plan.models`
  binds an input to the model its traces run on — capture in the *source*
  model's forward, inject into the target's. Two models never share a fused
  forward, so a cross-model edge always crosses a stage boundary as a saved
  value; the executor runs each trace on its input's model.
* **Generation as a terminal stage kind** (EU3, #484): a generation plan
  (:attr:`~causalab.neural.plan.Plan.generate`) lowers to the plain collect
  stages *plus* ONE terminal ``model.generate`` invoke
  (:attr:`StagedProgram.generate_key` — never inside ``stages``). A generate
  trace accepts only constants, so :func:`_generate_forcing` seeds every
  edge whose consumer is the generate invoke into :func:`_schedule`'s
  ``forced`` mapping (reason ``"generate-with-variable-intervention"``,
  checked before any fusability rule — a frame-aligned forward-rank edge
  that would ride the canonical barrier still stages): collect stages run
  first, their produce saves materialize through the existing slots + lazy
  ``aux_get`` + ``FeaturizedSite._coerce`` machinery, and the generate
  trace consumes them as saved values — the same path that carries
  cross-model constants. The executor emits the generate trace last, via
  the one generate emitter
  (:func:`~causalab.neural.plan._emit_generate_trace`), NEVER inside a session.

**Pass minimization.** Stage layering is longest-path over the cross-invoke
dependency edges, where an edge is free (same trace, the PL1 barrier recipe)
when it runs forward in rank between frame-aligned inputs whose producer is
not itself an in-trace consumer (one producers → consumers phase per trace,
the measured nnsight constraint), and costs one stage boundary otherwise.
Within a stage, connected in-trace edges form one trace group; a group whose
single rendezvous cannot serve every edge (latest signal after earliest
wait) is dissolved into staged edges instead. The canonical source → base
interchange therefore stays a single fused trace, and
:func:`~causalab.neural.plan.run_plan` (``lowering="auto"``) only ever pays
a second pass when one is semantically required. Sequential staged traces
are also the memory-light lowering (one forward resident at a time) — and
the natural form for cross-model patching, which is why PL4 lands here:
saved values are concrete tensors between locally-executed traces, which is
all a stage boundary needs, for single-model and cross-model plans alike
(remote/mixed execution is out of scope — the design doc's §6 boundary).

**No session.** The executor never wraps its traces in ``model.session()``
(EU1, #482): plain sequential traces are the only path. The session wrapper
was measured time-free and benefit-free — values cross traces as concrete
saved tensors either way, so its deferred layer bought nothing — while
leaving the ``tracer.iter`` step-counter and trace-body-abandonment
behavior of generation under a session unmeasured (and the unified engine
stages generation plans, EU3 #484). Measured hazard (nnsight 0.7):
a session context *defers its block body* — a fresh local assigned inside
``with model.session():`` never binds in the enclosing frame
(``UnboundLocalError``); the removed executor survived only because it
mutated a pre-existing ``logits`` dict inside the session body.
Re-introduction gate: a future PR that
brings ``model.session()`` back must first (a) pin step-counter integrity
and body-abandonment behavior under a session, (b) demonstrate a measured
GPU wall-clock win on a real workload, and (c) NEVER wrap a generate trace
(``Plan.generate`` lowers to ONE ``model.generate`` trace, outside any
session, always).

``OutOfOrderError`` guards: within every emitted trace, taps stay sorted by
``(layer, forward_rank_on(model), declaration order)`` (PL1's invariant), and
every shape that would deadlock or ``MissedProviderError`` a fused forward is
either staged away or refused up front with an actionable error — nnsight's
own out-of-order failures should be unreachable through this compiler.

Gradient requests lower like any plan (``lower_plan`` threads the
grad-leaves through :class:`LoweredPlan`), but *execute* only as the
degenerate schedule: :func:`~causalab.neural.plan.run_plan` gates them on
schedule shape (exactly one trace of one invoke — trace count is the key)
and diverts them to the single-trace gradient path
(:func:`~causalab.neural.plan._run_gradient_trace`); the plain executor
here never runs them (a backward across staged traces has no consumer yet)
and refuses them up front (``NotImplementedError``). A gradient plan never
carries a generate stage (gradients × generate is refused at construction).
Every refusal fires before any forward pass.
:func:`lower_staged` exposes the schedule (stages → trace groups → invokes)
without executing — the structural surface the tests pin pass-minimality on;
:func:`lower_plan` is the full lowering (schedule + taps + grad leaves +
per-edge ``staged_why`` reasons).

Design + as-landed record: ``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 5
("engine unification", #480) — including the refusal relocation map with
final anchors and the no-session re-introduction gate.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from nnterp import StandardizedTransformer

from causalab.neural.pipeline import ensure_position_ids
from causalab.neural.plan import (
    InvokeKey,
    Plan,
    PlanResult,
    StagingRequired,
    _build_taps,
    _check_generate_inputs,
    _Edge,
    _emit_generate_trace,
    _emit_invokes,
    _frame_of,
    _model_resolver,
    _slice_fused_logits,
    _stop_carrier,
    _Tap,
)

__all__ = [
    "LoweredPlan",
    "StagedProgram",
    "lower_plan",
    "lower_staged",
    "run_plan_staged",
]

#: The per-edge staging reasons :func:`_schedule` records in
#: :attr:`StagedProgram.staged_why` — the vocabulary strict-mode refusal
#: messages (:func:`_refuse_not_single`) are assembled from.
#: ``"generate-with-variable-intervention"`` (EU3, #484) is seeded through
#: :func:`_schedule`'s ``forced`` mapping by :func:`_generate_forcing`: an
#: edge whose consumer is the terminal generate invoke always stages,
#: whatever the fusability rules would have said.
_STAGED_WHY = (
    "generate-with-variable-intervention",
    "intervene-backwards",
    "cross-model",
    "variable-token-positions",
    "chain-across-invokes",
    "separate-concurrent-interventions",
)


@dataclasses.dataclass(frozen=True)
class StagedProgram:
    """The staged schedule: ``stages`` (run sequentially) of trace ``groups``
    (independent traces) of invoke keys (one ``tracer.invoke`` each, in
    emission order — in-trace producers first). ``in_trace`` holds the edges
    served by a barrier inside a group's trace; every other edge crosses a
    stage boundary as a saved value. ``staged_why`` maps each edge staged *by
    rule* to its reason (one of :data:`_STAGED_WHY`; edges staged only as a
    consequence of their endpoints' layering carry no entry) — the schedule
    facts ``run_plan(lowering="single")`` strictness messages are assembled
    from. ``generate_key`` is the ONE terminal ``model.generate`` invoke of a
    generation plan (EU3, #484) — never inside ``stages``: the executor runs
    every plain stage first (materializing the force-staged constants), then
    emits the generate trace last."""

    stages: tuple[tuple[tuple[InvokeKey, ...], ...], ...]
    in_trace: frozenset[_Edge]
    staged_why: Mapping[_Edge, str] = dataclasses.field(default_factory=dict)
    generate_key: InvokeKey | None = None

    @property
    def num_traces(self) -> int:
        """Every trace the program runs — the plain stage groups plus the
        terminal generate trace, when there is one. The schedule-shape
        currency: the gradient gate and ``lowering="single"`` strictness both
        key on it (a generation plan with no cross-input reads is one trace;
        each collect stage adds one)."""
        return sum(len(stage) for stage in self.stages) + (
            0 if self.generate_key is None else 1
        )


@dataclasses.dataclass(frozen=True)
class LoweredPlan:
    """:func:`lower_plan`'s output — the schedule plus everything the
    executor needs to run it. ``taps`` are the per-invoke in-trace actions;
    ``collects`` is the **live** dict the collect taps fill during execution
    (:attr:`~causalab.neural.plan.PlanResult.collects` aliases it);
    ``edges`` are all cross-invoke data dependencies (in-trace ones served by
    a barrier, per :attr:`StagedProgram.in_trace`); ``grad_leaves`` is the
    live dict a gradient plan's ``wrt`` taps fill with autograd leaves — the
    4th ``_build_taps`` element, threaded through so the post-trace backward
    (:func:`~causalab.neural.plan._execute_gradients`) can read ``.grad``
    off the raw site activations."""

    program: StagedProgram
    taps: Mapping[InvokeKey, list[_Tap]]
    collects: dict[str, Any]
    edges: tuple[_Edge, ...]
    grad_leaves: dict[str, Any]


def _input_of(plan: Plan, key: InvokeKey) -> Any:
    """The model input an invoke key runs — an alias ``(input, "clean")``
    runs its original's tensors."""
    return plan.inputs[key if isinstance(key, str) else key[0]]


def _toposort(nodes: Sequence[InvokeKey], edges: Sequence[_Edge]) -> list[InvokeKey]:
    """Kahn topological order over the invoke-level dependency graph; a cycle
    is unsatisfiable under any staging (each input runs once, so values flow
    one way between two inputs) and raises with the break-the-cycle recipe."""
    indegree: dict[InvokeKey, int] = {n: 0 for n in nodes}
    out: dict[InvokeKey, list[InvokeKey]] = defaultdict(list)
    for e in edges:
        if e.src in indegree and e.dst in indegree:
            indegree[e.dst] += 1
            out[e.src].append(e.dst)
    ready = [n for n in nodes if indegree[n] == 0]
    order: list[InvokeKey] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in out[n]:
            indegree[m] -= 1
            if indegree[m] == 0:
                ready.append(m)
    if len(order) != len(nodes):
        cyclic = sorted(str(n) for n in nodes if n not in set(order))
        raise ValueError(
            f"cyclic cross-input flow among inputs {cyclic} — each input runs "
            "once, so no trace order can deliver values in both directions. "
            "Break the cycle by declaring one input twice (e.g. "
            "inputs={'x': t, 'x2': t}) so one direction reads the extra pass."
        )
    return order


def _schedule(
    plan: Plan,
    nodes: Sequence[InvokeKey],
    edges: Sequence[_Edge],
    model_of: Any,
    forced: Mapping[_Edge, str] | None = None,
) -> tuple[dict[InvokeKey, int], set[_Edge], dict[_Edge, str]]:
    """Assign each invoke a stage and each edge in-trace or cross-stage;
    return ``(stage-of-invoke, in-trace edges, staged_why)``.

    Longest-path layering in topological order. An edge rides in-trace (cost
    0 — producer and consumer share one fused trace and a barrier) iff it is
    not force-staged, runs forward in rank, joins frame-aligned inputs on the
    same model (two models cannot share a fused forward — PL4), and its
    producer is not itself an in-trace consumer (one producers → consumers
    phase per trace). After layering, only edges whose endpoints actually
    landed in the same stage stay in-trace. Groups whose single rendezvous
    cannot serve every edge (latest signal after earliest wait — the
    measured ``MissedProviderError`` shape) have their edges force-staged
    (reason ``"separate-concurrent-interventions"``) and the layering
    reruns; each rerun forces at least one edge, so this terminates.

    ``staged_why`` records, per edge staged *by rule*, the first failing
    fusability test — ``"intervene-backwards"`` (read fires after the write
    in forward rank), ``"cross-model"``, ``"variable-token-positions"``
    (unknowable or mismatched padded frames), ``"chain-across-invokes"`` (the
    producer already consumes in-trace — the one-phase-per-trace
    constraint), or the ``forced`` reason. ``forced`` pre-seeds force-staged
    edges with their reasons — the hook the terminal generate stage
    (EU3 #484, ``_generate_forcing``) schedules through.
    """
    order = _toposort(nodes, edges)
    incoming: dict[InvokeKey, list[_Edge]] = defaultdict(list)
    for e in edges:
        incoming[e.dst].append(e)
    force_why: dict[_Edge, str] = dict(forced or {})

    def frames_align(e: _Edge) -> bool:
        fa = _frame_of(_input_of(plan, e.src))
        fb = _frame_of(_input_of(plan, e.dst))
        return fa is not None and fb is not None and fa[0] == fb[0]

    while True:
        stage: dict[InvokeKey, int] = {}
        consumes_in_trace: dict[InvokeKey, bool] = defaultdict(bool)
        in_trace: set[_Edge] = set()
        staged_why: dict[_Edge, str] = {}
        for node in order:
            why = {
                e: (
                    force_why.get(e)
                    or ("intervene-backwards" if not e.src_hook <= e.dst_hook else None)
                    or (
                        "cross-model"
                        if model_of(e.src) is not model_of(e.dst)
                        else None
                    )
                    or ("variable-token-positions" if not frames_align(e) else None)
                    or ("chain-across-invokes" if consumes_in_trace[e.src] else None)
                )
                for e in incoming[node]
            }
            stage[node] = max(
                (stage[e.src] + (0 if w is None else 1) for e, w in why.items()),
                default=0,
            )
            for e, w in why.items():
                if w is None and stage[e.src] == stage[node]:
                    in_trace.add(e)
                    consumes_in_trace[node] = True
                elif w is not None:
                    staged_why[e] = w

        conflict = _rendezvous_conflict(stage, in_trace)
        if conflict is None:
            return stage, in_trace, staged_why
        force_why.update({e: "separate-concurrent-interventions" for e in conflict})


def _rendezvous_conflict(
    stage: Mapping[InvokeKey, int], in_trace: set[_Edge]
) -> set[_Edge] | None:
    """The in-trace edges of the first group whose one barrier cannot serve
    every edge, or ``None``. A group = connected component of in-trace edges
    within one stage; its rendezvous works iff the latest produce hook is at
    or before the earliest consume hook (PL1's global condition)."""
    parent: dict[InvokeKey, InvokeKey] = {}

    def find(x: InvokeKey) -> InvokeKey:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in in_trace:
        parent[find(e.src)] = find(e.dst)
    by_group: dict[InvokeKey, set[_Edge]] = defaultdict(set)
    for e in in_trace:
        by_group[find(e.src)].add(e)
    for group_edges in by_group.values():
        signals = defaultdict(list)
        waits = defaultdict(list)
        for e in group_edges:
            signals[e.src].append(e.src_hook)
            waits[e.dst].append(e.dst_hook)
        latest_signal = max(max(hooks) for hooks in signals.values())
        earliest_wait = min(min(hooks) for hooks in waits.values())
        if latest_signal > earliest_wait:
            return set(group_edges)
    return None


def _grouped(
    ordered_nodes: Sequence[InvokeKey],
    stage: Mapping[InvokeKey, int],
    in_trace: set[_Edge],
) -> tuple[tuple[tuple[InvokeKey, ...], ...], ...]:
    """Materialize the schedule: per stage, connected in-trace components run
    as one fused trace (producers first — every group is one producers →
    consumers phase by construction) and every other invoke as its own
    trace. Deterministic: groups and their members follow ``ordered_nodes``
    (topological) order.

    Per-component scheduling has one observable consequence for *stateful*
    edits: a plan with ``SeededNoise`` edits at two or more layers on two or
    more **disconnected** inputs sharing one noise stream draws in
    per-input-trace order rather than the retired fused trace's layer-major
    order — different values for that (unshipped, unpinned, but expressible)
    shape. Single-input and edge-connected plans are unaffected."""
    peers: dict[InvokeKey, set[InvokeKey]] = defaultdict(set)
    for e in in_trace:
        peers[e.src].add(e.dst)
        peers[e.dst].add(e.src)
    consumers = {e.dst for e in in_trace}

    stages: dict[int, list[tuple[InvokeKey, ...]]] = defaultdict(list)
    seen: set[InvokeKey] = set()
    for node in ordered_nodes:
        if node in seen:
            continue
        component = [node]
        seen.add(node)
        i = 0
        while i < len(component):
            for peer in sorted(peers[component[i]], key=ordered_nodes.index):
                if peer not in seen:
                    seen.add(peer)
                    component.append(peer)
            i += 1
        component.sort(key=lambda k: (k in consumers, ordered_nodes.index(k)))
        stages[stage[node]].append(tuple(component))
    return tuple(tuple(stages[level]) for level in sorted(stages) if stages[level])


def _generate_invoke(plan: Plan) -> str:
    """The ONE plan input a generation plan's ops address — the terminal
    ``model.generate`` invoke (:attr:`StagedProgram.generate_key`). Raises
    the narrowed ≠1-input refusal (EU3, #484): *ops* must address ONE input
    (the generated one); reads of **other** inputs are legal — they go
    through ``read_sources`` and are force-staged into earlier collect
    stages. Model-free (reads op fields only)."""
    active = sorted({op.input for op in plan.ops})
    if len(active) != 1:
        raise NotImplementedError(
            f"a generation plan's ops must address ONE input — the one that "
            f"runs through the terminal model.generate trace — but ops "
            f"address {active if active else 'no input'}. Reads of other "
            "inputs go through read_sources (ReadSource(..., input=...)): "
            "the scheduler captures them in earlier collect stages and the "
            "generate trace consumes them as constants. Run one generation "
            "plan per generated input."
        )
    return active[0]


def _generate_forcing(plan: Plan, edges: Sequence[_Edge]) -> dict[_Edge, str]:
    """The terminal-generate forcing (EU3, #484): every edge whose consumer
    is the generate invoke is seeded into :func:`_schedule`'s ``forced``
    mapping (reason ``"generate-with-variable-intervention"``) — a generate
    trace accepts only constants, so no edge may ride into it on a barrier,
    whatever the fusability rules would have said (forcing is checked *before*
    ``frames_align`` and the rank tests, so even the canonical frame-aligned
    forward-rank interchange stages). The producers then run as earlier
    collect stages and their produce ``.save()``s enter the generate trace
    through the existing slots + lazy ``aux_get`` +
    ``FeaturizedSite._coerce`` machinery — zero new materialization code."""
    key = _generate_invoke(plan)
    return {e: "generate-with-variable-intervention" for e in edges if e.dst == key}


def lower_plan(model: StandardizedTransformer, plan: Plan) -> LoweredPlan:
    """Schedule ``plan`` — the ONE scheduler every plan goes through
    (EU2 #483; generation plans since EU3 #484).

    Always builds the superset taps (``_build_taps``: the clean-invoke
    reroute for same-input backward reads, read-tap ``.save()``s —
    value-identical to the retired ``staged=False`` build where both
    applied, per ``TestStagedSemantics``), then layers the invokes
    (:func:`_schedule`), recording per-edge ``staged_why`` reasons.
    Threads ``grad_leaves`` (the 4th ``_build_taps`` element) so gradient
    plans lower losslessly — forgetting it would silently break the
    post-trace backward (``TestGradientPlansMatchOracle`` catches).

    A generation plan (``Plan.generate``) schedules like any other, with
    its ops-addressed input as the terminal generate invoke: every edge
    into it is force-staged (:func:`_generate_forcing`), so it always
    schedules as its own singleton group — pulled out of ``stages`` into
    :attr:`StagedProgram.generate_key`, which the executor emits last.
    """
    model_of = _model_resolver(model, plan)
    taps, collects, edges, grad_leaves = _build_taps(model_of, plan)
    active = [
        k for k in taps if taps[k] or (isinstance(k, str) and k in plan.save_logits)
    ]
    generate_key: InvokeKey | None = None
    forced: dict[_Edge, str] | None = None
    if plan.generate is not None:
        generate_key = _generate_invoke(plan)
        forced = _generate_forcing(plan, edges)
    stage, in_trace, staged_why = _schedule(plan, active, edges, model_of, forced)
    stages = _grouped(active, stage, in_trace)
    if generate_key is not None:
        # Every edge into the generate invoke is forced (never in-trace) and
        # it produces nothing, so it grouped as a singleton — pull it out of
        # the plain stages (a stage emptied by the pull disappears).
        assert all(
            (generate_key in group) == (group == (generate_key,))
            for stage_groups in stages
            for group in stage_groups
        ), "the generate invoke must schedule as its own singleton group"
        stages = tuple(
            kept
            for kept in (
                tuple(g for g in stage_groups if g != (generate_key,))
                for stage_groups in stages
            )
            if kept
        )
    program = StagedProgram(
        stages=stages,
        in_trace=frozenset(in_trace),
        staged_why=staged_why,
        generate_key=generate_key,
    )
    return LoweredPlan(
        program=program,
        taps=taps,
        collects=collects,
        edges=tuple(edges),
        grad_leaves=grad_leaves,
    )


def lower_staged(model: StandardizedTransformer, plan: Plan) -> StagedProgram:
    """Schedule ``plan`` without executing it — the structural view (stages
    → trace groups → invokes) tests pin pass-minimality on; the structural
    alias of :func:`lower_plan`. Whole-component sites rank
    model-independently, so scheduling-only callers may pass ``model=None``;
    per-head sites resolve their rank through the model."""
    return lower_plan(model, plan).program


def _refuse_not_single(
    model: StandardizedTransformer, plan: Plan, lowered: LoweredPlan
) -> None:
    """The ``lowering="single"`` strictness gate: raise
    :class:`~causalab.neural.plan.StagingRequired` iff the schedule needs
    more than one trace — a schedule *fact*, not a separate compiler's
    refusal. The message is assembled from :attr:`StagedProgram.staged_why`
    (plus group-level facts for edge-less multi-trace schedules), preserving
    each retired single-trace refusal's key phrase — "two passes",
    "backward in time", "padded lengths", "pre-tokenized",
    "chained cross-input flow", "bound to a different model" — plus the
    ``"generate-with-variable-intervention"`` arm (EU3, #484): a generate
    trace accepts only constants — its reads cost collect stages by construction."""
    program = lowered.program
    if program.num_traces <= 1:
        return
    model_of = _model_resolver(model, plan)
    reasons: list[str] = []

    def add(msg: str) -> None:
        if msg not in reasons:
            reasons.append(msg)

    dissolved = "separate-concurrent-interventions" in program.staged_why.values()
    for e in lowered.edges:
        why = program.staged_why.get(e)
        if why == "generate-with-variable-intervention" and isinstance(e.src, str):
            add(
                f"ops[{e.slot[0]}] reads input {e.src!r} from inside the "
                "terminal generate trace — a generate trace accepts only "
                "constants, so the read is captured in an earlier collect "
                "stage and delivered as a saved value"
            )
        elif why == "generate-with-variable-intervention":
            add(
                f"ops[{e.slot[0]}] reads a site of the generated input "
                f"{e.dst!r} after the written site — the read reroutes to "
                "the input's clean prefill pass, an earlier collect stage "
                "over the same tensors, and the terminal generate trace "
                "consumes the value as a saved constant"
            )
        elif why == "intervene-backwards" and not isinstance(e.src, str):
            add(
                f"ops[{e.slot[0]}] reads a site at forward rank {e.src_hook} to "
                f"write an earlier one at rank {e.dst_hook} on the same input "
                f"{e.dst!r} — reading a later site to write an earlier one "
                "needs two passes over that input"
            )
        elif why == "intervene-backwards":
            add(
                f"a cross-input read at forward position {e.src_hook} "
                f"(input {e.src!r}) feeds a write at earlier position "
                f"{e.dst_hook} (input {e.dst!r}) — the fused forward cannot "
                "deliver a value backward in time through one barrier"
            )
        elif why == "cross-model":
            add(
                f"inputs {sorted({_key_input_name(e.src), e.dst})} are bound "
                "to a different model (Plan.models) — two models cannot share "
                "one fused forward"
            )
        elif why == "variable-token-positions":
            fa = _frame_of(_input_of(plan, e.src))
            fb = _frame_of(_input_of(plan, e.dst))
            if fa is None or fb is None:
                add(
                    "fusing multiple inputs into one trace needs pre-tokenized "
                    "inputs (a mapping with 'input_ids') — pass "
                    "pipeline.load(...) output"
                )
            else:
                add(
                    f"inputs {[_key_input_name(e.src), e.dst]} have different "
                    f"padded lengths {[fa[0], fb[0]]} — one left-padded fused "
                    "forward would shift the shorter frames out from under "
                    "their resolved positions"
                )
        elif why == "chain-across-invokes":
            add(
                f"input {_key_input_name(e.src)!r} both produces cross-input "
                "reads and consumes them — chained cross-input flow; one "
                "trace lowers a single producers → consumers phase"
            )
    if dissolved:
        add(
            "the plan's cross-input reads cannot share one rendezvous (the "
            "latest read fires after the earliest consuming write) — the "
            "fused forward cannot deliver a value backward in time through "
            "one barrier"
        )
    # Input-level chained flow: an input that is both an edge producer and an
    # edge consumer never fits one producers → consumers phase, whatever the
    # per-edge reasons said (they may name frames/model facts instead — this
    # supplement is skipped when a chain-across-invokes edge already spoke).
    srcs = {e.src for e in lowered.edges if isinstance(e.src, str)}
    chained = sorted(srcs & {e.dst for e in lowered.edges})
    if chained and not any(
        program.staged_why.get(e) == "chain-across-invokes" for e in lowered.edges
    ):
        add(
            f"inputs {chained} both produce cross-input reads and consume "
            "them — chained cross-input flow; one trace lowers a single "
            "producers → consumers phase"
        )
    # Multi-trace schedules with no rule-staged edge: disconnected components
    # run one trace group each (grouping is connected-components) — a group
    # may still have in-trace edges *inside* it, so the schedule facts live
    # on the groups, not on any staged edge.
    if not program.staged_why and any(len(stage) > 1 for stage in program.stages):
        group_reps = [g[0] for stage in program.stages for g in stage]
        if len({id(model_of(k)) for k in group_reps}) > 1:
            bound = sorted(
                k for k in group_reps if isinstance(k, str) and model_of(k) is not model
            )
            add(
                f"inputs {bound} are bound to a different model (Plan.models) "
                "— two models cannot share one fused forward"
            )
        elif any(
            _frame_of(_input_of(plan, k)) is None
            for k in group_reps
            if isinstance(k, str)
        ):
            add(
                "fusing multiple inputs into one trace needs pre-tokenized "
                "inputs (a mapping with 'input_ids') — pass "
                "pipeline.load(...) output"
            )
        else:
            add(
                "no cross-input data flow connects the scheduled groups — "
                "disconnected components run one trace group each (grouping "
                "is connected-components; the per-group traces are "
                "value-identical to a fused forward)"
            )
    if not reasons:  # pragma: no cover — every >1-trace schedule has a cause
        add("the schedule needs more than one trace")
    raise StagingRequired(
        f"this plan cannot lower to ONE fused trace — it schedules as "
        f"{program.num_traces} sequential traces: " + "; ".join(reasons) + ". "
        "lowering='single' asserts the degenerate (one-trace) schedule; run "
        "with lowering='auto' (the default) to execute the staged schedule."
    )


def _key_input_name(key: InvokeKey) -> str:
    """The plan-input name behind an invoke key (a ``(input, "clean")``
    alias reads its original's tensors)."""
    return key if isinstance(key, str) else key[0]


def _run_trace_group(
    group_model: StandardizedTransformer,
    plan: Plan,
    group: tuple[InvokeKey, ...],
    taps: Mapping[InvokeKey, list[_Tap]],
    in_trace: frozenset[_Edge],
    logits: dict[str, torch.Tensor],
) -> None:
    """ONE trace for one schedule group — the single executor body every
    plain plan runs through (EU2, #483: this merges the retired
    single-trace compiler's fast path with the staged executor's).

    A single-invoke group runs ``model.trace(inputs)`` directly (no invoke
    machinery); a multi-invoke group runs the fused multi-invoke trace with
    one barrier over its in-trace edges (:func:`~causalab.neural.plan.
    _emit_invokes` — PL1's recipe). Logits for requested inputs land in
    ``logits`` (sliced from the fused batch in grouped traces). A trace that
    saves no logits stops its forward after its deepest tap (CAP6, #459 —
    ``_stop_carrier`` is the single may-I-stop authority: it withholds the
    stop under persistent edits, CAP7 #460, whose mediators a truncated
    forward would strand; produce saves are taps, so cross-stage values
    land before the stop).
    """
    if len(group) == 1:
        key = group[0]
        inputs = _input_of(plan, key)
        if plan.generate is not None and not isinstance(key, str):
            # The hidden clean-prefill pass of a generation plan (the
            # ``(input, "clean")`` alias, EU3 #484) stands in for HF
            # ``generate``'s prefill, which numbers positions pad-aware from
            # the attention mask — so this plain forward must too: bare, it
            # would default to pad-blind ``arange`` positions and, on a
            # left-padded batch of an absolute-position model, graft a value
            # that is NOT the model's actual clean prefill activation.
            # (``_check_generate_inputs`` refuses explicit ``position_ids``
            # on the generated input for multi-step generation, so the key
            # is never already present there; ``ensure_position_ids`` no-ops
            # when it is — the prefill-only case.) Plain-plan clean passes
            # are self-consistent and deliberately untouched.
            inputs = ensure_position_ids(inputs)
        saves_logits = isinstance(key, str) and key in plan.save_logits
        stop_early = (
            not saves_logits and _stop_carrier(group_model, group, taps) is not None
        )
        with group_model.trace(inputs) as tracer:
            for tap in taps[key]:
                tap.fn(group_model)
            if saves_logits:
                logits[key] = group_model.logits.cpu().save()
            elif stop_early:
                tracer.stop()
        return

    group_edges = [e for e in in_trace if e.src in group]
    signal_at = _edge_positions(taps, group_edges, "src")
    wait_at = _edge_positions(taps, group_edges, "dst")
    frames = {k: _frame_of(_input_of(plan, k)) or (0, 0) for k in group}
    want = [k for k in group if isinstance(k, str) and k in plan.save_logits]
    sink: list[Any] = []
    with group_model.trace() as tracer:
        _emit_invokes(
            group_model,
            tracer,
            group,
            lambda k: _input_of(plan, k),
            taps,
            signal_at,
            wait_at,
            fused_logits_sink=sink if want else None,
        )
    if want:
        logits.update(_slice_fused_logits(sink[0], group, frames, want))


def _execute_program(
    model: StandardizedTransformer, plan: Plan, lowered: LoweredPlan
) -> PlanResult:
    """Run a lowered plan: stages sequentially, one trace per group
    (:func:`_run_trace_group`), then the terminal generate trace when the
    program has one (EU3, #484 —
    :func:`~causalab.neural.plan._emit_generate_trace`; its input-shape
    refusals fire *before* any stage runs, keeping every refusal ahead of
    every forward pass). Each trace runs on its group's model (groups are
    same-model by construction — cross-model edges never ride in-trace).
    Traces run plainly in sequence — never inside ``model.session()`` (the
    module's *No session* policy, EU1 #482): between locally-executed traces
    the saved produce values are concrete tensors, which is all the stage
    boundaries (and the generate trace's forced constants) need, for
    single-model and cross-model (PL4) plans alike."""
    model_of = _model_resolver(model, plan)
    generate_key = lowered.program.generate_key
    if generate_key is not None:
        _check_generate_inputs(plan, generate_key)
    logits: dict[str, torch.Tensor] = {}
    for stage_groups in lowered.program.stages:
        for group in stage_groups:
            group_model = model_of(group[0])
            assert all(model_of(k) is group_model for k in group), (
                "trace group spans models — cross-model edges must stage"
            )
            _run_trace_group(
                group_model, plan, group, lowered.taps, lowered.program.in_trace, logits
            )
    if generate_key is not None:
        sequences, scores = _emit_generate_trace(
            model_of(generate_key), plan, generate_key, lowered.taps
        )
        return PlanResult(
            collects=lowered.collects,
            logits=logits,
            sequences=sequences,
            scores=scores,
        )
    return PlanResult(collects=lowered.collects, logits=logits)


def run_plan_staged(model: StandardizedTransformer, plan: Plan) -> PlanResult:
    """Execute ``plan`` as a plain sequence of traces — the executor entry
    (:func:`lower_plan` + :func:`_execute_program`).

    Stages run sequentially; within a stage, each group is one trace — fused
    multi-invoke with a barrier for in-trace edges (PL1's recipe), plain
    single-invoke otherwise (:func:`_run_trace_group`, the one executor
    body). Values crossing a stage boundary were saved by their produce tap
    and enter later traces as constants (device/dtype coercion at the
    consuming site, as everywhere in the neural layer). A generation plan's
    terminal generate trace runs last, after every collect stage (EU3, #484
    — :attr:`StagedProgram.generate_key`). Never inside ``model.session()``
    (the module's *No session* policy, EU1 #482).

    Gradient plans are refused here (``NotImplementedError``): the plain
    executor never runs a backward — a backward across staged traces has no
    consumer yet. :func:`~causalab.neural.plan.run_plan` executes gradient
    plans by gating them on schedule shape and diverting to the single-trace
    gradient path before reaching this executor. Raises ``ValueError`` for
    cyclic cross-input flow — before any forward pass.
    Never raises :class:`~causalab.neural.plan.StagingRequired`: staging is
    the point.
    """
    if plan.gradients is not None:
        raise NotImplementedError(
            "this Plan requests gradients — the plain-trace executor never "
            "runs them (a backward across staged traces has no consumer "
            "yet). run_plan routes gradient plans to the single-trace "
            "gradient path (CAP3, #456), which executes single-input plans "
            "only."
        )
    return _execute_program(model, plan, lower_plan(model, plan))


def _edge_positions(
    taps: Mapping[InvokeKey, list[_Tap]],
    group_edges: Sequence[_Edge],
    end: str,
) -> dict[InvokeKey, int]:
    """Per-invoke barrier positions for one trace group: the tap index of a
    producer's LAST in-group produce (``end="src"`` — signal after it) or a
    consumer's FIRST in-group consuming edit (``end="dst"`` — wait before
    it). Resolved through each edge's ``slot``, so produce taps for *other*
    (cross-stage) consumers on the same invoke don't move the rendezvous."""
    positions: dict[InvokeKey, list[int]] = defaultdict(list)
    for e in group_edges:
        if end == "src":
            key, match = (
                e.src,
                lambda t, e=e: t.kind == "produce" and t.key[2:4] == e.slot,
            )
        else:
            key, match = (
                e.dst,
                lambda t, e=e: t.kind == "consume" and t.key[2] == e.slot[0],
            )
        idx = [i for i, t in enumerate(taps[key]) if match(t)]
        assert idx, "edge does not correspond to a built tap"
        positions[key].extend(idx)
    reduce = max if end == "src" else min
    return {k: reduce(v) for k, v in positions.items()}
