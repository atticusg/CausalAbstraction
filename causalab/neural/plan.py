"""Plan IR + the single-trace compiler — PL1.

The design doc's abstraction-stack item 4: a **Plan** is a declarative spec —
site-ops (each *collect* or *edit*) over a set of named inputs, plus what to
save — and the compiler lowers it to an nnsight trace program. This module
ships the IR and the **single-trace** lowering: every input becomes one
``tracer.invoke`` block inside one ``model.trace()``, taps are emitted in
forward-execution order, and cross-input values move through one
``tracer.barrier`` — the canonical cross-invoke interchange (source + base
invokes + barrier within ONE trace) that every interchange analysis lowers to:

.. code-block:: python

    src  = FeaturizedSite(Site("block_output", 5), rotation)
    dst  = FeaturizedSite(Site("block_output", 5), rotation)
    plan = Plan(
        inputs={"source": src_batch, "base": base_batch},
        ops=(
            EditOp("base", Edit(
                dst,
                g=lambda f, f_src: f_src,
                read_sources=(ReadSource(src, positions=[-1], input="source"),),
                positions=[-1],
            )),
            CollectOp("base", FeaturizedSite(Site("block_output", 9)), key="mid"),
        ),
        save_logits=("base",),
    )
    result = run_plan(model, plan)   # result.collects["mid"], result.logits["base"]

``mid`` here is collected *under* the layer-5 patch, in the same forward — the
collect∘intervene fusion pyvene needed the mixed model + ``sorted_keys``
contract for. It falls out of tap ordering: within one invoke every tap (an
edit's own read-sources, the RMW write, collects) is emitted in ascending
``(layer, forward_rank_on(model), declaration order)``, honoring nnsight's
constraint that one invoke touches modules in forward-pass order.

How nnsight actually runs a multi-invoke trace (measured on this backbone —
tiny-random Llama, nnsight 0.7; the facts the lowering is built on):

* The invokes are **fused into ONE forward** over the concatenated batch,
  **left-padded** to the longest input; activation values at each row's own
  token positions are preserved exactly.
* Per-layer accessor reads inside an invoke are **row-scoped** to that
  invoke's rows but come back **in the fused frame** — which is why this
  compiler requires frame-aligned inputs (below). nnterp's ``logits``
  accessor is trace-scoped (full fused batch); rows follow invoke definition
  order, so :func:`run_plan` saves it once and slices per input.
* Cross-invoke values move **forward only**: a value produced at an earlier
  forward position can feed a write at the same or a later position (the
  producing invoke defined first); a consumer whose write fires *before* a
  producer's read surfaces as nnsight's after-the-fact ``MissedProviderError``
  — the compiler rejects that shape up front as :class:`StagingRequired`.

What one fused trace does NOT cover — the honest boundaries:

* **Every plain plan is scheduled by ONE scheduler** (EU2 #483,
  :mod:`causalab.neural.staged`): :func:`run_plan` lowers the plan to a
  staged program — a plan that fits one fused forward schedules as the
  degenerate program (one stage, one trace) and runs exactly the trace
  described above. Data flow one fused trace cannot run — a read-source
  firing *after* its edit's site on the same input (the two-pass
  path-patching shape), cross-input flow against forward order, flow
  deeper than one producers → consumers phase, inputs whose padded frames
  differ, or inputs bound to different **models** (cross-model patching,
  PL4: ``Plan.models`` binds an input to the model it runs on; two models
  never share one fused forward) — schedules as plain sequential traces
  instead (no ``model.session()``, per :mod:`causalab.neural.staged`'s
  *No session* policy; separate traces have separate frames, so
  mixed-length inputs stage trivially). ``lowering="single"`` is a
  strictness assertion on the same schedule: it raises
  :class:`StagingRequired` (from schedule facts) iff the program needs
  more than one trace.
* **Batch layout is deliberately NOT fixed by the IR.** Each plan input is
  an already-batched tensor dict and each site read returns *that invoke's*
  rows — the canonical per-invoke nnsight form. How a counterfactual dataset
  maps onto invokes (fused vs. split vs. staged) is the F5 layout decision
  (#421, resolved — recorded in ``docs/REBASE_CAUSALAB_ON_NNTERP.md``,
  "Open design questions" → batching model), consumed at PL3
  (:mod:`causalab.neural.dataset`), not an IR commitment.
* **Gradients are implemented for single-input plain-forward plans** (CAP3,
  #456): a :class:`GradientRequest` makes the single-trace lowering run one
  backward over the traced forward and deliver ``d loss / d activation`` for
  the named collects in :attr:`PlanResult.gradients` — the
  saved-logits-backward contract (:mod:`causalab.neural.trainable`):
  graph-intact saves, ``backward()`` after the trace closes. Gradients gate
  on **schedule shape** (:func:`_refuse_gradient_shape`, keyed on trace
  count): a plan that schedules as more than one trace, or as one fused
  multi-invoke trace, is refused — gradients are implemented for
  single-input plans only (measured: an invoke's row-scoped reads branch
  off the fused forward, so a grad leaf made there never rejoins the
  continued forward; a backward across staged traces has no consumer yet).
  Gradient **generation** plans are refused at construction — backward
  through ``tracer.iter`` decode steps is unmeasured territory.

**Generation plans** (CAP2, #455) end in a terminal generate trace: give the
Plan a :class:`GenerateSpec` (``generate=GenerateSpec(max_new_tokens=N)``) and
its ops may carry ``step`` — the op then fires at that *generation step* of
ONE ``model.generate`` trace, via nnsight's ``tracer.iter``. ``step`` counts the
trace's forward passes: ``0`` is the prefill (the same pass a step-less op
targets), ``k`` the k-th KV-cached decode pass, whose activation frame is
one token wide — positions on a stepped op resolve in THAT frame (``[-1]`` /
``None`` address the new token). This is the capability pyvene structurally
forbade (``intervene_on_prompt=True`` touched the prompt pass only):
steering/patching at every generated token, not just prefill. The compiler
only ever emits a *bounded* iterator (an explicit step list — never
``iter[:]``, the documented deadlock), and refuses a step at or past
``max_new_tokens`` up front: an iteration that never happens is silently
skipped by nnsight AND the rest of the trace body is abandoned (measured).
Results come back as :attr:`PlanResult.sequences` (generated tokens only)
and, with ``output_scores``, per-step :attr:`PlanResult.scores` —
``save_logits`` is a plain-forward contract and is refused. Generation is a
**terminal stage kind** of the one scheduler (EU3, #484): a plan's *ops*
address ONE input — the generated one — but their ``read_sources`` may
address other plan inputs (``ReadSource(..., input="source")``): a generate
trace accepts only constants, so every such read is force-staged into an
earlier collect stage (:mod:`causalab.neural.staged`) and its saved value
enters the generate trace as a constant — the compiler derives the
split-forward layout :func:`causalab.neural.dataset.run_intervened_generation`
hand-coded before EU4 (#485; it now builds exactly these plans).

Refusals fire **before any forward pass**: the generation-plan checks and
the gradient gates are model-free for whole-component sites (scheduling
reads only site ranks and input shapes); strict-mode staging refusals and
the gradient schedule-shape gate fire on the computed schedule, before any
trace opens.

Design + as-landed record: ``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 5
("engine unification", #480) — the unified routing diagram, the refusal
relocation map with final anchors, the bypass registry, and the session
re-introduction gate.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Mapping, Sequence, Union

import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite

from causalab.neural.site import (
    Positions,
    backbone_has_edits,
    _index_key,  # pyright: ignore[reportPrivateUsage]
    _sequence_index,  # pyright: ignore[reportPrivateUsage]
    forward_key,
)

__all__ = [
    "CollectOp",
    "EditOp",
    "GenerateSpec",
    "GradientRequest",
    "Plan",
    "PlanResult",
    "SiteOp",
    "StagingRequired",
    "run_plan",
]

logger = logging.getLogger(__name__)


class StagingRequired(ValueError):
    """The plan is well-formed but cannot lower to ONE trace.

    Raised only under ``run_plan(..., lowering="single")`` — a strictness
    assertion on the schedule (EU2, #483): the scheduler
    (:func:`causalab.neural.staged.lower_plan`) determined the plan needs
    more than one trace, and the message names the schedule facts (per-edge
    ``staged_why`` reasons): a read of a site firing after the written site
    (same input — "two passes" — or across inputs against forward order —
    "backward in time"), cross-input flow beyond one producers → consumers
    phase ("chained cross-input flow"), inputs whose padded frames differ
    ("padded lengths" / "pre-tokenized"), inputs bound to different
    models (:attr:`Plan.models` — "bound to a different model", PL4 #406),
    or reads feeding a terminal generate trace ("a generate trace accepts
    only constants" — force-staged collect stages, EU3 #484).
    Under ``lowering="auto"`` (the default) the same schedule simply
    executes as plain sequential traces — nothing is raised.
    """


# --------------------------------------------------------------------------- #
#  IR                                                                          #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class CollectOp:
    """Read (and save) a :class:`FeaturizedSite` under one plan input.

    ``key`` names the saved tensor in :attr:`PlanResult.collects`. The read
    happens in-trace at the site's forward position, so a collect declared
    after an :class:`EditOp` sees that edit's effect (same pass) — declared
    at the *same site*, declaration order decides. ``step`` addresses one
    generation step of a generation plan (``Plan.generate``; ``0`` = prefill,
    ``k`` = the k-th KV-cached decode pass, a one-token frame — see the
    module docstring); it requires a :class:`GenerateSpec` on the plan.
    """

    input: str
    site: FeaturizedSite
    key: str
    positions: Positions | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("CollectOp.key must be a non-empty string")
        if self.step is not None and self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")


@dataclasses.dataclass(frozen=True)
class EditOp:
    """Apply a writing :class:`Edit` under one plan input.

    The edit's ``read_sources`` may address *other* plan inputs
    (``ReadSource(..., input="source")``) — the compiler stages those reads in
    the producing input's invoke and moves the values across the barrier; a
    plain ``ReadSource`` resolves under this op's own input. ``step``
    addresses one generation step of a generation plan (``Plan.generate``;
    ``0`` = prefill, ``k`` = the k-th KV-cached decode pass — the decode-time
    edit via ``tracer.iter``, see the module docstring); it requires a
    :class:`GenerateSpec` on the plan.
    """

    input: str
    edit: Edit
    step: int | None = None

    def __post_init__(self) -> None:
        if self.edit.g is None:
            raise ValueError(
                "EditOp wraps a writing Edit (g set); use CollectOp for a read"
            )
        if self.step is not None and self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")


SiteOp = Union[CollectOp, EditOp]


@dataclasses.dataclass(frozen=True)
class GradientRequest:
    """A Plan's request for gradients, executed by the single-trace lowering
    (CAP3, #456 — the PL1 headroom field, now implemented).

    ``loss`` maps a read-only mapping of the plan's saved values — every
    collect key → its saved tensor (graph-intact, still on device), plus each
    :attr:`Plan.save_logits` input key → that input's logits (same contract)
    — to a **scalar tensor**. ``wrt`` names the collect keys whose gradients
    to save: after the trace closes, the compiler runs ``loss(...).backward()``
    once (the saved-logits-backward contract, :mod:`causalab.neural.trainable`)
    and delivers ``d loss / d activation`` at each named collect's site,
    gathered to that collect's ``positions``, in :attr:`PlanResult.gradients`
    (CPU, detached — shaped like the collect itself).

    Honest boundaries: gradients run on **single-input plain-forward** plans
    (measured: an invoke's row-scoped reads branch off the fused multi-invoke
    forward, so a grad leaf made there never rejoins it; a generation plan
    cannot carry one — backward through ``tracer.iter`` decode steps is
    unmeasured, refused at construction); a ``wrt`` collect must address
    the **raw** activation (trivial featurizer, no ``feature_ids``) —
    feature-space gradients are the trainable-edit contract (ED3), not this
    one; the plan must run under grad mode (no ``torch.no_grad()``); and the
    schedule-shape gate (:func:`_refuse_gradient_shape`, keyed on trace
    count) refuses a plan that schedules as more than one trace or as one
    fused multi-invoke trace. Persistent edits
    (:mod:`causalab.neural.persistent`) compose like any other trace: the
    backward runs through the *edited* forward.
    """

    loss: Callable[[Mapping[str, torch.Tensor]], Any]
    wrt: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "wrt", tuple(self.wrt))
        if not self.wrt:
            raise ValueError("GradientRequest.wrt must name at least one collect key")


#: ``model.generate`` kwargs the generation lowering owns. ``use_cache=False``
#: would turn every decode step into a full re-forward, breaking the one-token
#: step frame stepped positions resolve against; the other three are the
#: lowering's own plumbing (fields on :class:`GenerateSpec` or fixed).
_RESERVED_GENERATE_KWARGS = (
    "max_new_tokens",
    "output_scores",
    "return_dict_in_generate",
    "use_cache",
)


@dataclasses.dataclass(frozen=True)
class GenerateSpec:
    """How a generation plan runs: ``model.generate(inputs,
    max_new_tokens=N, ...)`` instead of a plain forward (CAP2, #455).

    ``max_new_tokens`` is the step budget — it bounds every stepped op: the
    compiler only ever emits a *bounded* ``tracer.iter`` and refuses a
    ``step >= max_new_tokens`` at construction, because nnsight silently
    skips an iteration that never happens and abandons the rest of the trace
    body (measured, not theoretical — the design doc's "unbounded ``iter[:]``
    deadlocks" hazard). ``output_scores`` saves HF ``generate``'s per-step
    logits into :attr:`PlanResult.scores`. ``kwargs`` passes any other HF
    ``generate`` argument through (``do_sample``, ``min_new_tokens``,
    ``pad_token_id``, …); the four the lowering owns are rejected, and a
    ``min_new_tokens`` override *below* the last addressed step + 1 is
    refused at run time — it would let early EOS starve an addressed step,
    reintroducing the silent-skip failure.

    **What an op's reads see** (EU3, #484 — a generate trace accepts only
    constants, so every read that cannot run inside it is force-staged into
    an earlier collect stage and delivered as a saved value):

    * A **cross-input** read (``ReadSource(..., input="source")``) is
      captured in the source input's own plain forward, at the read's
      positions in *that input's* padded frame — never inside the generate
      trace.
    * A **same-input read at or before the written site** runs inside the
      generate trace, in the op's own step frame (the prefill frame for
      ``step``-less/``step=0`` ops; the one-token decode frame for
      ``step=k``).
    * A **same-input read after the written site** (the backward/self-graft
      shape) on a ``step``-less/``step=0`` op reroutes to the input's hidden
      ``(input, "clean")`` pass — a plain forward over the same tensors,
      numbered pad-aware from the attention mask exactly like HF
      ``generate``'s own prefill (``pipeline.ensure_position_ids``), staged
      *before* the generate trace: the value is the input's clean
      **prefill-frame** activation, the causal-tracing convention. On a
      ``step=k`` (k > 0) op the same shape is **refused** — the op's
      positions resolve in that step's one-token frame, so the reroute
      would silently reinterpret them in the full prompt frame. A read
      under the generation's own interventions must be expressed explicitly
      (capture it with a separate collect plan, or declare a duplicate
      input).
    """

    max_new_tokens: int
    output_scores: bool = False
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        reserved = sorted(set(self.kwargs) & set(_RESERVED_GENERATE_KWARGS))
        if reserved:
            raise ValueError(
                f"kwargs {reserved} are owned by the generation lowering — "
                "set max_new_tokens / output_scores as GenerateSpec fields; "
                "return_dict_in_generate and use_cache are fixed (the "
                "one-token step frame stepped positions resolve against "
                "assumes KV-cached decode)."
            )


@dataclasses.dataclass(frozen=True)
class Plan:
    """A declarative spec ``(inputs × site-ops × what-to-save)``.

    ``inputs`` maps input keys to model inputs — a padded batch dict from
    ``pipeline.load`` (required whenever the plan runs more than one input:
    the compiler must see ``input_ids`` to align frames and slice logits), or
    anything ``model.trace`` accepts for a single-input plan. ``ops`` are the
    site-ops; positions on them are already-resolved indices in the input's
    padded frame (the ST2 bridge output). ``save_logits`` names inputs whose
    full logits to save. An input no op or logits-save addresses is never run.

    ``models`` is cross-model patching (PL4): it binds an input key to the
    model that input's traces run on — capture in the *source* model's
    forward, inject into the target's (an unbound input runs on the model
    passed to :func:`run_plan`). Bindings compare by object identity, and a
    bound input's tensors must already be tokenized *by its own model's
    tokenizer* (at the plan layer inputs are pre-tokenized, so this is the
    caller's contract — the pipeline-level ``source_pipeline`` threading).
    Two distinct models never share a fused forward, so a genuinely
    cross-model plan always lowers to staged traces.

    ``generate`` makes this a **generation plan** (:class:`GenerateSpec`):
    the ops-addressed input runs through ONE terminal ``model.generate``
    trace and ops may carry ``step`` (decode-time edits via ``tracer.iter``
    — CAP2, see the module docstring). Ops must address exactly ONE input
    (the generated one); reads of *other* inputs go through
    ``read_sources`` and are captured in earlier collect stages (EU3, #484
    — see :class:`GenerateSpec`). A stepped op *requires* ``generate``,
    every step must be below ``max_new_tokens``, and ``save_logits`` is
    refused (generation results are :attr:`PlanResult.sequences` /
    ``scores``).

    Construction validates well-formedness only — whether the plan fits in
    one trace is the scheduler's call, because a staged plan is still a
    *valid* plan (the one scheduler lowers it, EU2 #483;
    ``lowering="single"`` merely asserts the one-trace schedule via
    :class:`StagingRequired`).
    """

    inputs: Mapping[str, Any]
    ops: Sequence[SiteOp] = ()
    save_logits: Sequence[str] = ()
    gradients: GradientRequest | None = None
    models: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    generate: GenerateSpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.save_logits, str):
            raise ValueError(
                f"save_logits takes a sequence of input keys, not the bare "
                f"string {self.save_logits!r}"
            )
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "ops", tuple(self.ops))
        object.__setattr__(self, "save_logits", tuple(self.save_logits))
        object.__setattr__(self, "models", dict(self.models))
        if not self.inputs:
            raise ValueError("a Plan needs at least one input")
        unknown_models = sorted(set(self.models) - set(self.inputs))
        if unknown_models:
            raise ValueError(
                f"models binds unknown inputs {unknown_models}; "
                f"known inputs: {sorted(self.inputs)}"
            )
        bound_to_none = sorted(k for k, m in self.models.items() if m is None)
        if bound_to_none:
            raise ValueError(
                f"models binds inputs {bound_to_none} to None — omit an input "
                "to run it on the model passed to run_plan"
            )
        if not self.ops and not self.save_logits:
            raise ValueError("an empty Plan (no ops, no save_logits) does nothing")
        for i, op in enumerate(self.ops):
            if op.input not in self.inputs:
                raise ValueError(
                    f"ops[{i}] addresses unknown input {op.input!r}; "
                    f"known inputs: {sorted(self.inputs)}"
                )
            if isinstance(op, EditOp):
                for j, rs in enumerate(op.edit.read_sources):
                    if rs.input is not None and rs.input not in self.inputs:
                        raise ValueError(
                            f"ops[{i}].edit.read_sources[{j}] addresses unknown "
                            f"input {rs.input!r}; known inputs: {sorted(self.inputs)}"
                        )
        keys = [op.key for op in self.ops if isinstance(op, CollectOp)]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"duplicate collect keys: {dupes}")
        for key in self.save_logits:
            if key not in self.inputs:
                raise ValueError(
                    f"save_logits names unknown input {key!r}; "
                    f"known inputs: {sorted(self.inputs)}"
                )
        if len(set(self.save_logits)) != len(self.save_logits):
            raise ValueError(f"duplicate save_logits entries: {self.save_logits}")
        if self.gradients is not None:
            missing = sorted(set(self.gradients.wrt) - set(keys))
            if missing:
                raise ValueError(f"gradients.wrt names unknown collect keys: {missing}")
            clashes = sorted(set(keys) & set(self.save_logits))
            if clashes:
                raise ValueError(
                    f"collect keys {clashes} collide with save_logits input keys "
                    "— a gradient plan's loss receives collects and logits in "
                    "one mapping, so the names must be distinct"
                )
            if self.generate is not None:
                raise ValueError(
                    "a Plan cannot carry both a GradientRequest and a "
                    "GenerateSpec — backward through tracer.iter decode "
                    "steps is unmeasured territory (CAP3 × CAP2). Compute "
                    "gradients with a plain-forward plan."
                )
        stepped = [i for i, op in enumerate(self.ops) if op.step is not None]
        if stepped and self.generate is None:
            raise ValueError(
                f"ops{stepped} address a generation step but the plan is a "
                "plain forward — give it a GenerateSpec "
                "(generate=GenerateSpec(max_new_tokens=N)) so the compiler "
                "lowers them onto tracer.iter inside a generate trace."
            )
        if self.generate is not None:
            over = sorted(
                {
                    op.step
                    for op in self.ops
                    if op.step is not None and op.step >= self.generate.max_new_tokens
                }
            )
            if over:
                raise ValueError(
                    f"steps {over} are out of range for max_new_tokens="
                    f"{self.generate.max_new_tokens} (valid steps: 0.."
                    f"{self.generate.max_new_tokens - 1}; step 0 is the "
                    "prefill) — nnsight silently skips an iteration that "
                    "never happens and abandons the rest of the trace, so "
                    "the bound is enforced up front."
                )
            if self.save_logits:
                raise ValueError(
                    "save_logits is a plain-forward contract; a generation "
                    "plan returns PlanResult.sequences (and per-step scores "
                    "via GenerateSpec(output_scores=True))."
                )


@dataclasses.dataclass
class PlanResult:
    """What :func:`run_plan` returns: ``collects`` keyed by
    :attr:`CollectOp.key` and per-input ``logits`` keyed by input, all CPU
    tensors (the package convention for collected values). Generation plans
    fill ``sequences`` instead of ``logits`` — the generated tokens only
    (prompt stripped), ``(batch, n_generated)`` per input — and, when
    :attr:`GenerateSpec.output_scores` is set, ``scores``: one
    ``(batch, vocab)`` tensor per generated step. Gradient plans fill
    ``gradients``, keyed by :attr:`GradientRequest.wrt` entry
    (``d loss / d activation``, shaped like the matching collect)."""

    collects: dict[str, torch.Tensor]
    logits: dict[str, torch.Tensor]
    sequences: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    scores: dict[str, list[torch.Tensor]] = dataclasses.field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Single-trace compiler                                                       #
# --------------------------------------------------------------------------- #
#: One invoke of the lowered program: a plan input, or ``(input, "clean")`` —
#: the hidden extra pass the staged compiler adds when a same-input read fires
#: after the written site (the read then observes the *clean* forward of that
#: input; for a read under interventions, declare an explicit duplicate input).
InvokeKey = Union[str, tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class _Edge:
    """One cross-invoke data dependency: the produce tap identified by
    ``slot`` on invoke ``src`` feeds the consuming edit (``slot[0]``-th op) on
    invoke ``dst``. ``src_hook <= dst_hook`` is the fused forward's
    arrow-of-time condition; the staged scheduler (PL2) stages edges that
    violate it (or whose frames differ) across traces."""

    src: InvokeKey
    dst: str
    src_hook: tuple[int, int]
    dst_hook: tuple[int, int]
    slot: tuple[int, int]  # (op index, read-source index) — the produce tap's id


@dataclasses.dataclass
class _Tap:
    """One in-trace action, scheduled by ``key`` within its invoke.

    ``key`` = ``(layer, forward_rank_on(model), op index, intra-op seq)`` —
    ascending forward order with declaration order as the tiebreak, so a
    collect declared after an edit at the same site reads the edited value.
    ``kind`` drives barrier placement: ``produce`` taps feed another input
    (signal after the last one), ``consume`` taps use cross-input values
    (wait before the first one).
    """

    key: tuple[int, int, int, int]
    fn: Callable[[StandardizedTransformer], None]
    kind: str  # "collect" | "read" | "produce" | "edit" | "consume"

    @property
    def hook(self) -> tuple[int, int]:
        """The forward position this tap fires at — the cross-invoke
        schedulability currency (declaration order breaks ties *within* an
        invoke; across invokes, definition order does)."""
        return self.key[:2]


def _model_resolver(
    model: StandardizedTransformer, plan: Plan
) -> Callable[[InvokeKey], Any]:
    """``invoke key → the model its trace runs on``: the input's
    :attr:`Plan.models` binding, else the default ``model`` passed to
    :func:`run_plan`. A ``(input, "clean")`` alias runs on its original's
    model. Identity (``is``) is the sameness the compilers compare by."""

    def model_of(key: InvokeKey) -> Any:
        return plan.models.get(key if isinstance(key, str) else key[0], model)

    return model_of


def _refuse_gradient_shape(plan: Plan, program: Any) -> None:
    """The gradient gate on **schedule shape** (pre-trace, EU2 #483): a
    gradient plan runs iff its schedule is the degenerate program — exactly
    ONE trace of ONE invoke. Keyed on **trace count** first (a no-edge
    two-input plan schedules as two traces — no fused forward is involved,
    but two traces still means a backward across staged traces, which has
    no consumer yet), then on the one group's invoke count (measured: inside
    a fused multi-invoke forward an invoke's row-scoped reads branch off the
    fused tensors, so a grad leaf made there never rejoins the forward).
    ``program`` is the :class:`~causalab.neural.staged.StagedProgram`."""
    if plan.gradients is None:
        return
    if program.num_traces == 1 and all(
        len(group) == 1 for stage in program.stages for group in stage
    ):
        return
    invokes = [key for stage in program.stages for group in stage for key in group]
    raise NotImplementedError(
        f"this gradient plan schedules as {program.num_traces} trace(s) over "
        f"invokes {invokes} — gradients are implemented for single-input "
        "plans only: inside a fused multi-invoke forward an invoke's "
        "row-scoped reads branch off the fused tensors, so a grad leaf made "
        "there never rejoins the forward (measured), and a backward across "
        "staged traces has no consumer yet. Run one gradient plan per input."
    )


def _check_gradient_support(plan: Plan) -> None:
    """Model-free refusals for gradient plans the single-trace lowering
    cannot honor: a ``wrt`` collect through a non-trivial featurizer (or a
    ``feature_ids`` subspace) — feature-space gradients are the trainable-edit
    contract (ED3, :mod:`causalab.neural.trainable`), not the plan compiler's.
    """
    if plan.gradients is None:
        return
    wrt = set(plan.gradients.wrt)
    for i, op in enumerate(plan.ops):
        if not isinstance(op, CollectOp) or op.key not in wrt:
            continue
        if not op.site.featurizer.is_trivial() or op.site.feature_ids is not None:
            raise NotImplementedError(
                f"ops[{i}] (collect {op.key!r}) is a gradients.wrt target but "
                "reads through a non-trivial featurizer or feature_ids — the "
                "plan compiler delivers gradients w.r.t. the raw activation "
                "only; feature-space gradients are the trainable-edit "
                "contract (causalab.neural.trainable)."
            )


def _build_taps(
    model_of: Callable[[InvokeKey], Any],
    plan: Plan,
) -> tuple[dict[InvokeKey, list[_Tap]], dict[str, Any], list[_Edge], dict[str, Any]]:
    """Flatten every op into per-invoke taps; return ``(taps_by_invoke,
    collects, edges, grad_leaves)`` where ``collects`` is filled in-trace by
    the taps and ``edges`` records every cross-invoke data dependency.

    Under a gradient plan every collect saves **graph-intact and on device**
    (no ``.cpu()`` — the loss consumes them; :func:`_finalize`
    offloads after the backward), and each ``gradients.wrt`` collect
    additionally turns its site's raw forward-path tensor into an autograd
    leaf (``requires_grad_(True)`` on the frozen-model activation, or
    ``retain_grad()`` when the graph already tracks it) saved into
    ``grad_leaves`` — the tensors whose ``.grad`` the post-trace backward
    fills.

    An edit is decomposed: each site-backed read-source becomes its own tap at
    *its* forward position (feeding a slot), and the RMW write runs at the
    edit site's position reading the slots — so taps from different ops
    interleave correctly instead of deadlocking on nnsight's forward-order
    constraint (which a bare ``Edit.apply`` inside a longer tap list would).

    Each tap ranks on the model of the input it runs under (``model_of``,
    :func:`_model_resolver`) — per-head sites reorder across architectures,
    and under cross-model plans a produce tap fires in the *source* model's
    forward.

    This is the superset build every lowering shares (EU2 #483 collapsed the
    retired ``staged=False`` single-trace variant): a same-input read firing
    after the written site reroutes to the hidden ``(input, "clean")``
    invoke — an extra clean pass over the same tensors — and every read tap
    ``.save()``s its value so it survives its trace (later traces, the
    terminal generate trace included, consume it as a concrete constant;
    within one trace the save is a no-op for consumers, which read the same
    proxy).
    """
    slots: dict[tuple[int, int], Any] = {}
    collects: dict[str, Any] = {}
    grad_leaves: dict[str, Any] = {}
    taps: dict[InvokeKey, list[_Tap]] = {key: [] for key in plan.inputs}
    edges: list[_Edge] = []
    grad_keys = (
        frozenset(plan.gradients.wrt) if plan.gradients is not None else frozenset()
    )
    graph_saves = plan.gradients is not None

    for op_idx, op in enumerate(plan.ops):
        if isinstance(op, CollectOp):
            layer, rank = forward_key(op.site.site, model_of(op.input))

            def collect_fn(
                m: StandardizedTransformer,
                op: CollectOp = op,
                leaf: bool = op.key in grad_keys,
                graph: bool = graph_saves,
            ) -> None:
                if leaf:
                    raw = op.site.site.read(m, None)
                    if raw.requires_grad:
                        raw.retain_grad()
                    else:
                        raw.requires_grad_(True)
                    grad_leaves[op.key] = raw.save()
                value = op.site.read(m, op.positions)
                collects[op.key] = (value if graph else value.cpu()).save()

            taps[op.input].append(_Tap((layer, rank, op_idx, 0), collect_fn, "collect"))
            continue

        edit = op.edit
        dst = forward_key(edit.site.site, model_of(op.input))
        aux_get: list[Callable[[], Any]] = []
        has_cross = False
        for rs_idx, rs in enumerate(edit.read_sources):
            if not rs.is_site:
                aux_get.append(lambda rs=rs: rs.value)
                continue
            src_input = op.input if rs.input is None else rs.input
            src = forward_key(rs.value.site, model_of(src_input))
            slot = (op_idx, rs_idx)

            def read_fn(
                m: StandardizedTransformer,
                rs: Any = rs,
                slot: tuple[int, int] = slot,
            ) -> None:
                slots[slot] = rs.value.read(m, rs.positions).save()

            if src_input == op.input and src <= dst:
                taps[op.input].append(
                    _Tap((src[0], src[1], op_idx, rs_idx), read_fn, "read")
                )
            elif src_input == op.input:
                if op.step is not None and op.step > 0:
                    # A stepped op's positions resolve in the ONE-token decode
                    # frame; the clean-prefill reroute resolves them in the
                    # full prompt frame — a silent reinterpretation, so the
                    # shape is refused (the retired generation lowering's
                    # refusal, restored). Step-less/step=0 ops keep the
                    # reroute: their frame IS the prefill frame.
                    raise ValueError(
                        f"ops[{op_idx}] (generation step {op.step}) reads a "
                        "site after the written site on the same input — "
                        "within a generation step the read must fire at or "
                        "before the written site (the op's positions resolve "
                        "in that step's one-token frame, so rerouting the "
                        "read to the clean prefill pass would silently "
                        "reinterpret them in the full prompt frame). Capture "
                        "the value with a separate collect plan and feed it "
                        "in as a constant ReadSource."
                    )
                clean: InvokeKey = (op.input, "clean")
                has_cross = True
                taps.setdefault(clean, []).append(
                    _Tap((src[0], src[1], op_idx, rs_idx), read_fn, "produce")
                )
                edges.append(_Edge(clean, op.input, src, dst, slot))
            else:
                has_cross = True
                taps[src_input].append(
                    _Tap((src[0], src[1], op_idx, rs_idx), read_fn, "produce")
                )
                edges.append(_Edge(src_input, op.input, src, dst, slot))
            aux_get.append(lambda slot=slot: slots[slot])

        def edit_fn(
            m: StandardizedTransformer,
            edit: Edit = edit,
            aux_get: tuple[Callable[[], Any], ...] = tuple(aux_get),
        ) -> None:
            g = edit.g
            assert g is not None  # EditOp.__post_init__ guarantees a writing Edit

            def g_wrapped(f: Any) -> Any:
                aux = tuple(FeaturizedSite._coerce(get(), f) for get in aux_get)
                return g(f, *aux)

            edit.site.edit(m, g_wrapped, edit.positions)

        taps[op.input].append(
            _Tap(
                (dst[0], dst[1], op_idx, len(edit.read_sources)),
                edit_fn,
                "consume" if has_cross else "edit",
            )
        )

    for tap_list in taps.values():
        tap_list.sort(key=lambda t: t.key)
    return taps, collects, edges, grad_leaves


def _frame_of(value: Any) -> tuple[int, int] | None:
    """``(padded length, batch size)`` of a pre-tokenized input, ``None`` when
    unknowable (a raw prompt). Frame identity is what lets two inputs share
    one fused forward without shifting each other's resolved positions."""
    ids = value.get("input_ids") if isinstance(value, Mapping) else None
    if ids is None:
        return None
    return (int(ids.shape[-1]), int(ids.shape[0]))


def _fuse_position_ids(
    ordered: Sequence[InvokeKey], inputs_of: Callable[[InvokeKey], Any]
) -> Callable[[InvokeKey], Any]:
    """Rewrite per-invoke ``position_ids`` for a fused multi-invoke trace.

    nnsight's invoke batching (``LanguageModel._batch``) merges ``input_ids`` /
    ``attention_mask`` / ``labels`` across invokes but passes every other key
    through from the *first* invoke unchanged — so per-invoke ``position_ids``
    (the left-pad-aware numbering ``ensure_position_ids`` attaches, load-bearing
    on absolute-position models) reach the fused forward with the wrong batch
    size and crash it. This helper is the compiler-owned fix: when any invoke
    carries ``position_ids``, strip them from every invoke and ride the
    **fused-shape** concatenation (invoke definition order — the measured row
    layout) on the first invoke's dict, which nnsight passes through verbatim.
    Invokes without ``position_ids`` contribute the ``arange`` default their
    solo forward would have used, so their rows are behavior-preserving.
    Callers have already frame-aligned the inputs (equal padded length), so the
    concatenation is well-formed. No-op when no invoke carries the key.
    """
    prepared = {key: inputs_of(key) for key in ordered}
    if not any(
        isinstance(d, Mapping) and "position_ids" in d for d in prepared.values()
    ):
        return inputs_of
    rows = []
    for key in ordered:
        d = prepared[key]
        if isinstance(d, Mapping) and "position_ids" in d:
            rows.append(d["position_ids"])
        else:
            ids = d["input_ids"]  # pre-tokenized: guaranteed by the frame check
            rows.append(
                torch.arange(ids.shape[-1], device=ids.device).expand(ids.shape[0], -1)
            )
    stripped = {
        key: {k: v for k, v in prepared[key].items() if k != "position_ids"}
        for key in ordered
    }
    first = ordered[0]
    stripped[first] = {**stripped[first], "position_ids": torch.cat(rows)}
    return lambda key: stripped[key]


def _stop_carrier(
    model: StandardizedTransformer,
    ordered: Sequence[InvokeKey],
    taps: Mapping[InvokeKey, list[_Tap]],
) -> InvokeKey | None:
    """The invoke that carries the early ``tracer.stop()`` — CAP6 (#459) —
    and the single "may this trace stop at all" authority.

    When a trace saves no logits, nothing downstream of its deepest tap is
    read, so the forward can stop right after it — layers L+1..N never run
    (pure wasted-compute elimination for collect-only plans; the design
    doc's §4 "Collect features at scale" row).

    **Except under persistent edits** (CAP7, :mod:`causalab.neural.
    persistent`): a default mediator whose site fires past the deepest tap
    would be stranded mid-wait by the stop (the measured
    ``MissedProviderError``), so an edited ``model``
    (:func:`causalab.neural.site.backbone_has_edits`) never stops — this
    returns ``None`` and the trace pays the full forward. Every stop this
    compiler (and the staged one) emits routes through here, so new stop
    sites inherit the guard; ``collect_ordered`` mirrors it inline
    (``site`` sits below this module in the import graph).

    The stop may fire only after EVERY tap in the trace has run. All taps at
    shallower hooks completed at earlier modules of the fused forward; at
    one module, nnsight handles invokes in definition order (hooks are
    inserted in mediator-index order — ``nnsight/intervention/hooks.py``)
    and blocks until each invoke's worker reaches its next event before
    handling the next invoke. So the safe carrier is the LAST invoke, in
    emission order, among those whose deepest tap sits at the trace-wide
    deepest hook: every other tap at that hook belongs to an
    earlier-defined invoke and has already run. Taps within an invoke are
    hook-sorted (:func:`_build_taps`), so the carrier's own saves — and its
    trailing barrier signal, if any — all land before the stop.
    """
    if backbone_has_edits(model):
        return None
    hooks = {k: max(t.hook for t in taps[k]) for k in ordered if taps[k]}
    if not hooks:
        return None
    deepest = max(hooks.values())
    return [k for k in ordered if hooks.get(k) == deepest][-1]


def _emit_invokes(
    model: StandardizedTransformer,
    tracer: Any,
    ordered: Sequence[InvokeKey],
    inputs_of: Callable[[InvokeKey], Any],
    taps: Mapping[InvokeKey, list[_Tap]],
    signal_at: Mapping[InvokeKey, int],
    wait_at: Mapping[InvokeKey, int],
    fused_logits_sink: list[Any] | None,
) -> None:
    """Emit one invoke per key inside an open multi-invoke trace, wiring one
    ``tracer.barrier`` over the participants: each producer signals right
    after its ``signal_at`` tap (its last in-trace produce), each consumer
    waits right before its ``wait_at`` tap (its first in-trace consume).
    When ``fused_logits_sink`` is given, the fused logits are saved once (in
    the last invoke — nnterp's ``logits`` is trace-scoped) and **appended**
    to it: the trace body runs in nnsight's worker frame, so results leave
    it by mutating pre-existing objects, never by rebinding locals.

    When ``fused_logits_sink`` is ``None``, nothing downstream of the taps
    is read from this trace — no logits, no later-layer sites (all sites
    are taps), and no backward either: gradient plans never reach a
    multi-invoke emission (the schedule-shape gate
    :func:`_refuse_gradient_shape` refuses anything but a one-trace,
    one-invoke schedule; the single-input gradient path emits its own
    trace), and a generate trace never emits here (a generation plan's
    collect stages are edge-less singletons — the single-invoke fast path —
    and the terminal trace is :func:`_emit_generate_trace`'s, whose emission
    deliberately never stops early: the generator output IS its product,
    and a ``tracer.stop()`` inside ``tracer.iter`` decode iterations is
    unproven territory) — so ``tracer.stop()`` is emitted after the deepest
    tap (:func:`_stop_carrier`) and the layers above it never run (CAP6,
    #459). Called only inside plain ``model.trace()`` bodies, never inside
    ``model.generate``.

    Called by the unified executor's per-group emission
    (:func:`causalab.neural.staged._run_trace_group`) with that group's
    in-trace edges — the degenerate one-group schedule reproduces the
    retired single-trace lowering's emission exactly. Per-invoke
    ``position_ids`` are re-fused for the batched forward (see
    :func:`_fuse_position_ids`).
    """
    inputs_of = _fuse_position_ids(ordered, inputs_of)
    participants = set(signal_at) | set(wait_at)
    barrier = tracer.barrier(len(participants)) if participants else None
    stop_after = (
        None if fused_logits_sink is not None else _stop_carrier(model, ordered, taps)
    )
    for pos, key in enumerate(ordered):
        with tracer.invoke(inputs_of(key)):
            for i, tap in enumerate(taps[key]):
                if barrier is not None and wait_at.get(key) == i:
                    barrier()
                tap.fn(model)
                if barrier is not None and signal_at.get(key) == i:
                    barrier()
            if fused_logits_sink is not None and pos == len(ordered) - 1:
                fused_logits_sink.append(model.logits.cpu().save())
            if key == stop_after:
                tracer.stop()


def _slice_fused_logits(
    fused: Any,
    ordered: Sequence[InvokeKey],
    frames: Mapping[InvokeKey, tuple[int, int]],
    want: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Split a fused-forward logits tensor back into per-input rows (invoke
    definition order — the measured row layout)."""
    logits: dict[str, torch.Tensor] = {}
    offset = 0
    for key in ordered:
        batch = frames[key][1]
        if isinstance(key, str) and key in want:
            logits[key] = fused[offset : offset + batch]
        offset += batch
    return logits


def run_plan(
    model: StandardizedTransformer,
    plan: Plan,
    *,
    lowering: str = "auto",
    preflight: bool = False,
) -> PlanResult:
    """Lower ``plan`` through the one scheduler and execute it (EU2, #483).

    Every plain (non-generation) plan is scheduled by
    :func:`causalab.neural.staged.lower_plan` and executed as plain
    sequential traces — the design doc's contract of the minimum number of
    forward passes: a plan that fits one fused forward schedules as the
    degenerate program (one stage, one trace) and runs exactly that fused
    trace. ``lowering`` selects strictness, not a compiler:

    * ``"auto"`` (the default) — execute whatever the schedule says.
    * ``"single"`` — a strictness assertion: raise :class:`StagingRequired`
      (message assembled from the schedule's per-edge ``staged_why`` facts)
      iff the program needs more than one trace; otherwise identical to
      ``"auto"``.
    * ``"staged"`` — a deprecated alias of ``"auto"`` (its only historical
      delta, forcing a separate staged executor, no longer exists).

    ``model`` is the model every input runs on unless :attr:`Plan.models`
    binds it to another one (cross-model patching, PL4) — a genuinely
    cross-model plan always takes the staged lowering.

    A **generation plan** (:attr:`Plan.generate`) lowers through the same
    scheduler, as plain collect stages plus ONE terminal ``model.generate``
    trace (EU3, #484): every read into the generate trace is force-staged
    (``"generate-with-variable-intervention"``) — captured in an earlier
    stage, consumed by the generate trace as a saved constant — and the
    generate trace itself (stepped ops on a bounded ``tracer.iter``) is
    emitted last, by the one generate emitter
    (:func:`_emit_generate_trace`), never inside a session.
    ``lowering="single"`` strictness applies like any plan: a generation
    plan with no cross-input (or clean-pass) reads is exactly one trace
    and passes. It never carries gradients (refused at construction).

    Persistent edits (:mod:`causalab.neural.persistent`) **compose**: every
    trace this compiler emits runs under the model's installed edits, so plan
    ops observe the *edited* model — at a shared site the persistent edit
    fires first (a collect reads the edited value; a plan write lands after
    it). The one interaction that doesn't compose is compiled away: the
    collect-only early stop is suppressed on an edited model (see

    :func:`causalab.neural.site.backbone_has_edits`). A gradient plan
    composes the same way — its backward runs through the edited forward.

    ``preflight=True`` runs the zero-compute ``scan()`` gate first
    (:func:`causalab.neural.preflight.preflight_plan` — CAP5, #458): a
    ``failed`` verdict raises
    :class:`~causalab.neural.preflight.PreflightError` with the legible cause
    *before any forward pass*; a model scan cannot run on (``unsupported``)
    logs a warning and proceeds — the gate never blocks a scan-less model and
    never substitutes a real trace for the scan. Generation plans report
    ``unsupported`` (scan cannot express the KV-cached decode loop), so the
    gate warns and hands them to the scheduler unvalidated.

    A **gradient plan** (:class:`GradientRequest`) is gated on schedule
    shape (:func:`_refuse_gradient_shape` — ``NotImplementedError`` unless
    the schedule is exactly one trace of one invoke) and diverted to the
    single-trace gradient path (:func:`_run_gradient_trace`): the same trace
    with graph-intact saves, then one ``loss.backward()`` after it closes —
    all before any forward pass is wasted, and never under
    ``torch.no_grad()`` (``RuntimeError``).
    """
    if lowering not in ("auto", "single", "staged"):
        raise ValueError(
            f"unknown lowering {lowering!r}; expected 'auto', 'single', or 'staged'"
        )

    if preflight:
        from causalab.neural.preflight import preflight_plan

        report = preflight_plan(model, plan)
        report.raise_if_failed()
        if report.status == "unsupported":
            logger.warning(
                "scan preflight unavailable (%s gate): %s — proceeding without "
                "a preflight verdict.",
                report.stage,
                report.error,
            )

    from causalab.neural.staged import (
        _execute_program,
        _refuse_not_single,
        lower_plan,
    )

    if plan.gradients is not None:
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "this Plan requests gradients but grad mode is off — do not "
                "wrap run_plan in torch.no_grad() for a gradient plan."
            )
        _check_gradient_support(plan)
    lowered = lower_plan(model, plan)
    if lowering == "single":
        _refuse_not_single(model, plan, lowered)
    if plan.gradients is not None:
        _refuse_gradient_shape(plan, lowered.program)
        return _run_gradient_trace(model, plan, lowered)
    return _execute_program(model, plan, lowered)


def _run_gradient_trace(
    model: StandardizedTransformer, plan: Plan, lowered: Any
) -> PlanResult:
    """The single-trace gradient path (CAP3, #456): the degenerate schedule's
    one trace with graph-intact saves, then one ``loss.backward()`` after it
    closes (:func:`_finalize`) — gradients land in
    :attr:`PlanResult.gradients` and every returned tensor is CPU.

    ``lowered`` is the plan's :class:`~causalab.neural.staged.LoweredPlan`;
    callers dispatch through :func:`_refuse_gradient_shape` first, so the
    program is exactly one trace of one invoke (a plain string input key —
    a clean-pass alias always comes with a second invoke). Logits save
    graph-intact (no ``.cpu()`` — the loss consumes them); a plan that saves
    no logits still stops after its deepest tap (CAP6 — the backward runs
    through the partial graph), with ``_stop_carrier`` withholding the stop
    under persistent edits (CAP7)."""
    ((group,),) = lowered.program.stages
    (key,) = group
    exec_model = _model_resolver(model, plan)(key)
    taps = lowered.taps
    logits: dict[str, torch.Tensor] = {}
    stop_early = (
        key not in plan.save_logits
        and _stop_carrier(exec_model, group, taps) is not None
    )
    with exec_model.trace(plan.inputs[key]) as tracer:
        for tap in taps[key]:
            tap.fn(exec_model)
        if key in plan.save_logits:
            logits[key] = exec_model.logits.save()
        elif stop_early:
            tracer.stop()
    return _finalize(plan, lowered.collects, logits, lowered.grad_leaves)


def _finalize(
    plan: Plan,
    collects: dict[str, Any],
    logits: dict[str, torch.Tensor],
    grad_leaves: dict[str, Any],
) -> PlanResult:
    """Close out a single-trace run: for a gradient plan, run the one
    backward (saved-logits contract) and offload the graph-intact saves to
    CPU; otherwise the saves are already CPU tensors."""
    if plan.gradients is None:
        return PlanResult(collects=collects, logits=logits)
    grads = _execute_gradients(plan, collects, logits, grad_leaves)
    return PlanResult(
        collects={k: v.detach().cpu() for k, v in collects.items()},
        logits={k: v.detach().cpu() for k, v in logits.items()},
        gradients=grads,
    )


def _execute_gradients(
    plan: Plan,
    collects: dict[str, Any],
    logits: dict[str, torch.Tensor],
    grad_leaves: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """One backward over the traced forward, after the trace closed.

    The loss consumes the graph-intact saves (collect keys + ``save_logits``
    input keys in one mapping — :class:`GradientRequest`); each ``wrt``
    leaf's ``.grad`` is then the raw site activation's gradient, gathered to
    the collect's positions exactly like the value read
    (:func:`causalab.neural.site._sequence_index` semantics) so gradient and
    collect align element-for-element.
    """
    assert plan.gradients is not None  # callers dispatch on this
    loss_inputs: dict[str, torch.Tensor] = dict(collects)
    for key in plan.save_logits:
        loss_inputs[key] = logits[key]
    loss = plan.gradients.loss(loss_inputs)
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        got = (
            f"a tensor of shape {tuple(loss.shape)}"
            if isinstance(loss, torch.Tensor)
            else f"a {type(loss).__name__}"
        )
        raise ValueError(f"GradientRequest.loss must return a scalar tensor, got {got}")
    if not loss.requires_grad:
        raise RuntimeError(
            "the gradient plan's loss does not depend on the traced forward — "
            "it must be computed from the saved collects/logits it receives, "
            "not from detached or fresh tensors."
        )
    loss.backward()
    positions_of = {
        op.key: op.positions for op in plan.ops if isinstance(op, CollectOp)
    }
    grads: dict[str, torch.Tensor] = {}
    for key in plan.gradients.wrt:
        leaf = grad_leaves[key]
        grad = leaf.grad
        if grad is None:
            raise RuntimeError(
                f"no gradient reached collect {key!r} — the loss does not "
                "depend on that site's forward (it sits after every value "
                "the loss reads, or the graph was cut)."
            )
        positions = positions_of[key]
        if positions is not None:
            grad = grad[_index_key(_sequence_index(positions))]
        grads[key] = grad.detach().cpu()
    return grads


def _check_generate_inputs(plan: Plan, key: str) -> None:
    """Model-free input-shape refusals for the terminal generate trace
    (CAP2, #455): the generated input ``key`` must be pre-tokenized (the
    compiler slices the generated tokens out of the generator output by
    prompt length); prompt-shaped ``position_ids`` are refused for
    multi-step generation (measured to corrupt decode steps — see
    ``pipeline.ensure_position_ids``; the prefill-only case,
    ``max_new_tokens == 1``, is exactly right and passes); and a
    ``GenerateSpec.kwargs`` ``min_new_tokens`` override *below* the last
    addressed step + 1 is refused — early EOS could then starve an
    addressed step, and nnsight silently skips the unfulfilled iteration
    and abandons the rest of the trace body (measured). Called by the
    executor **before any collect stage runs** — every refusal fires before
    any forward pass — and again by :func:`_emit_generate_trace` (defense
    in depth for direct emitter callers)."""
    spec = plan.generate
    assert spec is not None  # callers dispatch on it
    inputs = plan.inputs[key]
    if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
        raise ValueError(
            f"a generation plan needs a pre-tokenized input (a mapping with "
            f"'input_ids'), but input {key!r} is a {type(inputs).__name__} — "
            "the compiler must know the prompt length to slice the generated "
            "tokens out of the generator output."
        )
    if "position_ids" in inputs and spec.max_new_tokens > 1:
        raise ValueError(
            f"input {key!r} carries prompt-shaped position_ids but "
            f"max_new_tokens={spec.max_new_tokens} — multi-step generate "
            "numbers its own per-step positions, and an explicit "
            "position_ids is measured to corrupt decode steps (see "
            "pipeline.ensure_position_ids). Drop the key for multi-step "
            "generation; it is only correct for the prefill-only case "
            "(max_new_tokens == 1)."
        )
    last_step = max((0 if op.step is None else op.step for op in plan.ops), default=0)
    min_new = spec.kwargs.get("min_new_tokens", last_step + 1)
    if last_step > 0 and min_new < last_step + 1:
        raise ValueError(
            f"GenerateSpec.kwargs overrides min_new_tokens={min_new}, below "
            f"the last addressed generation step + 1 ({last_step + 1}) — "
            f"early EOS could then end generation before step {last_step} "
            "runs, and nnsight silently skips the unfulfilled iteration and "
            "abandons the rest of the trace body (measured). Raise "
            "min_new_tokens or drop the override."
        )


def _emit_generate_trace(
    exec_model: StandardizedTransformer,
    plan: Plan,
    key: str,
    taps: Mapping[InvokeKey, list[_Tap]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    """The generate-trace emitter — CAP2 (#455); since EU3 (#484) the
    terminal stage of the unified engine: ONE ``model.generate`` trace over
    the generated input ``key`` (:attr:`~causalab.neural.staged.
    StagedProgram.generate_key`), stepped ops on nnsight's ``tracer.iter``.
    Returns ``(sequences, scores)``; the collect taps fill the lowered
    plan's live ``collects`` dict as they run. ``taps`` is the scheduler's
    tap mapping (:attr:`~causalab.neural.staged.LoweredPlan.taps`) — this
    emitter builds no taps and refuses no cross-input/multi-input shape
    (the scheduler's job): a cross-input or clean-pass read was force-staged
    into an earlier collect stage, whose produce ``.save()`` already holds a
    concrete tensor by the time the consume tap here reads it through its
    ``aux_get`` slot (``FeaturizedSite._coerce`` places it). The input-shape
    refusals (:func:`_check_generate_inputs`) stay here — and fire earlier,
    in the executor, before any collect stage runs.

    Facts the emission is built on (measured on the tiny-random backbone,
    nnsight 0.7, against a forward-pass-counting raw-hook oracle):

    * ``step`` counts the trace's forward passes per module — ``0`` is the
      prefill (the same pass a step-less op targets, so both lower to the
      same group), ``k`` the k-th KV-cached decode pass, whose activation
      frame is ONE token wide (``(b, 1, d)``): positions on a stepped op
      resolve in that frame.
    * EVERY tap — iteration-0 ones included — rides ONE bounded
      ``tracer.iter[[k1, k2, ...]]`` (an explicit ascending step list, never
      an unbounded slice — the design doc's documented deadlock) entered as
      the FIRST statement of the trace body, with per-step dispatch inside
      the loop. The loop must come first: nnsight's step counter is a set of
      hooks registered at loop *entry*, so a tap emitted before the loop
      that touches a module a stepped tap also addresses consumes that
      module's pass-0 fire pre-registration and shifts every later step by
      one (measured: the plan's ``iter[k]`` then waits for pass ``k+1``,
      which may never come). Within every step group, taps stay in
      ascending forward order (PL1's invariant; the constraint holds inside
      a generate trace too — the scheduler's per-invoke tap sort delivers
      it, and the per-step dispatch here is a stable filter of that order).
    * An iteration that never happens is silently skipped AND the rest of
      the trace body is abandoned (the generator-output save never runs) —
      hence the construction-time step bound, and ``min_new_tokens``
      defaulting to the last addressed step + 1 so early EOS cannot starve
      an addressed step. A ``GenerateSpec.kwargs`` override *below* that
      floor would reintroduce exactly the refused failure, so it is
      rejected model-free (:func:`_check_generate_inputs`); an abandoned
      trace body from any other cause (e.g. a custom stopping criterion in
      ``kwargs``) is backstopped by a legible ``RuntimeError`` instead of a
      bare ``IndexError``.

    A generation plan carries at least one op (the IR-wide empty-plan
    refusal) and the generated input is derived from the ops on purpose —
    "an input no op addresses is never run" is the Plan contract. An
    *un-intervened* generation baseline is therefore not a Plan at all:
    plain ``pipeline.generate`` (or the dataset layer) owns it.

    No CAP6 early-exit here, deliberately: the collect-only
    ``tracer.stop()`` (:func:`_stop_carrier`, plain traces) never applies
    to a generate trace — the generator output is this emitter's product,
    every decode step must complete for the sequences/scores contract, and
    stopping a forward from inside a ``tracer.iter`` iteration is unproven
    territory (nothing measures what it does to the KV cache or the
    iterator's step counter). NEVER inside a session (the *No session*
    policy, :mod:`causalab.neural.staged`).

    Prefill-edit semantics are pyvene's ``intervene_on_prompt=True``: edits
    on pass 0 persist through the KV-cached decode — the contract
    :func:`causalab.neural.dataset.run_intervened_generation` lowers onto
    since EU4 (#485; this is the package's ONE intervened-generation
    emitter). ``position_ids`` keeps the
    plain-forward asymmetry as an explicit contract instead of a silent
    special case: prompt-shaped ``position_ids`` are exactly right for the
    prefill-only case (``max_new_tokens == 1``) and measured-wrong across
    decode steps (multi-step generate numbers its own — see
    ``pipeline.ensure_position_ids``), so multi-step inputs carrying the key
    are refused up front.
    """
    spec = plan.generate
    assert spec is not None  # callers dispatch on it
    _check_generate_inputs(plan, key)
    inputs = plan.inputs[key]
    last_step = max((0 if op.step is None else op.step for op in plan.ops), default=0)

    step_of = {
        op_idx: 0 if op.step is None else op.step for op_idx, op in enumerate(plan.ops)
    }
    taps_at: dict[int, list[_Tap]] = {}
    for tap in taps[key]:
        taps_at.setdefault(step_of[tap.key[2]], []).append(tap)

    defaults: dict[str, Any] = dict(
        max_new_tokens=spec.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=spec.output_scores,
        do_sample=False,
        use_cache=True,
    )
    if last_step > 0:
        defaults["min_new_tokens"] = last_step + 1
    tokenizer = getattr(exec_model, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        defaults["pad_token_id"] = tokenizer.pad_token_id
    defaults.update(spec.kwargs)

    steps = sorted(taps_at)
    sink: list[Any] = []
    with exec_model.generate(dict(inputs), **defaults) as tracer:
        for _step in tracer.iter[steps]:
            for tap in taps_at[_step]:
                tap.fn(exec_model)
        sink.append(exec_model.generator.output.save())

    if not sink:
        raise RuntimeError(
            "the generate trace body was abandoned before the generator "
            "output was saved — an addressed iteration never ran (nnsight "
            "skips it silently: generation ended before step "
            f"{last_step}, e.g. through a stopping criterion in "
            "GenerateSpec.kwargs). See the preceding nnsight 'was not "
            "provided' warning for the missed step."
        )
    out = sink[0]
    prompt_len = int(inputs["input_ids"].shape[-1])
    sequences = {key: out.sequences[:, prompt_len:].detach().cpu()}
    scores: dict[str, list[torch.Tensor]] = {}
    if spec.output_scores:
        scores[key] = [step.detach().cpu() for step in (out.scores or ())]
    return sequences, scores
