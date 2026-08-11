"""``scan()`` preflight for position specs and batch plans — CAP5 (#458).

nnsight's ``model.scan()`` runs a trace on **fake (meta) tensors with zero
compute**: shapes propagate through the whole forward, nothing executes and no
weights are needed on device. This module uses it as a fail-fast gate (the
#127 culture): validate a :class:`~causalab.neural.plan.Plan` against a model
**before any real forward**, so the classic failures surface up front with
legible errors instead of deep inside a GPU run —

* out-of-range **layer / head indices** (the site accessors' own bounds checks
  fire during the scan),
* **width mismatches** — a source read whose per-example position widths
  differ from the write positions' (the multi-token-variable on
  variable-length base/counterfactual pairs classic). Fake tensors do not
  value-check advanced-indexing writes, so the site layer carries an explicit
  broadcast check (``site._check_write_fits``) that fires under scan and real
  runs alike,
* malformed inputs / featurizer shape mismatches — anything the fake forward's
  shape propagation can see.

What ``scan()`` **cannot** see: data-dependent tensor indexing. Fake tensors
carry shapes, not values, so an out-of-bounds *token position* sails through a
scan unchecked. The gate therefore also checks position specs **statically**
against each input's padded frame (``input_ids`` shape) — the same
``-L <= p < L`` bound the run frame enforces — before any scan runs. Positions
on raw-prompt inputs (no ``input_ids`` to read a frame from) are skipped.

The verdict is a :class:`PreflightReport` with a three-way ``status``:

* ``"clean"`` — the scan ran and every check passed; a real ``run_plan``
  (``lowering="auto"``) is expected to execute the plan's mechanics.
* ``"failed"`` — a check failed; the real run would fail the same way.
  ``error`` carries the legible message.
* ``"unsupported"`` — **scan cannot validate this model or plan**: the bare
  scan probe failed, the plan touched an op fake tensors cannot express
  (e.g. data-dependent control flow in a transform), the plan is a
  *generation* plan (:attr:`~causalab.neural.plan.Plan.generate` — scan
  cannot express the KV-cached decode loop, so no verdict is meaningful), or
  the backbone carries *installed persistent edits*
  (:mod:`causalab.neural.persistent` — an in-scan failure corrupts the
  installed mediators, so no scan is attempted at all). This is *not* a
  verdict about the plan. nnterp's ``allow_dispatch`` silently falls back
  ``scan → trace`` in this situation — a real, dispatched forward — which a
  preflight must never do; the trichotomy is the point (the issue's
  "scan-clean" vs "scan unsupported" distinction). When a case is ambiguous
  (a ``NotImplementedError`` matching neither torch's meta-kernel wording nor
  a causalab raising frame) the gate prefers ``unsupported``: a false
  "unsupported" merely skips the verdict, a false "failed" would hard-block a
  run that would have succeeded.

Exceptions outside the verdict family — ``AttributeError``, ``NameError``,
anything that is neither a scan limitation nor an error class the real run
would raise — **propagate** out of :func:`preflight_plan` instead of being
reported as a plan failure: a bug in a transform closure (or in the preflight
itself) is not a verdict.

Wiring: :func:`preflight_plan` is the standalone gate;
``run_plan(..., preflight=True)`` (:mod:`causalab.neural.plan`) runs it as an
opt-in pre-run gate (``failed`` raises :class:`PreflightError` before any
forward; ``unsupported`` logs a warning and proceeds — a scan-less model must
not be blocked); ``causalab.neural.validate`` records per-model scan support
(:func:`check_scan_support`) alongside the load-time checks, so onboarding
answers "can this checkpoint be preflighted at all".

Scope note: the preflight dry-runs the plan as **sequential per-invoke
scans** in dependency order — the staged lowering's shape, which ``auto``
falls back to whenever one fused trace cannot schedule the plan. It therefore
validates the ``lowering="auto"`` semantics; a plan that only fails under a
*forced* ``lowering="single"`` (:class:`~causalab.neural.plan.StagingRequired`)
can still be preflight-clean.
"""

from __future__ import annotations

import collections.abc
import dataclasses
from typing import Any, Iterator, Literal

import torch
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    UnsupportedOperatorException,
)
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

from causalab.neural.plan import (
    CollectOp,
    Plan,
    _build_taps,
    _frame_of,
    _model_resolver,
)
from causalab.neural.site import backbone_has_edits
from causalab.neural.staged import _input_of, _toposort

__all__ = [
    "PreflightError",
    "PreflightReport",
    "check_scan_support",
    "preflight_plan",
]

PreflightStatus = Literal["clean", "failed", "unsupported"]

#: Exception types that mean "fake tensors cannot express this op", not "the
#: plan is wrong": guards/kernels torch's FakeTensorMode refuses. A plan-scan
#: failure of one of these types is classified ``unsupported`` — the same op
#: may run fine on real tensors.
_SCAN_LIMITATIONS: tuple[type[BaseException], ...] = (
    DataDependentOutputException,
    DynamicOutputShapeException,
    UnsupportedOperatorException,
    GuardOnDataDependentSymNode,
)

#: Exception types that ARE a plan verdict: causalab's own validation and
#: refusal errors (``ValueError``/``IndexError``/``KeyError``) plus the
#: backend's shape errors (``RuntimeError``) — the real run raises the same
#: class. ``NotImplementedError`` (⊂ ``RuntimeError``) is classified
#: separately (:func:`_classify_not_implemented`); anything outside this
#: family (``AttributeError``, ``NameError``, …) is a bug in a transform
#: closure or in the preflight itself and **propagates** instead of being
#: laundered into a "your plan failed" verdict.
_PLAN_VERDICT_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    IndexError,
    KeyError,
    TypeError,
    RuntimeError,
)

#: Message markers of torch's missing-fake/meta-kernel ``NotImplementedError``.
#: torch has re-worded this error across releases, so the set is deliberately
#: broad, and a unit test pins that the *installed* torch's wording still
#: matches (``tests/neural/test_preflight.py``) — a torch bump that drifts
#: past every marker fails there loudly instead of silently flipping preflight
#: verdicts in the field.
_META_KERNEL_MARKERS: tuple[str, ...] = (
    # torch ≤2.3 dispatch wording:
    #   "Could not run 'aten::X' with arguments from the 'Meta' backend."
    "'Meta' backend",
    # torch 2.9 fake-impl wording:
    #   "aten::X: attempted to run this operator with Meta tensors, but there
    #    was no fake impl or Meta kernel registered"
    "Meta tensors",
    "Meta kernel",
    "fake impl",
    # register_meta stubs that raise unimplemented by default (e.g. nonzero)
    "register_meta",
    # any future wording that still names the dispatched aten op
    "aten::",
)


class PreflightError(RuntimeError):
    """Raised by :meth:`PreflightReport.raise_if_failed` on a ``failed`` gate."""


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Outcome of :func:`preflight_plan` — see the module docstring for the
    three-way ``status`` semantics. ``stage`` names the gate that produced a
    non-clean verdict: ``"headroom"`` (gradient plans — the scan dry run
    cannot cover the backward, so they get no verdict),
    ``"generate"`` (generation plans are not preflightable — scan cannot
    express the decode loop), ``"positions"`` (static bounds vs the padded
    frame), ``"persistent-edits"`` (no scan is attempted on a backbone
    carrying installed persistent edits), ``"scan-probe"`` (bare scan does
    not run on this model), or ``"scan-plan"`` (the fake-mode dry run of the
    lowered plan)."""

    status: PreflightStatus
    stage: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only for a ``clean`` verdict (``unsupported`` is *no* verdict)."""
        return self.status == "clean"

    def raise_if_failed(self) -> None:
        """Raise :class:`PreflightError` on a ``failed`` verdict. ``clean``
        and ``unsupported`` pass silently — an unsupported scan is a missing
        verdict, not a failed plan; callers decide how loudly to say so."""
        if self.status == "failed":
            raise PreflightError(
                f"plan failed scan preflight at the {self.stage!r} gate: {self.error}"
            )


# --------------------------------------------------------------------------- #
#  scan support probe                                                          #
# --------------------------------------------------------------------------- #
def _probe_inputs() -> dict[str, torch.Tensor]:
    """The minimal batch the bare probe runs (mirrors nnterp's dummy input)."""
    return {"input_ids": torch.tensor([[0, 1, 1]])}


def _error_name(exc: BaseException) -> str:
    """The original exception type's name — nnsight wraps in-trace errors in a
    dynamic ``NNsightException(original_type, ExceptionWrapper)`` subclass, so
    ``bases[0]`` recovers the user-meaningful name."""
    cls = type(exc)
    if cls.__name__ == "NNsightException" and cls.__bases__:
        return cls.__bases__[0].__name__
    return cls.__name__


def _summarize(exc: BaseException) -> str:
    """One legible line for a (possibly nnsight-wrapped) exception. Wrapped
    exceptions stringify as a full traceback; the ``TypeName: message`` header
    line is the one that matters (searched from the end — multi-line torch
    errors append debugging epilogues after it)."""
    name = _error_name(exc)
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    if len(lines) > 1 and lines[0].startswith("Traceback"):
        for line in reversed(lines):
            if line.startswith(f"{name}:"):
                return line
        return lines[-1]
    return f"{name}: {exc}"


def _is_causalab_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` was *raised by causalab code* — a deliberate
    honest-boundary refusal (e.g. the unmapped ``mlp_activation`` tap), not a
    backend gap. For an unwrapped exception the raising frame is the live
    traceback's last frame; for an nnsight-wrapped one the live frames are
    nnsight's re-raise site, so the raising frame is recovered from the last
    ``File "…"`` line of the deferred traceback text the wrapper embeds."""
    raising_module: str | None = None
    tb = exc.__traceback__
    while tb is not None:
        raising_module = tb.tb_frame.f_globals.get("__name__", "")
        tb = tb.tb_next
    if raising_module is not None and type(exc).__name__ != "NNsightException":
        return raising_module.startswith("causalab")
    file_lines = [ln for ln in str(exc).splitlines() if ln.strip().startswith('File "')]
    return bool(file_lines) and "/causalab/" in file_lines[-1]


def _classify_not_implemented(exc: NotImplementedError) -> PreflightStatus:
    """``failed`` or ``unsupported`` for a ``NotImplementedError`` under scan.

    Two very different things raise this type in a fake-mode dry run: torch's
    dispatcher missing a fake/meta kernel for an op (a scan limitation —
    ``unsupported``), and causalab's own honest-boundary refusals (an unmapped
    ``mlp_activation`` tap, a fused-QKV head view — ``failed``: the real run
    raises the same error). The split keys on the message markers
    (:data:`_META_KERNEL_MARKERS`) and, failing those, on *where* the error
    was raised (:func:`_is_causalab_refusal`). Anything still ambiguous
    defaults to ``unsupported`` — the safe direction: a false "unsupported"
    merely skips the verdict and the real run still raises the legible error,
    while a false "failed" would hard-block (``PreflightError``) a run that
    would have succeeded.
    """
    if any(marker in str(exc) for marker in _META_KERNEL_MARKERS):
        return "unsupported"
    if _is_causalab_refusal(exc):
        return "failed"
    return "unsupported"


def check_scan_support(model: Any) -> str | None:
    """Whether a bare fake-mode forward runs on ``model`` — ``None`` when it
    does, else the one-line error that stopped it.

    This is the "scan unsupported for this model" detector: some forwards
    (data-dependent control flow, ops without fake/meta kernels) cannot run
    under ``FakeTensorMode`` at all. nnterp's ``allow_dispatch`` reacts by
    silently falling back to a real ``trace()``; a preflight must instead
    *report* the gap — no compute happens here either way. Works on
    undispatched models too (``scan`` never dispatches).
    """
    try:
        with model.scan(_probe_inputs(), use_cache=False):
            pass
    except Exception as exc:
        return _summarize(exc)
    return None


# --------------------------------------------------------------------------- #
#  static position checks (what fake tensors cannot see)                       #
# --------------------------------------------------------------------------- #
def _iter_position_specs(plan: Plan) -> Iterator[tuple[str, str, Any]]:
    """Every explicit position spec in the plan, as
    ``(label, input key, positions)`` — collect positions, edit write
    positions, and each site-backed read-source's positions (resolved to the
    input it reads under)."""
    for i, op in enumerate(plan.ops):
        if isinstance(op, CollectOp):
            if op.positions is not None:
                yield f"ops[{i}].positions", op.input, op.positions
            continue
        if op.edit.positions is not None:
            yield f"ops[{i}].edit.positions", op.input, op.edit.positions
        for j, rs in enumerate(op.edit.read_sources):
            if rs.is_site and rs.positions is not None:
                yield (
                    f"ops[{i}].edit.read_sources[{j}].positions",
                    op.input if rs.input is None else rs.input,
                    rs.positions,
                )


def _position_rows(positions: Any) -> tuple[list[list[int]], bool] | None:
    """Normalize a position spec to ``(rows, per_row)`` for static bounds
    checking, or ``None`` for forms this check does not understand (those are
    left to the scan / run, whose own validation raises)."""
    if isinstance(positions, torch.Tensor):
        if positions.dim() == 1:
            return [[int(p) for p in positions.tolist()]], False
        if positions.dim() == 2:
            return [[int(p) for p in row] for row in positions.tolist()], True
        return None
    rows = list(positions)
    if rows and isinstance(rows[0], collections.abc.Sequence):
        try:
            return [[int(p) for p in row] for row in rows], True
        except (TypeError, ValueError):
            return None
    try:
        return [[int(p) for p in rows]], False
    except (TypeError, ValueError):
        return None


def _check_positions_static(plan: Plan) -> str | None:
    """Bounds-check every position spec against its input's padded frame.

    Fake tensors carry no values, so an out-of-range token index passes a
    scan silently and only explodes in the real forward (as an opaque index
    error — or a CUDA assert that poisons the context). The frame is read
    from the input's ``input_ids`` shape (:func:`~causalab.neural.plan._frame_of`);
    raw-prompt inputs have no static frame and are skipped. Returns the first
    legible error, or ``None``.
    """
    for label, input_key, positions in _iter_position_specs(plan):
        frame = _frame_of(plan.inputs[input_key])
        if frame is None:
            continue  # raw prompt — length unknowable before tokenization
        length, batch = frame
        normalized = _position_rows(positions)
        if normalized is None:
            continue  # unrecognized form — the scan / run raises its own error
        rows, per_row = normalized
        if per_row and len(rows) != batch:
            return (
                f"{label} (input {input_key!r}): {len(rows)} per-row position "
                f"rows for a batch of {batch} examples — positions must be "
                "resolved for exactly the examples in the padded batch."
            )
        for ex_idx, row in enumerate(rows):
            oob = [p for p in row if not -length <= p < length]
            if oob:
                where = f"example {ex_idx}" if per_row else "all examples (flat row)"
                return (
                    f"{label} (input {input_key!r}): positions {oob} are out "
                    f"of bounds for the input's padded length {length} "
                    f"({where}). Out of bounds they would silently address "
                    "the wrong token or crash the forward — the classic cause "
                    "is an index computed for a differently-shaped input "
                    "(e.g. a base position reused on a shorter "
                    "counterfactual, #176)."
                )
    return None


# --------------------------------------------------------------------------- #
#  the gate                                                                    #
# --------------------------------------------------------------------------- #
def preflight_plan(model: Any, plan: Plan) -> PreflightReport:
    """Validate ``plan`` against ``model`` with **zero compute** — static
    checks plus a fake-mode (``model.scan()``) dry run of the lowered taps.

    Gates run cheapest-first; the first non-clean verdict wins:

    1. **headroom** (model-free) — a gradient plan
       (:class:`~causalab.neural.plan.GradientRequest`) fails the preflight:
       the fake-mode dry run never executes the backward, so no scan verdict
       would cover the gradient contract (preflight the plan without its
       GradientRequest instead).
    2. **generate** (model-free) — a generation plan (:attr:`Plan.generate`,
       CAP2 #468) is ``unsupported``: ``scan()`` cannot express the KV-cached
       decode loop (``tracer.iter``), so a scan of the prefill alone would
       validate neither stepped ops nor the one-token step frames their
       positions resolve in — any ``clean``/``failed`` verdict from it would
       be bogus.
    3. **positions** (model-free) — static bounds of every position spec
       against its input's padded frame (see :func:`_check_positions_static`).
    4. **persistent-edits** — a model carrying installed persistent edits
       (:mod:`causalab.neural.persistent`;
       :func:`~causalab.neural.site.backbone_has_edits`) is ``unsupported``,
       checked **before any scan opens**: a scan aborted by an in-scan
       validation failure corrupts the installed edit mediators' interleaver
       state and poisons every later real trace (measured on nnsight 0.7 —
       ``RuntimeError: release unlocked lock``). A preflight must never
       degrade the model it validates, so an edited backbone gets no scan at
       all. Preflight before ``install_edits``, or uninstall first.
    5. **scan-probe** — a bare scan on every model the plan runs on; a probe
       failure is ``unsupported`` (scan cannot validate this model at all).
    6. **scan-plan** — the plan's taps (:func:`~causalab.neural.plan._build_taps`)
       execute as sequential per-invoke scans in dependency
       order; cross-invoke values flow between scans as saved fake tensors.
       Layer/head bounds, featurizer shapes and write widths all propagate
       here. Errors of a fake-tensor-limitation type are ``unsupported``
       (``NotImplementedError`` is split by :func:`_classify_not_implemented`
       — a missing meta kernel is ``unsupported``, a causalab refusal is
       ``failed``); the verdict family (:data:`_PLAN_VERDICT_TYPES`) is
       ``failed`` — the real run raises the same class. Anything outside both
       (``AttributeError``, ``NameError``, …) **propagates**: a bug in a
       transform closure or in the preflight itself is not a plan verdict.

    The scans never dispatch the model and never run a real forward.
    """
    if plan.gradients is not None:
        return PreflightReport(
            "failed",
            stage="headroom",
            error="this Plan requests gradients — gradient plans are not "
            "preflightable: the fake-mode dry run never executes the "
            "backward, so no scan verdict would cover the gradient "
            "contract. Preflight the same plan without its GradientRequest; "
            "run_plan executes single-input gradient plans (CAP3, #456).",
        )

    if plan.generate is not None:
        return PreflightReport(
            "unsupported",
            stage="generate",
            error="generation plans are not preflightable: model.scan() cannot "
            "express the KV-cached decode loop (tracer.iter), so a scan of "
            "the prefill alone would validate neither stepped ops nor the "
            "one-token step frames their positions resolve in. Preflight the "
            "prefill as a plain-forward plan if needed.",
        )

    error = _check_positions_static(plan)
    if error is not None:
        return PreflightReport("failed", stage="positions", error=error)

    model_of = _model_resolver(model, plan)
    models: list[Any] = []
    for m in (model, *plan.models.values()):
        if not any(m is seen for seen in models):
            models.append(m)

    # Refuse edited backbones BEFORE any scan opens — including the bare
    # probe: an in-scan exception on an edited model corrupts the installed
    # mediators (see gate 4 above), and a probe can fail too.
    for m in models:
        if backbone_has_edits(m):
            return PreflightReport(
                "unsupported",
                stage="persistent-edits",
                error="the model carries installed persistent edits "
                "(causalab.neural.persistent) — a scan aborted by an in-scan "
                "validation failure corrupts the installed edit mediators and "
                "poisons later real traces, so no scan is attempted on an "
                "edited backbone. Preflight the plan before install_edits, or "
                "uninstall_edits first.",
            )

    for m in models:
        probe_error = check_scan_support(m)
        if probe_error is not None:
            return PreflightReport(
                "unsupported",
                stage="scan-probe",
                error=f"model.scan() does not run on this model: {probe_error}",
            )

    try:
        taps, _collects, edges, _grad_leaves = _build_taps(model_of, plan)
        active = [
            key
            for key in taps
            if taps[key] or (isinstance(key, str) and key in plan.save_logits)
        ]
        for key in _toposort(active, edges):
            invoke_model = model_of(key)
            with invoke_model.scan(_input_of(plan, key), use_cache=False):
                for tap in taps[key]:
                    tap.fn(invoke_model)
    except _SCAN_LIMITATIONS as exc:
        return PreflightReport("unsupported", stage="scan-plan", error=_summarize(exc))
    except NotImplementedError as exc:
        # Missing fake/meta kernel (scan limitation → unsupported) vs a
        # causalab honest-boundary refusal (→ failed; the run raises it too).
        # NotImplementedError ⊂ RuntimeError, so this arm must precede the
        # verdict-family arm below.
        return PreflightReport(
            _classify_not_implemented(exc), stage="scan-plan", error=_summarize(exc)
        )
    except _PLAN_VERDICT_TYPES as exc:
        # causalab's validation/refusal errors and the backend's shape errors
        # — the real run raises the same class, so this IS a plan verdict.
        return PreflightReport("failed", stage="scan-plan", error=_summarize(exc))
    # Anything else (AttributeError, NameError, …) is a bug — in a transform
    # closure or in the preflight itself — not a plan verdict: it propagates
    # rather than masquerading as "your plan failed".
    return PreflightReport("clean")
