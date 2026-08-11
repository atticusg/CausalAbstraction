"""Persistent interventions over nnsight's ``model.edit()`` — CAP7 (#460).

Pyvene's ``IntervenableModel`` was ephemeral, so causalab threaded a per-batch
plan through every call site. nnsight can instead install an intervention **on
the model object**: ``model.edit(inplace=True)`` captures the with-body as a
default mediator that re-runs on every future traced forward
(``Envoy._default_mediators``; ``clear_edits()`` empties it). This module is
the causalab lifecycle around that primitive — apply a fixed steering vector
(or any other static :class:`~causalab.neural.edit.Edit`) once for an entire
eval, with no per-batch :class:`~causalab.neural.plan.Plan` threading:

.. code-block:: python

    from causalab.neural.modes import steer
    from causalab.neural.persistent import install_edits, uninstall_edits

    install_edits(pipeline.model, steer(Site("block_output", 12), vector))
    ...  # every traced forward now runs steered: run_plan, collects, generate
    uninstall_edits(pipeline.model)   # restores the base model bitwise

Lifecycle
---------
* **install** — :func:`install_edits` validates each edit eagerly (below),
  then opens one ``model.edit(inplace=True)`` block per edit — one backbone
  mediator per :class:`Edit`, each in its own call frame — and records the
  installed edits in a per-model registry (a ``WeakKeyDictionary``, so the
  bookkeeping dies with the model).
* **verify** — :func:`installed_edits` is the verifying read: it returns the
  recorded edits only after checking the registry against the backbone's
  mediator count, and raises :class:`PersistentEditError` when they disagree —
  an out-of-band ``model.clear_edits()`` or a raw ``model.edit()`` that this
  module didn't mediate. Both :func:`install_edits` and
  :func:`uninstall_edits` verify first, so drift surfaces at the next
  lifecycle call instead of compounding silently.
* **uninstall** — :func:`uninstall_edits` calls ``model.clear_edits()`` and
  empties the registry; the base model's outputs are restored bitwise
  (pinned in ``tests/neural/test_persistent.py``). ``force=True`` skips the
  drift check — the recovery path the drift errors prescribe.
* :func:`persistent_edits` is the scoped form: a context manager that
  installs on entry and uninstalls on exit.

Composition contract (persistent edit × traced Plan)
----------------------------------------------------
A persistent edit makes the model *the edited model*: every traced execution
— ``run_plan`` (single-trace and staged), ``collect_ordered`` /
``Site.collect`` / ``FeaturizedSite.collect``, the dataset wrappers, and
traced ``model.generate`` — runs under the installed edits, measured on
nnsight 0.7 and pinned in ``tests/neural/test_persistent.py``:

* The backbone prepends default mediators to every trace, so at a shared
  site the persistent edit fires **first**: a per-trace collect reads the
  *edited* activation, and a per-trace write lands **after** (overwriting)
  the edit's.
* Under a traced ``model.generate`` the edit applies **once, to the
  prefill**, and persists through the KV-cached decode — the same semantics
  as the engine's one generate emitter (``plan._emit_generate_trace``, which
  ``dataset.run_intervened_generation`` lowers onto since EU4 #485).
* One measured hazard is compiled away rather than composed: ``tracer.stop()``
  ends the forward at the deepest tap, stranding the mediator of any edit
  whose site fires later (nnsight raises ``MissedProviderError`` at trace
  exit). The compilers' single may-I-stop authority —
  ``plan._stop_carrier``, which every stop the plan and staged lowerings
  emit routes through (CAP6, #459) — therefore withholds the stop on an
  edited model, and ``collect_ordered`` mirrors the same
  :func:`causalab.neural.site.backbone_has_edits` guard inline: an edited
  model trades the tail of the forward for correctness.

Loud refusals — everything a persistent edit cannot mean:

* **Raw-HF execution.** nnsight edits live in the tracing layer; a plain HF
  call (``pipeline.hf_model.generate`` / ``hf_model(...)``) bypasses them.
  ``LMPipeline.generate`` therefore *refuses* (:class:`PersistentEditError`)
  while edits are installed — a steered eval that silently generated
  unsteered outputs is exactly the failure mode this module exists to
  prevent. Run generation through the traced path
  (``causalab.neural.dataset.run_intervened_generation``) or uninstall first.
* **Plan-only shapes.** A persistent edit runs on every forward of whatever
  input arrives — there are no named plan inputs, so a cross-input
  ``ReadSource(..., input=...)`` is refused at install; so is a read-only
  edit (``g=None``: nothing to install) and a site-backed read source firing
  after the written site (needs two passes — a staged Plan, not an edit).
* **Frame-bound positions.** The edit outlives any single batch, so its
  positions (and its read sources') must mean the same thing in every
  padded frame: ``None`` (all positions) or right-anchored flat indices
  (all negative — safe under the pipeline's left-pad convention). Absolute
  (non-negative) and per-row positions are resolved against one batch's
  frame and are refused; resolve those per batch through a Plan instead.
"""

from __future__ import annotations

import collections.abc
import contextlib
import weakref
from typing import Any, Iterator

import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit
from causalab.neural.site import Positions, forward_key

__all__ = [
    "PersistentEditError",
    "install_edits",
    "installed_edits",
    "persistent_edits",
    "uninstall_edits",
]


class PersistentEditError(RuntimeError):
    """The model's persistent-edit state cannot honor the request.

    Raised when the causalab registry and the nnsight backbone disagree about
    what is installed (an out-of-band ``model.clear_edits()`` or raw
    ``model.edit()``), and by ``LMPipeline.generate`` when plain HF generation
    would silently bypass installed edits.
    """


#: model → the edits installed through :func:`install_edits`, in install order.
#: Weak keys: the bookkeeping must not keep a model alive, and dies with it.
_INSTALLED: "weakref.WeakKeyDictionary[Any, tuple[Edit, ...]]" = (
    weakref.WeakKeyDictionary()
)


def _backbone_count(model: StandardizedTransformer) -> int:
    """How many default mediators the nnsight backbone carries — the ground
    truth :func:`installed_edits` verifies the registry against (one per
    ``model.edit()`` block; ``clear_edits()`` resets it to zero)."""
    return len(getattr(model, "_default_mediators", ()) or ())


# --------------------------------------------------------------------------- #
#  Install-time validation                                                     #
# --------------------------------------------------------------------------- #
def _require_frame_independent(positions: Positions | None, what: str) -> None:
    """Refuse positions that only mean something in one batch's padded frame.

    A persistent edit runs on every future forward, whatever the batch shape,
    so only frame-independent forms are installable: ``None`` (all positions)
    or a flat row of **negative** indices (right-anchored — the last real
    token of every row under the pipeline's left-pad convention). Non-negative
    indices count from a frame-specific pad boundary and per-row/ragged rows
    are born from one batch's resolution — both belong in a per-batch Plan.
    """
    if positions is None:
        return
    refuse = (
        f"{what} must be frame-independent — a persistent edit runs on every "
        f"future forward, so positions must mean the same thing in every "
        f"padded batch frame: None (all positions) or a flat row of negative "
        f"(right-anchored) indices. Got {positions!r}; resolve frame-bound "
        f"positions per batch through a Plan (causalab.neural.plan) instead."
    )
    if isinstance(positions, torch.Tensor):
        if positions.dim() != 1 or bool((positions >= 0).any()):
            raise ValueError(refuse)
        return
    rows = list(positions)
    for p in rows:
        if isinstance(p, collections.abc.Sequence) or isinstance(p, torch.Tensor):
            raise ValueError(refuse)  # per-row / nested — frame-bound
        if int(p) >= 0:
            raise ValueError(refuse)


def _validate(model: StandardizedTransformer, edit: Edit) -> None:
    """Everything :meth:`Edit.apply` would refuse *inside* the deferred
    mediator body, checked eagerly instead — a bad edit must fail at install
    time, not poison every future trace. Plus the persistent-only constraints
    (frame-independent positions; see the module docstring)."""
    # Runtime guard: a mediator built from a non-Edit would fail inside every
    # future trace, so the type error must fire here (the annotation alone
    # can't — callers holding Any reach this at runtime).
    if not isinstance(edit, Edit):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"install_edits takes Edit values, got {type(edit).__name__}")
    if edit.g is None:
        raise ValueError(
            "a read-only Edit (g=None) has nothing to install persistently — "
            "collect per input via Edit.collect / a Plan instead"
        )
    cross = [i for i, rs in enumerate(edit.read_sources) if rs.input is not None]
    if cross:
        raise ValueError(
            f"read_sources{cross} address another plan input — a persistent "
            "edit runs on every forward of whatever input arrives; there are "
            "no named inputs to read across. Run cross-input edits through a "
            "Plan (causalab.neural.plan.run_plan)."
        )
    _require_frame_independent(edit.positions, "a persistent Edit's positions")
    dst = forward_key(edit.site.site, model)
    for i, rs in enumerate(edit.read_sources):
        if not rs.is_site:
            continue
        _require_frame_independent(
            rs.positions, f"a persistent Edit's read_sources[{i}].positions"
        )
        src = forward_key(rs.value.site, model)
        if src > dst:
            raise ValueError(
                f"read_sources[{i}] ({rs.value.site!r}, rank {src}) fires after "
                f"this Edit's site ({edit.site.site!r}, rank {dst}) in forward "
                "order — reading a later site to write an earlier one needs "
                "two passes (a staged Plan), which a persistent edit cannot "
                "express."
            )
    # A frozen Site validates its layer only against a model; check now so an
    # out-of-range layer fails here rather than inside every future trace.
    check_layer = getattr(edit.site.site, "_check_layer", None)
    if check_layer is not None:
        check_layer(model)


# --------------------------------------------------------------------------- #
#  Lifecycle                                                                   #
# --------------------------------------------------------------------------- #
def _install_one(model: StandardizedTransformer, edit: Edit) -> None:
    """One ``model.edit`` block per :class:`Edit` — one backbone mediator each.

    A separate function call per edit so every mediator captures its own
    frame: nnsight compiles the with-body's *source* and re-executes it
    against the captured frame on every future trace, so ``edit`` here must
    be a fresh local per installation, never a shared loop variable.
    """
    with model.edit(inplace=True):
        edit.apply(model)


def install_edits(model: StandardizedTransformer, *edits: Edit) -> tuple[Edit, ...]:
    """Install ``edits`` persistently on ``model``; return everything now
    installed (previous installs included, in install order).

    Verifies the existing state first (so out-of-band drift surfaces here,
    loudly, before it compounds), then validates every edit eagerly (see
    :func:`_validate`), then installs. Installs stack: a second call appends.
    """
    if not edits:
        raise ValueError("install_edits needs at least one Edit")
    already = installed_edits(model)
    for edit in edits:
        _validate(model, edit)
    for edit in edits:
        _install_one(model, edit)
    installed = already + tuple(edits)
    _INSTALLED[model] = installed
    return installed


def installed_edits(model: StandardizedTransformer) -> tuple[Edit, ...]:
    """The verifying read: the edits installed through :func:`install_edits`,
    in install order — after checking the registry against the backbone.

    Raises :class:`PersistentEditError` when they disagree: fewer backbone
    mediators than recorded edits means an out-of-band ``model.clear_edits()``;
    more means a raw ``model.edit()`` this module didn't mediate. Either way
    the recorded state is no longer trustworthy — recover with
    ``uninstall_edits(model, force=True)`` and reinstall what you need.
    """
    recorded = _INSTALLED.get(model, ())
    n_backbone = _backbone_count(model)
    if n_backbone == len(recorded):
        return recorded
    if n_backbone < len(recorded):
        cause = (
            f"the backbone carries {n_backbone} default mediator(s) but "
            f"{len(recorded)} edit(s) are recorded — model.clear_edits() was "
            "called out-of-band"
        )
    else:
        cause = (
            f"the backbone carries {n_backbone} default mediator(s) but only "
            f"{len(recorded)} edit(s) are recorded — a raw model.edit() was "
            "installed outside install_edits"
        )
    raise PersistentEditError(
        f"persistent-edit state on {type(model).__name__} is inconsistent: "
        f"{cause}. Recover with uninstall_edits(model, force=True), then "
        "reinstall through install_edits."
    )


def uninstall_edits(
    model: StandardizedTransformer, *, force: bool = False
) -> tuple[Edit, ...]:
    """Remove every persistent edit from ``model``; return what was recorded.

    Clears the backbone (``model.clear_edits()`` — *all* default mediators)
    and the registry, restoring the base model's traced outputs bitwise.
    Verifies state first and raises :class:`PersistentEditError` on drift;
    ``force=True`` skips the check and clears unconditionally — the recovery
    path the drift errors prescribe. A no-op (returning ``()``) on a model
    with nothing installed.
    """
    removed = _INSTALLED.get(model, ()) if force else installed_edits(model)
    if force or removed:  # verified: no edits recorded ⇒ no mediators to clear
        model.clear_edits()
    _INSTALLED.pop(model, None)
    return removed


@contextlib.contextmanager
def persistent_edits(
    model: StandardizedTransformer, *edits: Edit
) -> Iterator[tuple[Edit, ...]]:
    """Scoped persistence: :func:`install_edits` on entry,
    :func:`uninstall_edits` on exit (also on exception). Yields everything
    installed. The exit uninstall is the strict form — drift inside the block
    raises on the way out."""
    installed = install_edits(model, *edits)
    try:
        yield installed
    finally:
        uninstall_edits(model)
