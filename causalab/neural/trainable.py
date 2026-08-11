"""Trainable edits — ED3: the grad contract, the loss slice, the outer loop.

Training a mask (DBM) or a rotation (DAS) on the new stack needs four things
pyvene used to hide inside ``TrainableIntervention`` subclasses and the
``train_interventions`` loop; they land here, once:

**The grad contract** (the F6 spike's output, pinned by
``tests/neural/test_trainable.py`` against a grad-enabled raw-hook oracle —
same modules, same loss, same ``param.grad``):

* Base-model parameters are frozen **once at load**
  (:func:`freeze_model_parameters`, wired into ``LMPipeline`` for freshly
  loaded models) — not per-intervenable-model (pyvene's
  ``disable_model_gradients`` dance). Gradients w.r.t. activations still flow
  wherever a trainable leaf participates.
* Trainable parameters live in ordinary modules applied **in-trace** — a
  featurizer (``SubspaceFeaturizer`` rotation) or a
  :class:`~causalab.neural.modes.MaskGate` — and receive gradients through the
  site write (``LayerAccessor.__setitem__``, including the tuple-rewrap path)
  and through *both* featurize paths (the base read and the source).
* **Saved-logits backward:** save the logits **on-device, graph intact**
  (``model.logits.save()`` — no ``.cpu()``, no detach), compute the loss and
  call ``backward()`` *outside* the trace. This is the simpler of the two
  candidate contracts (vs. loss-inside-trace) and is the one the toolkit
  builds on; :func:`traced_label_loss` is its canonical form.

**The training edit shapes** (:func:`das_edit`, :func:`dbm_edit`): both take a
**pre-collected raw activation** as the source (the pyvene
``source_representations`` pattern — capture the counterfactual run's
activation once, outside the training loop) and featurize it **live in the
base trace** through the site's current featurizer, exactly as pyvene's
``forward(base, source)`` featurized both sides — so a DAS rotation receives
gradient through the source path as well as the base path. ED2's
:func:`~causalab.neural.modes.interchange`/:func:`~causalab.neural.modes.mask`
treat a tensor source as *already-featurized* constants; the trainable shapes
must not, hence these constructors.

**The differentiable loss slice** (``LM_loss_and_metric_fn``'s semantics,
decoupled from pyvene — lands here because training needs it at Wave 7,
before the scoring adapter exists; MX1/MX2 *consume* it, never re-implement
it): concatenate right-padded label tokens onto the left-padded base
(:func:`concat_label_inputs`, re-deriving ``position_ids`` from the joint
mask), run the edited forward, slice the logits that predict the label span,
cross-entropy with pad ignore (:func:`label_ce_loss`).

**The hard-threshold readout** (:func:`selected_feature_ids`): which features
a trained mask selected. The *outer optimization loop* over these primitives
(epochs, AdamW, the temperature anneal) is a training loop and therefore
lives in ``methods/`` — :mod:`causalab.methods.edit_training` (CODEBASE §3
invariant 1); the production DAS/DBM harness is
``causalab/methods/trained_subspace/train.py``.

**Explicit device placement** (:func:`place_edit_parameters`): each edit's
modules move to its site's layer device — on an ``hf_device_map``-sharded
model every layer may live on its own GPU, and feature-space math runs where
the activation lives. pyvene needed a ``get_device`` monkeypatch here
(``intervenable_model.py``); the new stack owns placement explicitly. The
optimizer then steps parameters wherever they live.

Scope
-----
Edits run over **one input per trace**: training sources come pre-collected
(raw activations captured from the counterfactual run) or from a same-trace
earlier site. Cross-input interchange inside one trace is PL1 (#403);
dataset-scale paired batching is PL3 (#405); LR schedulers, early stopping,
and experiment-harness wiring arrive when MX2 reroutes ``train_interventions``
(#409). Multiple edits in one trace are applied in forward order of their
*sites*; an edit whose site-backed source fires before another edit's site
needs the plan compiler's global ordering (PL1) — constant/raw sources (the
training pattern) impose no such constraint.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import Any, Iterable, Iterator, Sequence

import torch
from nnterp import StandardizedTransformer

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.modes import MaskGate
from causalab.neural.pipeline import LMPipeline, ensure_position_ids
from causalab.neural.site import Positions, Site, forward_key

__all__ = [
    "concat_label_inputs",
    "das_edit",
    "dbm_edit",
    "edit_parameters",
    "force_last_token_logits",
    "freeze_model_parameters",
    "label_ce_loss",
    "place_edit_parameters",
    "selected_feature_ids",
    "site_device",
    "traced_label_loss",
]


# --------------------------------------------------------------------------- #
#  Parameters: freeze, discover, place                                         #
# --------------------------------------------------------------------------- #
def freeze_model_parameters(model: StandardizedTransformer) -> None:
    """Freeze the base model's parameters — once, idempotently.

    Trainable edits optimize featurizer/gate parameters only; the frozen base
    both saves grad buffers and pins the contract that a training step never
    perturbs the model. Gradients w.r.t. *activations* still flow from any
    trainable leaf onward (frozen parameters don't sever the graph), which is
    also what pullback-style analyses rely on.
    """
    for p in model._model.parameters():
        p.requires_grad_(False)


def _iter_edit_modules(edit: Edit) -> Iterable[torch.nn.Module]:
    """Every module an :class:`Edit` computes through: its site's featurizer
    pair, a module-typed ``g`` (the :class:`MaskGate`), and the featurizer
    pairs of site-backed read-sources."""
    fsites = [edit.site] + [
        rs.value for rs in edit.read_sources if isinstance(rs.value, FeaturizedSite)
    ]
    for fsite in fsites:
        yield fsite.featurizer.featurizer
        yield fsite.featurizer.inverse_featurizer
    if isinstance(edit.g, torch.nn.Module):
        yield edit.g


def edit_parameters(edits: Sequence[Edit]) -> list[torch.nn.Parameter]:
    """The deduplicated trainable parameters across ``edits`` — what the
    optimizer steps. Shared modules (the same rotation featurizing both the
    base read and a source read, or one :class:`MaskGate` reused across
    batches) contribute their parameters once; frozen featurizers
    (``trainable=False``) contribute none."""
    params: dict[int, torch.nn.Parameter] = {}
    for edit in edits:
        for module in _iter_edit_modules(edit):
            for p in module.parameters():
                if p.requires_grad:
                    params[id(p)] = p
    return list(params.values())


def site_device(model: StandardizedTransformer, fsite: FeaturizedSite) -> torch.device:
    """The device of the layer a site taps — where its feature-space math runs.

    ``embeddings`` reads the token-embedding module; everything else reads its
    decoder layer. On a sharded (``hf_device_map``) model each layer may live
    on its own GPU; on a single-device model this is just that device.
    """
    site = fsite.site
    if getattr(site, "component", None) == "embeddings":
        module = model.model.embed_tokens
    else:
        module = model.model.layers[site.layer]
    return next(module.parameters()).device


def place_edit_parameters(
    model: StandardizedTransformer, edits: Sequence[Edit]
) -> None:
    """Move each edit's modules to the device of the site they compute at.

    Site-backed read-sources move to *their own* site's layer device (their
    featurize runs at that read). pyvene placed interventions per
    ``hf_device_map`` key and monkeypatched ``get_device`` for its internal
    index tensors; here placement is one explicit pass, and
    :class:`FeaturizedSite`'s coercion handles the residual cross-device value
    movement (a source read landing on the destination's device).
    """
    for edit in edits:
        dev = site_device(model, edit.site)
        for module in _iter_edit_modules(edit):
            module.to(dev)
        for rs in edit.read_sources:
            if isinstance(rs.value, FeaturizedSite):
                src_dev = site_device(model, rs.value)
                rs.value.featurizer.featurizer.to(src_dev)
                rs.value.featurizer.inverse_featurizer.to(src_dev)


# --------------------------------------------------------------------------- #
#  Training edit shapes: raw sources featurized live                           #
# --------------------------------------------------------------------------- #
def _featurized(site: Site | FeaturizedSite) -> FeaturizedSite:
    return site if isinstance(site, FeaturizedSite) else FeaturizedSite(site)


def _constant(raw_source: torch.Tensor) -> torch.Tensor:
    """Detach a pre-collected source: it is a *constant* of the optimization —
    gradients flow through its live featurization, never into however it was
    produced. Without this, a source collected while any upstream parameter
    still required grad drags its whole capture graph into every training step
    (and the second step's backward crashes on the freed graph)."""
    return raw_source.detach()


def _featurize_raw(fsite: FeaturizedSite, raw: torch.Tensor) -> torch.Tensor:
    """Featurize a raw activation through ``fsite``'s *current* featurizer (the
    live parameters — gradients flow), gathered to its ``feature_ids``. The
    reconstruction error is discarded: the write path re-featurizes base and
    keeps *base's* error, per the ST3 error-term contract."""
    features, _ = fsite.featurizer.featurize(raw)
    if fsite.feature_ids is None:
        return features
    return features[..., list(fsite.feature_ids)]


def das_edit(
    site: Site | FeaturizedSite,
    raw_source: torch.Tensor,
    *,
    positions: Positions | None = None,
) -> Edit:
    """The DAS training shape: interchange where the source is a
    **pre-collected raw activation** ``(batch, len(positions), d)``, featurized
    live in the base trace by the site's (trainable) featurizer — the rotation
    receives gradient through both the base and the source featurize, exactly
    like pyvene's ``FeatureInterchangeIntervention.forward(base, source)``.
    For inference-time interchange with already-featurized constants use
    :func:`causalab.neural.modes.interchange`."""
    fsite = _featurized(site)

    def g(f: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        return _featurize_raw(fsite, raw)

    return Edit(
        fsite,
        g=g,
        read_sources=(ReadSource(_constant(raw_source)),),
        positions=positions,
    )


class _LiveSourceGate(torch.nn.Module):
    """:func:`dbm_edit`'s ``g``: featurize the raw source live, then gate. A
    module (with the gate as a registered submodule) rather than a closure so
    :func:`edit_parameters` / :func:`place_edit_parameters` see the gate's
    parameters on the :class:`Edit` value."""

    def __init__(self, fsite: FeaturizedSite, gate: MaskGate) -> None:
        super().__init__()
        self.gate = gate
        self._fsite = (
            fsite  # plain attribute: its featurizer is discovered via the site
        )

    def forward(self, f: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        return self.gate(f, _featurize_raw(self._fsite, raw))


def dbm_edit(
    site: Site | FeaturizedSite,
    raw_source: torch.Tensor,
    gate: MaskGate,
    *,
    positions: Positions | None = None,
) -> Edit:
    """The DBM training shape: :class:`MaskGate` blend where the source is a
    **pre-collected raw activation**, featurized live (see :func:`das_edit`) —
    pyvene's ``FeatureMaskIntervention.forward`` semantics. The gate anneals
    via :func:`train_edits`; read the trained selection with
    :func:`selected_feature_ids`."""
    fsite = _featurized(site)
    return Edit(
        fsite,
        g=_LiveSourceGate(fsite, gate),
        read_sources=(ReadSource(_constant(raw_source)),),
        positions=positions,
    )


# --------------------------------------------------------------------------- #
#  The differentiable loss slice                                               #
# --------------------------------------------------------------------------- #
def _label_trace(text: str) -> CausalTrace:
    """A minimal single-variable trace so ``pipeline.load`` can tokenize a bare
    label string (the same shim ``LM_loss_and_metric_fn`` used)."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def concat_label_inputs(
    pipeline: LMPipeline,
    base_inputs: dict[str, torch.Tensor],
    labels: Sequence[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Concatenate tokenized labels onto a (left-padded) base batch for the
    label-scoring forward; returns ``(joint_inputs, label_ids)``.

    Labels are tokenized **right**-padded to ``pipeline.max_new_tokens`` with
    no specials and no chat wrap (they continue the prompt). Any load-time
    ``position_ids`` on the base are dropped and re-derived from the *joint*
    attention mask: base's right-aligned real tokens and the label's
    left-aligned real tokens are contiguous, so the cumsum numbers the whole
    real span continuously regardless of padding side.
    """
    label_batch = pipeline.load(
        [_label_trace(text) for text in labels],
        max_length=pipeline.max_new_tokens,
        padding_side="right",
        add_special_tokens=False,
        use_chat_template=False,
    )
    joint = {
        k: torch.cat([v, label_batch[k].to(v.device)], dim=-1)
        for k, v in base_inputs.items()
        if k in ("input_ids", "attention_mask")
    }
    return ensure_position_ids(joint), label_batch["input_ids"]


def label_ce_loss(
    logits: torch.Tensor, label_ids: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """Cross-entropy over the label span: the positions predicting the ``L``
    label tokens are the last ``L+1 … 1`` logits (next-token shift), so slice
    ``[:, -L-1:-1]`` and ignore label padding."""
    n_labels = label_ids.shape[-1]
    sliced = logits[:, -n_labels - 1 : -1]
    return torch.nn.functional.cross_entropy(
        sliced.reshape(-1, sliced.shape[-1]),
        label_ids.reshape(-1).to(sliced.device),
        ignore_index=pad_token_id,
    )


def _ordered(edits: Sequence[Edit], model: StandardizedTransformer) -> list[Edit]:
    """Edits sorted by their site's forward position — the order one trace must
    request them in (each edit's own source reads are ordered by
    ``Edit.apply``; cross-edit source/site interleavings are PL1's)."""
    return sorted(edits, key=lambda e: forward_key(e.site.site, model))


@contextlib.contextmanager
def force_last_token_logits(hf_model: Any, n_positions: int) -> Iterator[None]:
    """Patch an HF causal-LM so its forward defaults to
    ``logits_to_keep=n_positions``, computing the lm_head only at the last
    ``n_positions`` positions.

    Saving full-sequence logits with the autograd graph costs
    ``seq × vocab`` activations plus the full-vocab lm_head compute over all
    positions; a consumer that only reads the trailing positions (the label
    span here, the final token in pullback) keeps identical values under a
    negative-index slice — the per-position lm_head rows are the same matmul
    (verified bit-identical). nnsight dispatches through the module's
    ``forward``, so the patch applies inside traces too. Models whose
    ``forward`` does not accept ``logits_to_keep`` are left untouched (the
    honest no-op fallback)."""
    if "logits_to_keep" not in inspect.signature(hf_model.forward).parameters:
        yield
        return
    orig = hf_model.forward

    @functools.wraps(orig)
    def _patched(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("logits_to_keep", n_positions)
        return orig(*args, **kwargs)

    hf_model.forward = _patched
    try:
        yield
    finally:
        hf_model.forward = orig


def traced_label_loss(
    model: StandardizedTransformer,
    inputs: dict[str, torch.Tensor],
    label_ids: torch.Tensor,
    edits: Sequence[Edit],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One edited forward under the pinned grad contract; returns
    ``(loss, pred_ids)``.

    The trace applies every edit (forward-ordered) and saves the logits
    **on-device with the autograd graph intact**; the loss slice and any
    ``backward()`` happen outside the trace (saved-logits backward). The
    forward runs under :func:`force_last_token_logits` with
    ``n_labels + 1`` positions — loss and ``pred_ids`` only ever read the
    last ``n_labels + 1`` logits, so trimming the lm_head there is
    value-identical while avoiding the full ``seq × vocab`` logits (and
    their graph) per step. ``pred_ids`` is the detached argmax over the
    label span — for accuracy metrics, not the loss.
    """
    n_labels = label_ids.shape[-1]
    with force_last_token_logits(model._model, n_labels + 1):
        with model.trace(inputs):
            for edit in _ordered(edits, model):
                edit.apply(model)
            logits = model.logits.save()
    loss = label_ce_loss(logits, label_ids, pad_token_id)
    pred_ids = logits[:, -n_labels - 1 : -1].argmax(dim=-1).detach().cpu()
    return loss, pred_ids


# --------------------------------------------------------------------------- #
#  DBM readout                                                                 #
# --------------------------------------------------------------------------- #
def selected_feature_ids(gate: MaskGate) -> list[int] | None:
    """The hard-threshold readout of a trained gate — which features the mask
    selected. Per-feature gates return the indices whose (hard) gate is on;
    a tied gate keeps pyvene's convention: ``None`` (= all features) when on,
    ``[]`` when off."""
    on = torch.sigmoid(gate.mask.detach()) > 0.5
    if gate.mask.numel() == 1:
        return None if bool(on.item()) else []
    return torch.nonzero(on.cpu()).flatten().tolist()
