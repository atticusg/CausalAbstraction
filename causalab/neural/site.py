"""Component/layer → nnterp accessor + position slice: the Site core.

A :class:`Site` names *where* to read or write in a transformer — a
``(component, layer)`` pair — and resolves it to nnterp's standardized accessor
plus an optional token-position slice, over a ``StandardizedTransformer``. The
component vocabulary is deliberately the pyvene-era component-location strings
(``block_input``/``block_output``/``mlp_*``/``attention_output``), so the
migration was 1:1, plus ``embeddings`` for the token-embedding output.

Seven components are covered:

* ``embeddings`` — the token-embedding output (nnterp's settable
  ``token_embeddings`` property). Layer-less: construct with ``layer=0`` and it
  sorts before every block;
* ``block_input`` / ``block_output`` — the residual stream entering / leaving a
  decoder block (nnterp ``layers_input[i]`` / ``layers_output[i]``);
* ``mlp_input`` / ``mlp_output`` — the MLP sublayer's input / output
  (``mlps_input[i]`` / ``mlps_output[i]``);
* ``mlp_activation`` — the architecture's intermediate MLP activation. nnterp
  exposes **no** named accessor for it, so it is reached by a raw submodule tap
  resolved through :data:`_MLP_ACTIVATION_TAPS` (extend that table to onboard a
  new MLP shape). **The two supported families expose different tensors under
  this one name** — inherited 1:1 from the pyvene component and the oracle in
  ``tests/neural/activations/hook_oracle.py``: SwiGLU Llama taps ``act_fn``'s
  *output*, which is ``act(gate_proj(x))`` and is **not** the tensor entering
  ``down_proj`` (that is ``act(gate) * up``); GPT-2 taps ``c_proj``'s *input*,
  which **is** the down-projection's input;
* ``attention_output`` — the whole attention sublayer output, all heads
  (``attentions_output[i]``). Per-**head** value/query views are a separate
  concern handled by :mod:`causalab.neural.head_view` (F4/ST4).

Notes
-----
* **Standalone by design.** Like :mod:`causalab.neural.head_view`, this needs only
  the nnterp backbone (F1) + F3's ``StandardizedTransformer`` — not the pyvene
  layer, the featurizer (ST3), or the ``Template`` position-resolver machinery.
  Those compose *around* this core: :meth:`Site.read` / :meth:`Site.write` accept
  already-resolved integer ``positions`` — either one flat row applied to every
  batch element, or per-row indices in the padded-batch frame as produced by the
  ST2 bridge (:func:`causalab.neural.positions.resolve_positions`) — and the
  featurize/inverse wrap is applied by :mod:`causalab.neural.featurized_site`
  (ST3), not here.
* **Forward order.** nnsight runs the trace body interleaved with the forward pass,
  so reads of different modules in one trace must be requested in execution order.
  :data:`INTRA_BLOCK_RANK` is the canonical intra-block ordering vocabulary — its
  ranks are gapped so ST4's per-head taps (``query``/``key``/``value``/
  ``attention_value``, already reserved) slot between ``block_input`` and
  ``attention_output`` without renumbering; ``head_view._KIND_RANK`` derives from
  it. :func:`collect_ordered` is the shared single-pass primitive (sort taps, one
  trace, save, stop early) that :func:`collect_sites` and ``HeadView.collect``
  both lower onto; a single :meth:`Site.collect` never trips the ordering.
* **Writes** mutate the *read* proxy in place (``proxy[:, positions] = value``,
  or ``proxy[:] = value`` for a whole-tensor replacement). Both branches therefore
  ride the same tuple-safe accessor **read** (nnterp unwraps tuple outputs on
  ``__getitem__``), so a write never depends on the accessor's lazily-detected
  tuple state. ``value`` is first moved to the site's device and dtype — on an
  ``hf_device_map``-sharded model each layer may live on its own GPU, and the
  underlying ``index_put_``/``copy_`` require an exact device/dtype match
  (nnterp's ``steer`` uses the same ``.to(layer_device)`` pattern).
* **One-shot reads** (:meth:`Site.collect` / :func:`collect_sites`) return CPU
  tensors (the package convention for collected activations) and stop the forward
  pass after the deepest tap — a shallow read never pays for the remaining layers
  or the unembed.
"""

from __future__ import annotations

import collections.abc
import dataclasses
from typing import Any, Callable, Literal, Protocol, Sequence, Union, cast, get_args

import torch
from nnterp import StandardizedTransformer

__all__ = [
    "Component",
    "COMPONENTS",
    "INTRA_BLOCK_RANK",
    "Positions",
    "RaggedIndex",
    "Site",
    "SiteLike",
    "backbone_has_edits",
    "WritableSite",
    "collect_ordered",
    "collect_sites",
    "forward_key",
    "hf_text_config",
]


def backbone_has_edits(model: Any) -> bool:
    """True when the nnsight backbone carries persistent default-graph edits
    (``model.edit()`` — managed by :mod:`causalab.neural.persistent`).

    The shared probe behind the early-stop guards: ``tracer.stop()`` ends the
    forward at the deepest tap, and a persistent edit whose site fires later
    then never receives its module event — nnsight raises
    ``MissedProviderError`` at trace exit (measured on 0.7). Collect paths
    therefore run the full forward whenever this is true. Reads the backbone's
    own state (``_default_mediators``), not the causalab registry, so edits
    installed out-of-band are guarded too. Lives here (not in ``persistent``)
    because this module sits below :mod:`causalab.neural.persistent` in the
    import graph and both :func:`collect_ordered` and the plan compiler need
    it.
    """
    return bool(getattr(model, "_default_mediators", None))


#: What ``Site.read`` / ``Site.write`` accept as token positions: one flat row of
#: indices applied to every batch element (``Sequence[int]`` / 1-D tensor), or
#: per-row indices in the padded-batch frame (``Sequence[Sequence[int]]`` / 2-D
#: ``(batch, k)`` tensor) as produced by the ST2 bridge
#: (:func:`causalab.neural.positions.resolve_positions`). Per-row rows may be
#: **ragged** (different widths per example): they batch as one flat
#: gather/scatter and read/write a flat ``(total_positions, hidden)`` view —
#: see :class:`RaggedIndex`.
Positions = Union[Sequence[int], Sequence[Sequence[int]], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class RaggedIndex:
    """Flat advanced index over ragged per-row positions (PL3, #405).

    ``row_ids[j]`` / ``col_ids[j]`` address the ``j``-th selected position's
    (batch row, sequence index); ``widths`` records how many positions each
    example contributed, in row order, so consumers can re-nest the flat
    ``(total_positions, hidden)`` view per example (``torch.split(value,
    widths)``). One ``proxy[row_ids, col_ids]`` gather/scatter expresses what
    equal-width rows express as a ``(batch, k)`` slice — no per-example
    fallback, no length-bucketing.
    """

    row_ids: torch.Tensor
    col_ids: torch.Tensor
    widths: tuple[int, ...]


def _sequence_index(positions: Positions | None) -> Any:
    """Normalize ``positions`` into an index for the sequence axis.

    Returns ``None`` (all positions), a flat index (uniform across the batch,
    applied as ``proxy[:, idx]``), a ``(batch, k)`` LongTensor of equal-width
    per-row indices (applied as ``proxy[arange(batch)[:, None], idx]``), or a
    :class:`RaggedIndex` when per-row widths differ (applied as one flat
    ``proxy[row_ids, col_ids]`` gather/scatter yielding
    ``(total_positions, ...)``).
    """
    if positions is None:
        return None
    if isinstance(positions, torch.Tensor):
        if positions.dim() == 1:
            return positions
        if positions.dim() == 2:
            return positions.long()
        raise ValueError(
            f"positions tensor must be 1-D (uniform across the batch) or 2-D "
            f"(per-row), got {positions.dim()}-D"
        )
    rows = list(positions)
    if rows and isinstance(rows[0], collections.abc.Sequence):
        nested = cast("list[Sequence[int]]", rows)
        widths = tuple(len(row) for row in nested)
        if len(set(widths)) > 1:
            return RaggedIndex(
                row_ids=torch.tensor(
                    [i for i, row in enumerate(nested) for _ in row],
                    dtype=torch.long,
                ),
                col_ids=torch.tensor(
                    [int(p) for row in nested for p in row], dtype=torch.long
                ),
                widths=widths,
            )
        return torch.tensor([[int(p) for p in row] for row in nested], dtype=torch.long)
    return rows


def _index_key(idx: Any) -> Any:
    """The subscript addressing a normalized position index on the sequence
    axis — usable for both reads (``proxy[key]``) and in-place writes
    (``proxy[key] = value``). Flat rows broadcast over the batch
    (``[:, idx]``); a per-row ``(batch, k)`` index pairs with an
    ``arange(batch)`` column so row ``i`` gathers/scatters its own indices; a
    :class:`RaggedIndex` pairs its flat row/column ids, selecting
    ``(total_positions, ...)``."""
    if isinstance(idx, RaggedIndex):
        return idx.row_ids, idx.col_ids
    if isinstance(idx, torch.Tensor) and idx.dim() == 2:
        return torch.arange(idx.shape[0]).unsqueeze(1), idx
    return slice(None), idx


def _write_slice_shape(proxy_shape: Sequence[int], idx: Any) -> tuple[int, ...]:
    """The shape of the slice a positional write scatters into, for a
    normalized index (:func:`_sequence_index` output): the whole activation
    (``idx=None``), ``(batch, k, ...)`` for flat or equal-width per-row
    indices, or the flat ``(total_positions, ...)`` view for a
    :class:`RaggedIndex`."""
    shape = tuple(int(s) for s in proxy_shape)
    if idx is None:
        return shape
    if isinstance(idx, RaggedIndex):
        return (int(idx.row_ids.shape[0]),) + shape[2:]
    if isinstance(idx, torch.Tensor):
        k = int(idx.shape[1]) if idx.dim() == 2 else int(idx.shape[0])
        return (shape[0], k) + shape[2:]
    return (shape[0], len(idx)) + shape[2:]


def _check_write_fits(
    value: torch.Tensor, slice_shape: tuple[int, ...], where: str
) -> None:
    """Refuse a positional write whose ``value`` cannot broadcast to the
    selected slice — the width-mismatch class (e.g. a source read whose
    per-example position widths differ from the write positions', the classic
    variable-length base/counterfactual failure).

    In a real forward the underlying ``index_put_``/``copy_`` raises anyway
    (opaquely, possibly as a CUDA assert); under ``model.scan()`` fake tensors
    do **not** value-check advanced-indexing writes, so without this explicit
    check the scan preflight (:mod:`causalab.neural.preflight`) would miss the
    mismatch entirely. One check, both modes, one legible error. The rule is
    setitem's one-way broadcast: right-aligned, every value dim ``1`` or equal,
    no extra leading dims.
    """
    vshape = tuple(int(s) for s in value.shape)
    fits = len(vshape) <= len(slice_shape) and all(
        v == 1 or v == t for v, t in zip(reversed(vshape), reversed(slice_shape))
    )
    if not fits:
        raise ValueError(
            f"write to {where}: value of shape {vshape} does not broadcast to "
            f"the {slice_shape} slice its positions select. Widths must pair "
            "up: a source read must contribute the same number of positions "
            "per example as the write positions address (a multi-token "
            "variable resolved on variable-length base/counterfactual pairs "
            "is the classic mismatch), or the value must be a broadcastable "
            "vector."
        )


Component = Literal[
    "embeddings",
    "block_input",
    "attention_output",
    "mlp_input",
    "mlp_activation",
    "mlp_output",
    "block_output",
]

#: The site components, in intra-block forward-execution order (derived from the
#: ``Component`` literal — the single declaration of the vocabulary).
COMPONENTS: tuple[Component, ...] = get_args(Component)

#: Canonical intra-block forward-execution ranks, shared across the neural layer:
#: within one decoder block the residual enters (``block_input``), attention runs
#: (whose per-head internals — consumed by ``head_view`` — fire as q → k → v →
#: o-input on separate-projection models; fused-QKV models reorder, which is why
#: ordering code resolves ranks through ``forward_rank_on(model)``), then the MLP
#: (``mlp_input`` → ``mlp_activation`` → ``mlp_output``), then the residual leaves
#: (``block_output``). ``embeddings`` precedes every block. Ranks are gapped by 10
#: so finer taps can slot in without renumbering the published ordering; key with
#: ``layer`` (as :func:`collect_sites` does) to order taps across layers too.
INTRA_BLOCK_RANK: dict[str, int] = {
    "embeddings": -10,
    "block_input": 0,
    "query": 10,  # per-head q-projection output (head_view.HeadSite)
    "key": 20,  # reserved: per-head k_proj output (no consumer yet)
    "value": 30,  # per-head v-projection output (head_view.HeadSite)
    "attention_probabilities": 35,  # softmax(QK^T) probs (attention_probs, CAP4)
    "attention_value": 40,  # per-head o-projection input (head_view.HeadSite)
    "attention_output": 50,
    "mlp_input": 60,
    "mlp_activation": 70,
    "mlp_output": 80,
    "block_output": 90,
}

# nnterp ``StandardizedTransformer`` accessor attribute for each component that has
# a per-layer one. ``embeddings`` uses the layer-less ``token_embeddings`` property;
# ``mlp_activation`` has no named accessor and is tapped by a raw submodule path
# (see ``_MLP_ACTIVATION_TAPS``).
_ACCESSOR: dict[str, str] = {
    "block_input": "layers_input",
    "block_output": "layers_output",
    "mlp_input": "mlps_input",
    "mlp_output": "mlps_output",
    "attention_output": "attentions_output",
}

#: ``(child-module name, which side of it carries the activation)`` — the
#: ``mlp_activation`` tap registry, tried in order against the MLP's children.
#: Onboarding a new MLP shape is one entry here (mirrored in the hook oracle),
#: not an edit to dispatch logic. See the module docstring for the semantic
#: caveat: the tapped tensor differs across families.
_MLP_ACTIVATION_TAPS: tuple[tuple[str, Literal["input", "output"]], ...] = (
    ("act_fn", "output"),  # SwiGLU family (Llama/Qwen/Mistral/Gemma): act(gate)
    ("c_proj", "input"),  # GPT-2: the down-projection's input
)

# The tables above are three views of one vocabulary — fail at import time if a
# component is added to one but not the others.
assert set(COMPONENTS) <= set(INTRA_BLOCK_RANK), "component missing a forward rank"
assert set(_ACCESSOR) == set(COMPONENTS) - {"embeddings", "mlp_activation"}, (
    "accessor table out of sync with the component vocabulary"
)


class SiteLike(Protocol):
    """What :func:`collect_sites` needs from a site: a layer, a model-aware
    forward rank, and an in-trace read. :class:`Site` and
    :class:`causalab.neural.head_view.HeadSite` both satisfy it (structurally —
    head_view imports this module, so the protocol lives here and stays
    import-cycle-free). ``layer`` is read-only — every implementation is a
    frozen dataclass."""

    @property
    def layer(self) -> int: ...

    def forward_rank_on(self, model: StandardizedTransformer) -> int: ...

    def read(
        self, model: StandardizedTransformer, positions: Positions | None = None
    ) -> Any: ...


def forward_key(site: SiteLike, model: StandardizedTransformer) -> tuple[int, int]:
    """The ``(layer, forward_rank_on(model))`` sort key — the forward-order
    currency every trace scheduler orders taps and edits by
    (:func:`collect_ordered`'s tap keys, the plan compiler's op ranks,
    ``Edit``'s read-source ordering, ``trainable``'s edit ordering).

    One derivation, on the site layer, so the ordering cannot drift between
    consumers. Resolving the rank through ``forward_rank_on(model)`` keeps
    per-head taps correct on fused-QKV models.
    """
    return (site.layer, site.forward_rank_on(model))


class WritableSite(SiteLike, Protocol):
    """A :class:`SiteLike` that can also be written — what
    :class:`causalab.neural.featurized_site.FeaturizedSite` wraps. :class:`Site`
    and :class:`causalab.neural.head_view.HeadSite` both satisfy it, so
    featurized (and Edit/Plan) machinery addresses whole components and single
    heads uniformly (ST4)."""

    @property
    def forward_rank(self) -> int: ...

    def write(
        self,
        model: StandardizedTransformer,
        value: Any,
        positions: Positions | None = None,
    ) -> None: ...


def hf_text_config(model: StandardizedTransformer) -> Any:
    """The HF config carrying the *text-model* fields — ``config.text_config``
    when the architecture nests them under a multimodal wrapper (e.g. Gemma3),
    else the top-level config (nnterp's ``text_config`` rule from
    ``nnterp.rename_utils``). The config is resolved from the standardized
    model or its underlying ``_model`` (wrapping a pre-loaded module can leave
    ``model.config`` unset). Raises :class:`ValueError` when no config is
    found. Reading GQA/head fields (``num_key_value_heads``, ``head_dim``)
    from the raw top-level config is silently wrong on nesting models — route
    them through this helper."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(getattr(model, "_model", None), "config", None)
    if cfg is None:
        raise ValueError("StandardizedTransformer exposes no HF config")
    nested = getattr(cfg, "text_config", None)
    return nested if nested is not None else cfg


def _num_layers(model: StandardizedTransformer) -> int | None:
    """The model's layer count — the standardized ``num_layers`` when exposed,
    falling back to the (text-)config (``num_hidden_layers`` resolves on GPT-2
    too via ``GPT2Config.attribute_map``)."""
    n = getattr(model, "num_layers", None)
    if n is None:
        try:
            cfg = hf_text_config(model)
        except ValueError:
            return None
        n = getattr(cfg, "num_hidden_layers", None)
    return int(n) if n is not None else None


@dataclasses.dataclass(frozen=True)
class Site:
    """A read/write location: ``component`` at ``layer``.

    A lightweight, model-free spec — the ``StandardizedTransformer`` is supplied at
    read/write time, so one :class:`Site` addresses the same location on any model
    that carries the standardized accessors. ``positions`` are passed per call as
    already-resolved token indices (``Template``/``TokenPosition`` resolution is
    ST2); the featurizer wrap is ST3. ``embeddings`` is layer-less — construct it
    with ``layer=0``.
    """

    component: Component
    layer: int

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(
                f"unknown component {self.component!r}; expected one of {COMPONENTS}"
            )
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")
        if self.component == "embeddings" and self.layer != 0:
            raise ValueError(
                "embeddings is layer-less; construct Site('embeddings', 0)"
            )

    # -- contract ---------------------------------------------------------------- #
    @property
    def forward_rank(self) -> int:
        """Execution rank of this component within one decoder block (see
        :data:`INTRA_BLOCK_RANK`). :func:`collect_sites` sorts by
        ``(layer, forward_rank_on(model))`` to honour nnsight's forward-order
        constraint."""
        return INTRA_BLOCK_RANK[self.component]

    def forward_rank_on(self, model: StandardizedTransformer) -> int:
        """Execution rank of this site's tap on ``model``. Whole-component ranks
        are architecture-independent, so this is :attr:`forward_rank`; it exists
        for protocol symmetry with ``HeadSite``, whose per-head taps reorder on
        fused-QKV models."""
        return self.forward_rank

    @property
    def is_mlp_activation(self) -> bool:
        """True for the one component with no named nnterp accessor (reached by a
        raw, architecture-specific submodule tap)."""
        return self.component == "mlp_activation"

    def _check_layer(self, model: StandardizedTransformer) -> None:
        if self.component == "embeddings":  # layer pinned to 0 in __post_init__
            return
        n = _num_layers(model)
        if n is not None and not 0 <= self.layer < n:
            raise IndexError(f"layer {self.layer} out of range for a {n}-layer model")

    # -- mlp_activation submodule tap -------------------------------------------- #
    def mlp_activation_kind(
        self, model: StandardizedTransformer
    ) -> tuple[str, Literal["input", "output"]]:
        """The submodule + I/O side that expose ``mlp_activation``, resolved per
        architecture through :data:`_MLP_ACTIVATION_TAPS` (trace-free, so
        unit-testable). Raises :class:`NotImplementedError` for an MLP shape the
        registry doesn't cover (the honest boundary, like ``head_view``'s
        fused-QKV refusal) — onboarding one is a registry entry, not new logic."""
        mlp = model.model.layers[self.layer].mlp
        children = {name for name, _ in mlp.named_children()}
        for submodule, io in _MLP_ACTIVATION_TAPS:
            if submodule in children:
                return submodule, io
        raise NotImplementedError(
            f"mlp_activation has no standardized accessor and this MLP "
            f"(children={sorted(children)}) matches no entry in "
            f"site._MLP_ACTIVATION_TAPS. Add its intermediate-activation tap there "
            f"(and mirror it in tests/neural/activations/hook_oracle.py)."
        )

    # -- in-trace accessors (call inside `with model.trace(...)`) ---------------- #
    def _proxy(self, model: StandardizedTransformer) -> Any:
        """The in-trace read/write handle for this site's full activation."""
        if self.component == "embeddings":
            return model.token_embeddings
        if self.is_mlp_activation:
            submodule, io = self.mlp_activation_kind(model)
            node = getattr(model.model.layers[self.layer].mlp, submodule)
            return getattr(node, io)
        return getattr(model, _ACCESSOR[self.component])[self.layer]

    def read(
        self, model: StandardizedTransformer, positions: Positions | None = None
    ) -> Any:
        """In-trace read proxy for this site's activation, optionally sliced to
        ``positions`` on the sequence axis. A flat row reads the same indices in
        every batch element (``proxy[:, positions]`` — rank-agnostic, so future
        non-3-D sites slice the same way); equal-width per-row positions (the
        padded-frame rows :func:`causalab.neural.positions.resolve_positions`
        produces) gather each row's own indices, yielding ``(batch, k, ...)``;
        ragged per-row positions gather as one flat advanced index, yielding
        ``(total_positions, ...)`` (:class:`RaggedIndex` — split by its
        ``widths`` to re-nest per example). ``positions=None`` reads all
        positions. Call inside ``with model.trace(...):``."""
        idx = _sequence_index(positions)
        self._check_layer(model)
        proxy = self._proxy(model)
        if idx is None:
            return proxy
        return proxy[_index_key(idx)]

    def write(
        self,
        model: StandardizedTransformer,
        value: Any,
        positions: Positions | None = None,
    ) -> None:
        """In-trace write of ``value`` to this site. With ``positions`` it is an
        in-place slice write (``proxy[:, positions] = value`` for a flat row;
        per-row positions scatter each row's own indices, mirroring
        :meth:`read` — for ragged rows ``value`` is the flat
        ``(total_positions, ...)`` form); with ``positions=None`` the whole activation is replaced
        in place (``proxy[:] = value``). All branches mutate the read proxy —
        nnsight tracks the mutation and propagates it into the forward pass — so
        whole-tensor replacement rides the same tuple-safe accessor read as
        positional writes. A tensor ``value`` is first moved to the site's
        device and dtype (sharded ``hf_device_map`` models place layers on
        different devices; a silent dtype cast is intended — intervention values
        follow the activation's precision). Call inside
        ``with model.trace(...):``."""
        idx = _sequence_index(positions)
        self._check_layer(model)
        proxy = self._proxy(model)
        if isinstance(value, torch.Tensor):
            value = value.to(device=proxy.device, dtype=proxy.dtype)
            _check_write_fits(
                value,
                _write_slice_shape(proxy.shape, idx),
                f"Site({self.component!r}, layer {self.layer})",
            )
        if idx is None:
            proxy[:] = value
        else:
            proxy[_index_key(idx)] = value

    # -- one-shot read ----------------------------------------------------------- #
    def collect(
        self,
        model: StandardizedTransformer,
        inputs: Any,
        positions: Positions | None = None,
    ) -> torch.Tensor:
        """Read this site in a single forward pass and return a concrete **CPU**
        tensor ``(batch, seq | len(positions), hidden)``. The forward stops right
        after the tap (:func:`collect_ordered`), so a shallow read never pays for
        the remaining layers or the unembed."""
        return collect_ordered(
            model,
            inputs,
            [(forward_key(self, model), lambda m: self.read(m, positions))],
        )[0]


def collect_ordered(
    model: StandardizedTransformer,
    inputs: Any,
    taps: Sequence[tuple[tuple[int, int], Callable[[StandardizedTransformer], Any]]],
    *,
    offload: bool = True,
) -> list[Any]:
    """Run every ``((layer, intra_block_rank), read_fn)`` tap in **one** forward
    pass, saving each read in forward-execution order and stopping the pass after
    the deepest tap.

    This is the shared single-pass primitive under :func:`collect_sites` and
    ``HeadView.collect`` — and the building block the plan compiler (PL1) grows
    from. It owns the two trace-correctness contracts in one place: taps are
    *sorted* by their ``(layer, rank)`` key before reading (an unordered read list
    would raise nnsight's ``MissedProviderError``), and ``tracer.stop()`` ends the
    forward after the last save (no wasted layers/unembed on collect-only passes)
    — unless the model carries persistent edits (:func:`backbone_has_edits`):
    stopping before a deeper edit's module event strands its mediator, so an
    edited model pays the full forward instead.
    Results align with ``taps`` (not with the internal read order); ``offload=True``
    (the package convention for collected activations) moves each save to CPU so a
    large tap list never accumulates on-device memory.
    """
    order = sorted(range(len(taps)), key=lambda i: taps[i][0])
    stop_early = not backbone_has_edits(model)
    saved: list[Any] = [None] * len(taps)
    with model.trace(inputs) as tracer:
        for i in order:
            value = taps[i][1](model)
            saved[i] = (value.cpu() if offload else value).save()
        if stop_early:
            tracer.stop()
    return saved


def collect_sites(
    model: StandardizedTransformer,
    inputs: Any,
    sites: Sequence[SiteLike],
    positions: Sequence[Positions | None] | None = None,
) -> list[torch.Tensor]:
    """Read several sites in **one** forward pass, ordered by forward position.

    Accepts any mix of :class:`SiteLike` sites (whole-component :class:`Site`,
    per-head ``HeadSite``), lowers onto :func:`collect_ordered` (forward-order
    sorting + early stop) and returns concrete **CPU** tensors **aligned with
    ``sites``**. Ordering resolves through ``forward_rank_on(model)`` so per-head
    taps sort correctly on fused-QKV models too. ``positions`` is an optional
    per-site list of resolved indices (``None`` = all positions for that site);
    it must match ``sites`` in length. Every position form works with every
    site — ``Site`` and ``HeadSite`` share the flat / equal-width per-row /
    ragged normalization (:func:`_sequence_index`).
    """
    if positions is not None and len(positions) != len(sites):
        raise ValueError(
            f"positions has length {len(positions)} but there are {len(sites)} sites"
        )
    pos_for = [None] * len(sites) if positions is None else list(positions)
    return collect_ordered(
        model,
        inputs,
        [
            (
                forward_key(site, model),
                lambda m, site=site, pos=pos: site.read(m, cast(Any, pos)),
            )
            for site, pos in zip(sites, pos_for)
        ],
    )
