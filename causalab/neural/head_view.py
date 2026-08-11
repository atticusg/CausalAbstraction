"""Per-head value / query / attention-output access for a transformer.

nnterp's ``StandardizedTransformer`` exposes attention only at whole-sublayer
granularity (``attentions_output[i]``). :class:`HeadView` adds the missing per-head
view by slicing the attention projections into ``head_dim``-wide vectors:

* **value** — a slice of the value projection's output (RoPE-free), addressed in
  **KV-head space** under grouped-query attention (one vector per KV head, shared by
  its query-head group);
* **query** — a slice of the query projection's output (pre-RoPE), in query-head space;
* **attention-value (sender)** — a slice of the output projection's *input* (the
  per-head attention output before the output projection), in query-head space.

``head_dim`` honours an explicit ``config.head_dim`` for models that decouple it from
``hidden // n_heads`` (e.g. Qwen3).

Two projection families are supported, sharing one flat-slice code path:

* **separate projections** (Llama, Qwen, all causalab GQA targets) — ``q_proj`` /
  ``v_proj`` outputs and ``o_proj`` input;
* **fused QKV** (GPT-2 ``c_attn``) — column slices of the fused projection output,
  laid out ``[q | k | v]`` (each ``hidden``-wide), and ``c_proj``'s input for the
  sender. The fused path deliberately slices the projection *module*'s output
  rather than reaching the per-head views through nnsight ``.source``: the source
  view ops (``value_states_view_*``) come out of ``.split()`` — multi-view outputs
  autograd refuses to mutate in place — so they are read-only, and their names
  track the transformers source line by line. Module-output slices are writable
  and stable.

:class:`HeadSite` wraps a single ``(kind, layer, head)`` into the
:class:`causalab.neural.site.Site` protocol (``read`` / ``write`` / ``collect`` /
forward rank), so head-addressed locations compose with residual/MLP sites in one
ordered pass (ST4).

Notes:

* **Forward order.** nnsight runs the trace body interleaved with the forward pass,
  so reads of different modules in one trace must be requested in execution order.
  Separate projections fire q → v → o-input, matching
  :data:`causalab.neural.site.INTRA_BLOCK_RANK`; on a fused model query and value
  collapse onto the single ``c_attn`` tap, which fires *first* — so rank resolution
  is model-aware (:meth:`HeadView.kind_rank`, :meth:`HeadSite.forward_rank_on`).
  :meth:`HeadView.collect` handles the ordering for you.
* **Unrecognized fused layouts refused.** The fused path assumes the equal-width
  ``[q | k | v]`` column layout (no GQA, coupled ``head_dim``); a fused model that
  breaks those assumptions raises :class:`NotImplementedError` (the honest
  boundary, like ``site._MLP_ACTIVATION_TAPS``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Sequence, get_args

import torch
from nnterp import StandardizedTransformer

from causalab.neural.site import (
    INTRA_BLOCK_RANK,
    Positions,
    _check_write_fits,
    _index_key,
    _sequence_index,
    _write_slice_shape,
    collect_ordered,
    forward_key,
    hf_text_config,
)

__all__ = ["HeadKind", "HeadSite", "HeadView"]

HeadKind = Literal["value", "query", "attention_value"]

# Forward-execution rank of each receiver's source module within one attention block
# for the *separate-projection* family: q_proj fires before v_proj fires before
# o_proj's input. Derived from the canonical intra-block ordering in
# :data:`causalab.neural.site.INTRA_BLOCK_RANK` (which reserves these slots between
# ``block_input`` and ``attention_output``), so per-head and whole-sublayer taps sort
# on one shared scale. Fused models reorder — use :meth:`HeadView.kind_rank`.
_KIND_RANK: dict[str, int] = {
    kind: INTRA_BLOCK_RANK[kind] for kind in ("query", "value", "attention_value")
}


class HeadView:
    """Per-head value/query/attention-output view over a ``StandardizedTransformer``.

    Standalone by design: it needs only the nnterp backbone (F1) — not the F3
    pipeline — plus the shared forward-order vocabulary and single-pass collect
    primitive from :mod:`causalab.neural.site`. :class:`HeadSite` wraps single
    head locations into the ``Site`` protocol on top of this view.
    """

    def __init__(self, model: StandardizedTransformer) -> None:
        self.model = model
        # The three fields nnterp standardizes come from its instance
        # attributes (set with ``raise_error=False``, hence the None-guards);
        # the two it doesn't (``num_key_value_heads``, ``head_dim``) are read
        # from the *text* config — a raw top-level read crashes or, worse,
        # silently falls back to the wrong GQA values on ``text_config``-
        # nesting models (e.g. Gemma3).
        cfg = hf_text_config(model)
        hidden_size = getattr(model, "hidden_size", None)
        self.hidden_size: int = int(
            hidden_size if hidden_size is not None else cfg.hidden_size
        )
        n_heads = getattr(model, "num_heads", None)
        self.n_heads: int = int(
            n_heads if n_heads is not None else cfg.num_attention_heads
        )
        self.n_kv_heads: int = int(
            getattr(cfg, "num_key_value_heads", None) or self.n_heads
        )
        self.head_dim: int = int(
            getattr(cfg, "head_dim", None) or (self.hidden_size // self.n_heads)
        )
        num_layers = getattr(model, "num_layers", None)
        self.num_layers: int = int(
            num_layers if num_layers is not None else cfg.num_hidden_layers
        )
        if self.n_heads % self.n_kv_heads != 0:  # pragma: no cover - malformed config
            raise ValueError(
                f"n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}"
            )

    # -- contract ---------------------------------------------------------------- #
    @property
    def group_size(self) -> int:
        """Query heads per KV head (``1`` when not grouped)."""
        return self.n_heads // self.n_kv_heads

    def kv_head_for(self, query_head: int) -> int:
        """The KV head a query head reads from — ``query_head // group_size``
        (identity when not grouped). The value receiver lives in this KV-head space,
        so the injected value reaches *every* query head in the group."""
        return query_head // self.group_size

    def head_column_slice(self, head: int) -> slice:
        """The ``[head*head_dim : (head+1)*head_dim]`` column slice of a flat
        projection output — the write handle :meth:`write` composes (with the
        fused-value column offset where needed)."""
        return slice(head * self.head_dim, (head + 1) * self.head_dim)

    @property
    def is_fused(self) -> bool:
        """True when attention has no separate ``q_proj``/``v_proj``/``o_proj``
        (fused QKV, e.g. GPT-2 ``c_attn``) — the receivers then slice columns of
        the fused projection output instead."""
        return not {"q_proj", "v_proj", "o_proj"}.issubset(self._attn_children())

    def n_heads_for(self, kind: HeadKind) -> int:
        """Valid head count for ``kind`` — KV heads for ``"value"`` (KV-head
        space), query heads otherwise."""
        return self.n_kv_heads if kind == "value" else self.n_heads

    def kind_rank(self, kind: HeadKind) -> int:
        """Forward-execution rank of ``kind``'s tap on *this* model, on the shared
        :data:`~causalab.neural.site.INTRA_BLOCK_RANK` scale. Separate projections
        fire q → v → o-input; on a fused model query and value both read the single
        ``c_attn`` output, which fires first — both collapse to the ``query`` rank
        (double-tapping one module in a trace is order-safe)."""
        if kind not in _KIND_RANK:
            raise ValueError(f"unknown head kind: {kind!r}")
        if self.is_fused and kind in ("query", "value"):
            return INTRA_BLOCK_RANK["query"]
        return _KIND_RANK[kind]

    # -- module handles ---------------------------------------------------------- #
    def _attn(self, layer: int):
        """The standardized ``self_attn`` envoy at ``layer``."""
        return self.model.model.layers[layer].self_attn

    def _attn_children(self) -> set[str]:
        return {name for name, _ in self._attn(0).named_children()}

    def _check_fused_layout(self) -> None:
        """The honest boundary for fused models: only the GPT-2-style equal-width
        ``[q | k | v]`` ``c_attn`` layout (no GQA, coupled ``head_dim``) is
        supported."""
        children = self._attn_children()
        supported = (
            {"c_attn", "c_proj"}.issubset(children)
            and self.n_kv_heads == self.n_heads
            and self.n_heads * self.head_dim == self.hidden_size
        )
        if not supported:
            raise NotImplementedError(
                f"fused-QKV attention (children={sorted(children)}, "
                f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
                f"head_dim={self.head_dim}, hidden={self.hidden_size}) does not "
                f"match the supported GPT-2-style layout (c_attn/c_proj with "
                f"equal-width [q|k|v] columns). Add its column layout to "
                f"HeadView._flat_handle."
            )

    def _check_head(self, kind: HeadKind, head: int) -> None:
        n = self.n_heads_for(kind)
        if not 0 <= head < n:
            space = "KV" if kind == "value" else "query"
            raise IndexError(
                f"head {head} out of range for {n} {space} heads ({kind!r})"
            )

    def _flat_handle(self, layer: int, kind: HeadKind) -> tuple[Any, int]:
        """In-trace flat projection proxy for ``kind`` plus the column offset of its
        first head. Separate projections expose each receiver on its own module;
        a fused ``c_attn`` output carries ``[q | k | v]`` columns, so value heads
        start at ``2 * hidden``. Reads *and* writes ride these module-level proxies
        (nnsight tracks their mutation; derived ``.source`` views are read-only)."""
        attn = self._attn(layer)
        if not self.is_fused:
            if kind == "query":
                return attn.q_proj.output, 0
            if kind == "value":
                return attn.v_proj.output, 0
            if kind == "attention_value":
                return attn.o_proj.input, 0
        else:
            self._check_fused_layout()
            if kind == "query":
                return attn.c_attn.output, 0
            if kind == "value":
                return attn.c_attn.output, 2 * self.hidden_size
            if kind == "attention_value":
                return attn.c_proj.input, 0
        raise ValueError(f"unknown head kind: {kind!r}")

    # -- in-trace proxy accessors (call inside `with head_view.model.trace(...)`) - #
    def heads(self, layer: int, kind: HeadKind):
        """Per-head proxy for ``kind`` at ``layer`` —
        ``(batch, seq, n_heads_for(kind), head_dim)``."""
        proxy, offset = self._flat_handle(layer, kind)
        n = self.n_heads_for(kind)
        width = n * self.head_dim
        return proxy[..., offset : offset + width].unflatten(-1, (n, self.head_dim))

    def queries(self, layer: int):
        """Per-head query proxy ``(batch, seq, n_heads, head_dim)`` — the query
        projection's output sliced (pre-RoPE, query-head space)."""
        return self.heads(layer, "query")

    def values(self, layer: int):
        """Per-head value proxy ``(batch, seq, n_kv_heads, head_dim)`` — the value
        projection's output sliced (RoPE-free, KV-head space)."""
        return self.heads(layer, "value")

    def attention_values(self, layer: int):
        """Per-head attention-output (sender) proxy ``(batch, seq, n_heads, head_dim)``
        — the output projection's *input* sliced (query-head space)."""
        return self.heads(layer, "attention_value")

    def query_proxy(self, layer: int, head: int):
        """Single query head ``(batch, seq, head_dim)``."""
        self._check_head("query", head)
        return self.queries(layer)[:, :, head, :]

    def value_proxy(self, layer: int, kv_head: int):
        """Single KV-head value ``(batch, seq, head_dim)`` — ``kv_head`` is a KV index."""
        self._check_head("value", kv_head)
        return self.values(layer)[:, :, kv_head, :]

    def attention_value_proxy(self, layer: int, head: int):
        """Single query head's attention output (sender) ``(batch, seq, head_dim)``."""
        self._check_head("attention_value", head)
        return self.attention_values(layer)[:, :, head, :]

    # -- in-trace write ----------------------------------------------------------- #
    def write(
        self,
        layer: int,
        kind: HeadKind,
        head: int,
        value: Any,
        positions: Positions | None = None,
    ) -> None:
        """In-trace write of ``value`` into one head's ``head_dim``-wide slot — the
        per-head intervention primitive (ED1 builds on it). Writes go through the
        flat projection proxy as a column-slice assignment
        (``proxy[:, positions, cols] = value``; whole sequence when ``positions``
        is ``None``) — the F4-validated pattern that reproduces a hand-rolled
        projection hook. ``positions`` accepts the same forms as ``Site.write``:
        a flat row broadcast over the batch, equal-width per-row ``(batch, k)``
        rows in the padded frame (the ST2 bridge output — ``value`` rows
        scatter each row's own indices), or ragged per-row rows (``value`` is
        the flat ``(total_positions, head_dim)`` form). ``head`` is a KV index
        for ``kind="value"``. A tensor ``value`` is first moved to the proxy's
        device and dtype (mirrors ``Site.write``). Call inside
        ``with model.trace(...):``."""
        self._check_head(kind, head)
        idx = _sequence_index(positions)
        proxy, offset = self._flat_handle(layer, kind)
        cols = slice(offset + head * self.head_dim, offset + (head + 1) * self.head_dim)
        if isinstance(value, torch.Tensor):
            value = value.to(device=proxy.device, dtype=proxy.dtype)
            # The head slot is a (batch, seq, head_dim) view of the flat
            # projection — same width-mismatch guard as ``Site.write`` (real
            # setitem raises opaquely; scan's fake tensors not at all).
            _check_write_fits(
                value,
                _write_slice_shape(
                    (proxy.shape[0], proxy.shape[1], self.head_dim), idx
                ),
                f"HeadSite({kind!r}, layer {layer}, head {head})",
            )
        if idx is None:
            proxy[:, :, cols] = value
        else:
            row_key, seq_key = _index_key(idx)
            proxy[row_key, seq_key, cols] = value

    # -- one-shot reads ---------------------------------------------------------- #
    def collect(self, inputs: Any, requests: Sequence[HeadSite]) -> list[torch.Tensor]:
        """Read every request in a single forward pass, ordered by forward position.

        Groups reads by ``(layer, kind)`` so each projection is tapped once, then
        lowers onto :func:`causalab.neural.site.collect_ordered` — the shared
        single-pass primitive that sorts taps by execution position (nnsight's
        forward-order constraint, with the fused reordering via
        :meth:`kind_rank`) and stops the forward after the deepest tap — and
        slices out each request's head. Returns concrete tensors
        ``(batch, seq, head_dim)`` aligned with ``requests``.
        """
        groups = sorted(
            {(r.layer, r.kind) for r in requests},
            key=lambda lk: (lk[0], self.kind_rank(lk[1])),
        )
        tensors = collect_ordered(
            self.model,
            inputs,
            [
                (
                    (layer, self.kind_rank(kind)),
                    lambda m, lyr=layer, k=kind: self.heads(lyr, k),
                )
                for layer, kind in groups
            ],
            offload=False,
        )
        saved = dict(zip(groups, tensors))
        return [saved[(r.layer, r.kind)][:, :, r.head, :] for r in requests]

    def collect_query(self, inputs: Any, layer: int, head: int) -> torch.Tensor:
        """One query head ``(batch, seq, head_dim)``."""
        return self.collect(inputs, [HeadSite("query", layer, head)])[0]

    def collect_value(self, inputs: Any, layer: int, kv_head: int) -> torch.Tensor:
        """One KV-head value ``(batch, seq, head_dim)`` (``kv_head`` is a KV index)."""
        return self.collect(inputs, [HeadSite("value", layer, kv_head)])[0]

    def collect_attention_value(
        self, inputs: Any, layer: int, head: int
    ) -> torch.Tensor:
        """One query head's attention output (sender) ``(batch, seq, head_dim)``."""
        return self.collect(inputs, [HeadSite("attention_value", layer, head)])[0]


@dataclasses.dataclass(frozen=True)
class HeadSite:
    """A per-head read/write location: ``kind`` at ``(layer, head)`` — the
    head-addressed counterpart of :class:`causalab.neural.site.Site`, speaking the
    same protocol (``read`` / ``write`` / ``collect`` / ``forward_rank`` /
    ``forward_rank_on``) so ED1/PL5 address heads and whole components uniformly,
    and :func:`causalab.neural.site.collect_sites` mixes both in one ordered pass.

    For ``kind="value"`` the ``head`` is a **KV-head** index (``0..n_kv_heads-1``,
    pyvene ``head_value_output`` parity); map a query head to its KV head with
    :meth:`HeadView.kv_head_for`. For ``"query"`` and ``"attention_value"`` the
    ``head`` is a query-head index (``0..n_heads-1``). Like ``Site``, it is a
    lightweight model-free spec — the ``StandardizedTransformer`` is supplied at
    read/write time, and ``positions`` are already-resolved token indices (ST2).
    """

    kind: HeadKind
    layer: int
    head: int

    def __post_init__(self) -> None:
        if self.kind not in get_args(HeadKind):
            raise ValueError(
                f"unknown head kind {self.kind!r}; expected one of {get_args(HeadKind)}"
            )
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")
        if self.head < 0:
            raise ValueError(f"head must be non-negative, got {self.head}")

    # -- contract ---------------------------------------------------------------- #
    @property
    def forward_rank(self) -> int:
        """Execution rank on the shared :data:`~causalab.neural.site.INTRA_BLOCK_RANK`
        scale for the *separate-projection* family (every causalab GQA target).
        Fused models reorder — ordering code passes through
        :meth:`forward_rank_on`, which resolves per model."""
        return INTRA_BLOCK_RANK[self.kind]

    def forward_rank_on(self, model: StandardizedTransformer) -> int:
        """Execution rank of this site's tap on ``model`` (fused-aware — see
        :meth:`HeadView.kind_rank`)."""
        return HeadView(model).kind_rank(self.kind)

    # -- in-trace accessors (call inside `with model.trace(...)`) ---------------- #
    def read(
        self, model: StandardizedTransformer, positions: Positions | None = None
    ) -> Any:
        """In-trace read proxy for this head's activation
        ``(batch, seq | k, head_dim)``, optionally sliced to ``positions`` on
        the sequence axis — the same position forms as ``Site.read``: a flat
        row broadcast over the batch (``proxy[:, positions]``), equal-width
        per-row ``(batch, k)`` rows (the ST2 bridge output) gathering each
        row's own indices, or ragged per-row rows yielding the flat
        ``(total_positions, head_dim)`` view. Call inside
        ``with model.trace(...):``."""
        idx = _sequence_index(positions)
        hv = HeadView(model)
        self._check_layer(hv)
        hv._check_head(self.kind, self.head)
        proxy = hv.heads(self.layer, self.kind)[:, :, self.head, :]
        if idx is None:
            return proxy
        return proxy[_index_key(idx)]

    def write(
        self,
        model: StandardizedTransformer,
        value: Any,
        positions: Positions | None = None,
    ) -> None:
        """In-trace write of ``value`` into this head's slot (whole sequence when
        ``positions`` is ``None``) — delegates to :meth:`HeadView.write`. Call
        inside ``with model.trace(...):``."""
        hv = HeadView(model)
        self._check_layer(hv)
        hv.write(self.layer, self.kind, self.head, value, positions)

    def _check_layer(self, hv: HeadView) -> None:
        if not 0 <= self.layer < hv.num_layers:
            raise IndexError(
                f"layer {self.layer} out of range for a {hv.num_layers}-layer model"
            )

    # -- one-shot read ----------------------------------------------------------- #
    def collect(
        self,
        model: StandardizedTransformer,
        inputs: Any,
        positions: Positions | None = None,
    ) -> torch.Tensor:
        """Read this head in a single forward pass and return a concrete **CPU**
        tensor ``(batch, seq | k, head_dim)`` — the exact mirror of
        ``Site.collect`` (early stop after the tap, CPU offload convention)."""
        hv = HeadView(model)  # bounds-check before the trace: fail fast, unwrapped
        self._check_layer(hv)
        hv._check_head(self.kind, self.head)
        return collect_ordered(
            model,
            inputs,
            [(forward_key(self, model), lambda m: self.read(m, positions))],
        )[0]
