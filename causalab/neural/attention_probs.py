"""Attention-probability editing over nnterp's editable trace target — CAP4 (#457).

causalab previously only *read* attention patterns, via a raw HF forward with
``output_attentions=True`` (:mod:`causalab.methods.attention_pattern_analysis`).
nnterp's ``attention_probabilities[i]`` exposes the post-softmax pattern —
``(batch, n_heads, query_seq, key_seq)`` — as an **editable** in-trace target,
which this module wires into the Site/Edit engine:

* :class:`AttentionProbabilitiesSite` — the whole pattern at one layer as a
  :class:`~causalab.neural.site.WritableSite`, so it composes with the
  existing machinery unchanged: wrap it in a
  :class:`~causalab.neural.featurized_site.FeaturizedSite`, put it on an
  :class:`~causalab.neural.edit.Edit`, chain edits across layers through
  :class:`~causalab.neural.plan.Plan`/:func:`~causalab.neural.plan.run_plan`
  (``EditOp``/``CollectOp``), or mix it into
  :func:`~causalab.neural.site.collect_sites` reads.
* :func:`knockout` / :func:`renormalize` — the two write-side intervention
  modes over that site, as thin :class:`Edit` constructors in the
  :mod:`causalab.neural.modes` style.

Load-time gating
----------------
The pattern only materializes under the **eager** attention kernel (sdpa/flash
never build the ``(seq, seq)`` probability matrix), so the model must be loaded
with ``enable_attention_probs=True`` — surfaced through
``LMPipeline(..., enable_attention_probs=True)`` and
:func:`causalab.io.pipelines.load_pipeline` — which makes nnterp force
``attn_implementation="eager"`` (the pipeline-wide default is sdpa since SH3,
#424). At that load nnterp also runs its ``attention_probabilities
.check_source()`` causal-validation gate (the F2-deferred adoption, #393): the
probabilities have the expected shape, sum to 1, and *modifying them changes
the logits* — so an enabled accessor is a **validated** editable target, not
just a resolvable module path. Caveat: nnterp only runs that gate under
``check_renaming=True`` (the default); with ``check_renaming=False`` it
disables the accessor outright.

Semantics
---------
``knockout`` zeroes the selected ``(head, query, key)`` block of the pattern —
a single edge, a column (e.g. "no head may attend to BOS"), or the whole
pattern. With ``redistribute=True`` (the default) each *affected* query row is
rescaled to sum to 1, redistributing the knocked-out mass over the surviving
keys proportionally; with ``redistribute=False`` the mass is simply removed
(the row is no longer a distribution — the attention output shrinks). A row
whose entire support is knocked out has nothing to rescale and stays zero.
``renormalize`` (the standalone mode) rescales the selected rows to sum to 1
without zeroing anything — the complement half, for re-normalizing after an
external pattern write.

Notes
-----
* **Positions address the query axis.** ``read``/``write``/``collect`` accept
  one flat row of already-resolved indices applied to every batch element
  (``proxy[:, :, positions]``); the per-row and ragged forms of
  :data:`~causalab.neural.site.Positions` are refused (the honest boundary —
  a per-row gather on axis 2 under a leading batched-heads slice has no
  clean advanced-indexing form; slice inside ``g`` instead). The mode
  constructors themselves never use site positions: head/query/key selection
  happens in ``g`` on the full pattern, where negative indices resolve
  against the run's actual sequence length.
* **Forward order.** The pattern fires inside the attention block — after the
  q/k/v projections, before the o-projection input — so the site ranks at
  :data:`~causalab.neural.site.INTRA_BLOCK_RANK`'s ``attention_probabilities``
  slot (35), between ``value`` (30) and ``attention_value`` (40).
* **No feature space.** The pattern is a probability simplex, not a feature
  vector; both constructors refuse a non-trivial featurizer or
  ``feature_ids`` on a passed-in :class:`FeaturizedSite`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, Union

import torch
from nnterp import StandardizedTransformer

from causalab.neural.edit import Edit
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.site import (
    INTRA_BLOCK_RANK,
    Positions,
    _num_layers,
    _sequence_index,
    collect_ordered,
    forward_key,
)

__all__ = ["AttentionProbabilitiesSite", "knockout", "renormalize"]

#: What the mode constructors accept as a head / query / key selection: one
#: index, a sequence of indices (negatives resolve against the run's actual
#: axis length, as in plain tensor indexing), or ``None`` for the whole axis.
AxisSelection = Union[int, Sequence[int], None]


def _flat_index(positions: Positions | None, what: str) -> Any:
    """Normalize ``positions`` to the one form this site supports: a flat row
    of query indices shared by every batch element. Per-row (equal-width or
    ragged) forms are refused — see the module docstring."""
    idx = _sequence_index(positions)
    if idx is None:
        return None
    if isinstance(idx, list) or (isinstance(idx, torch.Tensor) and idx.dim() == 1):
        return idx
    raise NotImplementedError(
        f"{what} supports only a flat row of query positions shared by the "
        "batch; per-row / ragged positions are not supported on the "
        "attention-probability pattern — slice inside the edit's g instead."
    )


@dataclasses.dataclass(frozen=True)
class AttentionProbabilitiesSite:
    """The post-softmax attention pattern at ``layer`` —
    ``(batch, n_heads, query_seq, key_seq)`` — as a
    :class:`~causalab.neural.site.WritableSite`.

    A lightweight, model-free spec like :class:`~causalab.neural.site.Site`:
    the ``StandardizedTransformer`` is supplied at read/write time and must
    have been loaded with ``enable_attention_probs=True``
    (``LMPipeline(..., enable_attention_probs=True)``) — an unavailable
    accessor fails fast with the load-flag remedy. ``positions`` (optional,
    everywhere) are already-resolved **query** indices, one flat row applied
    to every batch element.
    """

    layer: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")

    # -- contract ---------------------------------------------------------------- #
    @property
    def forward_rank(self) -> int:
        """Execution rank within one decoder block (see
        :data:`~causalab.neural.site.INTRA_BLOCK_RANK`): the pattern fires
        after the q/k/v projections and before the o-projection input."""
        return INTRA_BLOCK_RANK["attention_probabilities"]

    def forward_rank_on(self, model: StandardizedTransformer) -> int:
        """Architecture-independent (the softmax always fires between the
        projections and the value mix), so this is :attr:`forward_rank` —
        protocol symmetry with ``Site`` / ``HeadSite``."""
        return self.forward_rank

    def _check_available(self, model: StandardizedTransformer) -> None:
        # nnterp stores the disabled state on ``accessor.enabled`` (read here);
        # ``model.attn_probs_available`` is its public proxy of the same bit
        # (asserted consistent by the property tier). A model without the
        # accessor at all (not a ``StandardizedTransformer``) is equally
        # unavailable — fail with the remedy, not an AttributeError downstream.
        accessor = getattr(model, "attention_probabilities", None)
        if accessor is None or not getattr(accessor, "enabled", True):
            raise ValueError(
                "attention probabilities are disabled on this model — load it "
                "with enable_attention_probs=True (e.g. LMPipeline(..., "
                "enable_attention_probs=True)), which forces eager attention "
                "and runs nnterp's check_source() validation gate. Note "
                "check_renaming=False also disables the accessor."
            )

    def _check_layer(self, model: StandardizedTransformer) -> None:
        n = _num_layers(model)
        if n is not None and not 0 <= self.layer < n:
            raise IndexError(f"layer {self.layer} out of range for a {n}-layer model")

    def _proxy(self, model: StandardizedTransformer) -> Any:
        """The in-trace read/write handle for the full pattern."""
        self._check_available(model)
        self._check_layer(model)
        return model.attention_probabilities[self.layer]

    # -- in-trace accessors (call inside `with model.trace(...)`) ---------------- #
    def read(
        self, model: StandardizedTransformer, positions: Positions | None = None
    ) -> Any:
        """In-trace read proxy for the pattern, optionally sliced to
        ``positions`` on the **query** axis (``proxy[:, :, positions]``) —
        ``(batch, n_heads, seq | k, key_seq)``. Call inside
        ``with model.trace(...):``."""
        idx = _flat_index(positions, "AttentionProbabilitiesSite.read")
        proxy = self._proxy(model)
        if idx is None:
            return proxy
        return proxy[:, :, idx]

    def write(
        self,
        model: StandardizedTransformer,
        value: Any,
        positions: Positions | None = None,
    ) -> None:
        """In-trace write of ``value`` into the pattern (whole pattern when
        ``positions`` is ``None``, else the selected query rows). Mutates the
        read proxy in place — nnsight tracks the mutation and propagates it
        into the rest of the forward (the ``attn @ V`` mix, hence the logits).
        A tensor ``value`` is first moved to the pattern's device and dtype
        (mirrors ``Site.write``). Call inside ``with model.trace(...):``."""
        idx = _flat_index(positions, "AttentionProbabilitiesSite.write")
        proxy = self._proxy(model)
        if isinstance(value, torch.Tensor):
            value = value.to(device=proxy.device, dtype=proxy.dtype)
        if idx is None:
            proxy[:] = value
        else:
            proxy[:, :, idx] = value

    # -- one-shot read ----------------------------------------------------------- #
    def collect(
        self,
        model: StandardizedTransformer,
        inputs: Any,
        positions: Positions | None = None,
    ) -> torch.Tensor:
        """Read the pattern in a single forward pass and return a concrete
        **CPU** tensor ``(batch, n_heads, seq | len(positions), key_seq)``.
        The forward stops right after the tap (:func:`collect_ordered`)."""
        return collect_ordered(
            model,
            inputs,
            [(forward_key(self, model), lambda m: self.read(m, positions))],
        )[0]


# --------------------------------------------------------------------------- #
#  Shared coercions                                                            #
# --------------------------------------------------------------------------- #
def _featurized(
    site: AttentionProbabilitiesSite | FeaturizedSite, what: str
) -> FeaturizedSite:
    """Accept a bare :class:`AttentionProbabilitiesSite` or a
    :class:`FeaturizedSite` already wrapping one — and refuse anything else:
    the mode ``g``'s mask math assumes the ``(batch, heads, query, key)``
    pattern layout, and the pattern is a probability simplex, not a feature
    space (no featurizer, no ``feature_ids``)."""
    fsite = site if isinstance(site, FeaturizedSite) else FeaturizedSite(site)
    if not isinstance(fsite.site, AttentionProbabilitiesSite):
        raise ValueError(
            f"{what} operates on an AttentionProbabilitiesSite; got a "
            f"FeaturizedSite over {type(fsite.site).__name__}"
        )
    if not fsite.featurizer.is_trivial() or fsite.feature_ids is not None:
        raise ValueError(
            f"{what} edits raw attention probabilities — a featurizer or "
            "feature_ids selection is not supported on this site"
        )
    return fsite


def _selection(sel: AxisSelection, what: str) -> list[int] | None:
    """Normalize an axis selection to a non-empty list of ints (or ``None``
    for the whole axis), rejecting the empty selection at construction."""
    if sel is None:
        return None
    if isinstance(sel, bool):
        # bool subclasses int, so e.g. heads=True would silently coerce to
        # head [1] — a likely flag/selection mix-up; refuse it legibly.
        raise ValueError(
            f"{what} must be an index, a sequence of indices, or None for the "
            f"whole axis — got the bool {sel!r}"
        )
    if isinstance(sel, int):
        return [sel]
    ids = [int(i) for i in sel]
    if not ids:
        raise ValueError(f"{what} must be non-empty (use None for the whole axis)")
    return ids


def _axis_mask(n: Any, sel: list[int] | None, device: Any) -> torch.Tensor:
    """Boolean ``(n,)`` mask with ``sel`` set (negatives resolve as in plain
    tensor indexing); all-True when ``sel`` is ``None``."""
    if sel is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[sel] = True
    return mask


# --------------------------------------------------------------------------- #
#  The two attention-pattern modes                                             #
# --------------------------------------------------------------------------- #
def knockout(
    site: AttentionProbabilitiesSite | FeaturizedSite,
    *,
    heads: AxisSelection = None,
    query_positions: AxisSelection = None,
    key_positions: AxisSelection = None,
    redistribute: bool = True,
    eps: float = 1e-12,
) -> Edit:
    """Zero the selected ``(head, query, key)`` block of the attention
    pattern — an edge (one query attending one key), a key column ("nobody
    attends to BOS"), a head's whole pattern, or (all ``None``) the entire
    pattern.

    Each selection is an index, a sequence of indices, or ``None`` for the
    whole axis; negatives resolve against the run's actual head/sequence
    sizes inside the trace. With ``redistribute=True`` (default) every
    *affected* ``(head, query)`` row is rescaled to sum to 1 — the removed
    mass is redistributed proportionally over the surviving keys (the flag is
    deliberately not named after the standalone :func:`renormalize` mode it
    would shadow); untouched rows are left bit-identical. With
    ``redistribute=False`` the mass is simply removed. A row whose entire
    support was knocked out stays zero either way (``eps`` guards the
    division).
    """
    fsite = _featurized(site, "knockout")
    head_sel = _selection(heads, "heads")
    query_sel = _selection(query_positions, "query_positions")
    key_sel = _selection(key_positions, "key_positions")

    def g(f: torch.Tensor) -> torch.Tensor:
        # f: (batch, n_heads, query_seq, key_seq)
        head_mask = _axis_mask(f.shape[-3], head_sel, f.device)
        query_mask = _axis_mask(f.shape[-2], query_sel, f.device)
        key_mask = _axis_mask(f.shape[-1], key_sel, f.device)
        zero = (
            head_mask[:, None, None]
            & query_mask[None, :, None]
            & key_mask[None, None, :]
        )
        out = f.masked_fill(zero, 0.0)
        if redistribute:
            rows = (head_mask[:, None] & query_mask[None, :])[None, :, :, None]
            out = torch.where(
                rows, out / out.sum(dim=-1, keepdim=True).clamp_min(eps), out
            )
        return out

    return Edit(fsite, g=g)


def renormalize(
    site: AttentionProbabilitiesSite | FeaturizedSite,
    *,
    heads: AxisSelection = None,
    query_positions: AxisSelection = None,
    eps: float = 1e-12,
) -> Edit:
    """Rescale the selected ``(head, query)`` rows of the attention pattern to
    sum to 1 — the standalone half of :func:`knockout`'s redistribution, for
    restoring the simplex after an external pattern write (e.g. a
    ``redistribute=False`` knockout earlier in the same trace). Untouched rows
    are left bit-identical; an all-zero row stays zero (``eps`` guards the
    division)."""
    fsite = _featurized(site, "renormalize")
    head_sel = _selection(heads, "heads")
    query_sel = _selection(query_positions, "query_positions")

    def g(f: torch.Tensor) -> torch.Tensor:
        head_mask = _axis_mask(f.shape[-3], head_sel, f.device)
        query_mask = _axis_mask(f.shape[-2], query_sel, f.device)
        rows = (head_mask[:, None] & query_mask[None, :])[None, :, :, None]
        return torch.where(rows, f / f.sum(dim=-1, keepdim=True).clamp_min(eps), f)

    return Edit(fsite, g=g)
