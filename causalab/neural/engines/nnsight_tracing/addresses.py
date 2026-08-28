"""Interior addresses over nnsight ``.source`` — the upstreamable half.

The module-boundary vocabulary needs no table: envoys mirror the module tree,
so the shared site map addresses them directly (N4). The *interiors* — tensors
``transformers`` computes inside one function call — are reached through
nnsight's ``.source``, which names every call and assignment in a forward.
This module is the table of those names and the matcher that resolves them,
and deliberately nothing else:

* **the tables are pure data** — ``(module, op pattern, peel chain, field,
  fires, requires)`` per component, keyed by the stream vocabulary of
  :mod:`causalab.neural.shared.streams`;
* **the matcher is pure string logic** — substring match, call-op
  disambiguation, exactly-one-hit or a refusal carrying the full op
  inventory;
* **the navigation lives in the executor** — recursive ``.source`` drilling
  only works inside a trace, so the ~10 lines that walk a resolved address
  stay in :mod:`causalab.neural.engines.nnsight_tracing.executor`.

**This module imports nothing from ``causalab``** (enforced by a test): it is
exactly the hybrid/interior accessor layer nnterp's issue #18 asks for, and
keeping it protocol-free is what makes "move it upstream later" a file move
rather than a rewrite. Protocol lowering — SiteSpec resolution, layouts,
write math, refusal policy — stays in ``neural/shared/`` and never comes here.

Why substring patterns, not exact names
---------------------------------------

``.source`` names ops after the variable or symbol plus a positional suffix
(``attn_weights_1``), and the suffix moves when a transformers release adds or
removes a line — that is how transformers 5 broke nnterp's GPT-2 dropout
address. A pattern matches by substring and *refuses* on zero or multiple
hits, so a drifted forward fails loudly with the real inventory (the CI
canary's failure mode) instead of silently reading a neighbouring tensor.
The one systematic ambiguity — a variable that is first assigned and then
called, so both ops carry its name — is resolved structurally: the *call* op
is the hit whose own source line invokes the matched symbol.

📐 The measured facts the tables encode (2026-08-28, transformers 5.16.1,
verified on the real Qwen3.6-35B-A3B and on ``tiny-random/qwen3.5-moe``):
``apply_rotary_pos_emb`` returns ``(q, k)`` post-RoPE; inside the eager
attention function ``attn_weights_1`` is the post-mask softmax input and
``attn_weights_2`` the softmax output (``softmax(attn_weights_1) ==
attn_weights_2`` exactly); the interface call's ``output[0]`` is z, already
transposed back to ``(b, s, H, d)``; and both delta kernels need an
``implementation_0`` peel (the N7 table's first entries).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Iterable, Mapping

__all__ = [
    "ADDRESSES",
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "MOE_EXPERTS",
    "AddressResolutionError",
    "SourceAddress",
    "match_op",
]


class AddressResolutionError(ValueError):
    """A pattern did not resolve to exactly one op.

    A plain ``ValueError`` on purpose: this module knows nothing of the
    protocol's error vocabulary. The executor wraps it with the component,
    layer and library version before it reaches a document author.
    """


@dataclasses.dataclass(frozen=True)
class SourceAddress:
    """One interior tensor, addressed through ``.source``.

    ``module`` is the child path under the layer the ops live on (e.g.
    ``"self_attn"``) — documentation and upstreaming data; the executor
    already holds the resolved envoy and navigates from it.
    """

    module: str
    #: Substring matched against ``source.names`` — NEVER a hardcoded ``_n``
    #: suffix for a symbol that appears once (the suffix is what drifts).
    op_pattern: str
    #: Call ops to drill *through*, one ``.source`` level per element, each
    #: matched by the same substring rule. 📐 ``("implementation_0",)`` is
    #: required on both delta kernels on transformers 5.16.1 (N7).
    peel: tuple[str, ...] = ()
    #: The assignment/op *inside* the drilled source that carries the value,
    #: e.g. ``"attn_weights_1"`` — same substring rule. ``None`` means the
    #: matched op's own output is the value.
    field: str | None = None
    #: ``(positional_index, keyword)`` into the op's ``inputs`` instead of its
    #: output — how a kernel's in-place-updated argument is reached (N7's
    #: ``initial_state``).
    arg: tuple[int, str] | None = None
    #: Which element of a tuple-valued output the component means.
    tuple_index: int | None = None
    #: How often the op fires per forward: ``"once"`` | ``"per_chunk"`` |
    #: ``"per_step"`` | ``"per_expert"``. Everything but ``"once"`` needs
    #: ``tracer.iter`` loop machinery (N7/N6).
    fires: str = "once"
    #: Implementation switches the address is only valid under —
    #: ``{"attn_eager"}``: the fused kernels never materialize the tensor;
    #: ``{"experts_grouped"}``: the grouped experts kernel is where the
    #: per-expert interior's ops live.
    requires: frozenset[str] = frozenset()
    #: The value's rows are expert rows — ``(batch·position·top_k, …)`` — and
    #: the executor re-packs them token-major to the declared 2-D native
    #: shape ``(batch·position, top_k·…)`` (N6). Pure row bookkeeping; the
    #: declared :class:`FeatureShape` stays the semantic description.
    expert_rows: bool = False
    #: Op pattern (same substring rule, matched on the same drilled source as
    #: the value) of the ``torch.sort`` whose ``output[1]`` maps sorted rows →
    #: token-major rows. When set, the value's rows are in the kernel's
    #: expert-sorted order and the executor un-sorts reads / re-sorts writes
    #: through it — the sorted layout is grouped_mm bookkeeping, never the
    #: component's meaning.
    align: str | None = None


_OP_SUFFIX = re.compile(r"_\d+$")


def match_op(
    pattern: str,
    names: Iterable[str],
    line_of: Callable[[str], str] | None = None,
) -> str:
    """The one op ``pattern`` names, or a refusal carrying the inventory.

    Substring match over ``names``. When several ops match — the systematic
    case is a variable assigned and then called, both ops named after it —
    the hits whose own source line *calls* the matched symbol (the op's name
    minus its positional suffix, immediately followed by ``(``) are preferred,
    which ``line_of`` makes possible; anything still ambiguous refuses rather
    than guessing.
    """
    all_names = list(names)
    hits = [n for n in all_names if pattern in n]
    if len(hits) > 1 and line_of is not None:
        calls = [n for n in hits if f"{_OP_SUFFIX.sub('', n)}(" in line_of(n)]
        if calls:
            hits = calls
    if len(hits) == 1:
        return hits[0]
    what = "no op matches" if not hits else f"{len(hits)} ops match ({hits})"
    raise AddressResolutionError(
        f"pattern {pattern!r}: {what}. The installed library's forward names "
        f"these ops: {all_names}. A missing or ambiguous pattern usually means "
        "a transformers release moved this forward's code — re-verify the "
        "address table against the new source."
    )


# --------------------------------------------------------------------------- #
# the tables, keyed by the shared stream vocabulary
# --------------------------------------------------------------------------- #

#: The full-attention mixer's interior (N5). All five live in ``self_attn``'s
#: forward or inside its ``attention_interface(...)`` call. ``attention_z`` is
#: the call's own return (``output[0]``, already ``(b, s, H, d)``) — the
#: drilled ``attn_output_0`` is the pre-transpose ``(b, H, s, d)`` tensor, a
#: different box. Only the softmax's neighbourhood needs eager: q, k and z
#: exist under every implementation.
FULL_ATTENTION: dict[str, SourceAddress] = {
    "attention_query": SourceAddress(
        module="self_attn",
        op_pattern="apply_rotary_pos_emb",
        tuple_index=0,
    ),
    "attention_key": SourceAddress(
        module="self_attn",
        op_pattern="apply_rotary_pos_emb",
        tuple_index=1,
    ),
    "attention_scores": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        # ⚠️ `_1`, the post-mask softmax input — not `_0` (pre-mask). The
        # component is *defined* as the softmax's input (softmax(scores) ==
        # pattern, pinned exact); the pre-mask tensor is a different box.
        field="attn_weights_1",
        requires=frozenset({"attn_eager"}),
    ),
    "attention_probs": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        # the softmax's output, read AND written here: a write is consumed by
        # the value multiply downstream (#53's finding), where a write to the
        # mixer's returned attn_weights would reach nothing.
        field="attn_weights_2",
        requires=frozenset({"attn_eager"}),
    ),
    "attention_z": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        tuple_index=0,
    ),
}

#: The Gated DeltaNet interior — filled by N7.
LINEAR_ATTENTION: dict[str, SourceAddress] = {}

#: The per-expert MoE interior (N6). Not a mixer stream: the ops live under
#: ``mlp.experts``, so the executor keys into this table by component (a
#: ``kind="interior"`` site) rather than by ``stream_at``.
#:
#: All five live inside the grouped experts kernel (``experts_forward`` is the
#: dispatch's call — the same assigned-then-called ambiguity as
#: ``attention_interface``, resolved the same way). 📐 Measured on
#: ``tiny-random/qwen3.5-moe`` and matching the real A3B's inventory: the
#: kernel sorts the ``(token, slot)`` rows by expert (``torch_sort``), runs
#: the fused gate_up projection (``_apply_gate``'s ``chunk`` splits it),
#: down-projects, un-sorts (``inv_perm``) and weights (``weighted_out_1``,
#: token-major again). The sorted layout is bookkeeping, so every sorted-space
#: value carries ``align`` and is presented token-major.
MOE_EXPERTS: dict[str, SourceAddress] = {
    "expert_gate_proj": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        peel=("self__apply_gate",),
        field="gate_up_out_chunk",
        tuple_index=0,
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_up_proj": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        peel=("self__apply_gate",),
        field="gate_up_out_chunk",
        tuple_index=1,
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_activation": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # the _apply_gate call's own return: act(gate)·up, the down-projection's
        # input — the same tensor `shared_expert_activation` names on the
        # shared expert
        field="self__apply_gate",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_permutation": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # the kernel's own inverse permutation — token-major by construction,
        # so no align. The `_1` suffix is load-bearing: `inv_perm_0` is the
        # empty_like allocation, `_1` the filled table, and neither line is a
        # call, so the call-op rule cannot separate them.
        field="inv_perm_1",
        expert_rows=True,
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_output": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # after the kernel's own un-sort and the router weighting: token-major
        # weighted contributions, summing to `routed_output` over the top-k
        field="weighted_out_1",
        expert_rows=True,
        requires=frozenset({"experts_grouped"}),
    ),
}

#: Every table, keyed the way :func:`causalab.neural.shared.streams.stream_at`
#: answers — the executor's single lookup point.
ADDRESSES: Mapping[str, Mapping[str, SourceAddress]] = {
    "full_attention": FULL_ATTENTION,
    "linear_attention": LINEAR_ATTENTION,
}
