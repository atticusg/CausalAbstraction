"""Key/value-side attention detail: capture and score reconstruction.

GATED: everything here requires ``attention_style == "fused-qkv-absolute"``
(GPT-2-style fused-QKV attention with absolute position embeddings) and
raises :class:`NotImplementedError` otherwise. The main path-patching engine
is unaffected by this gate — it never recomputes attention.

Capture is pyvene, like everything in this package: per-head q/k/v come from
pyvene's ``query_output`` / ``key_output`` / ``value_output`` components
(pyvene splits GPT-2's fused ``c_attn`` itself), and construction runs the
same capability check as the engine — a family whose mapping lacks these
units raises :class:`UnsupportedArchitectureError`. Pre-softmax scores are
**derived arithmetic** on the captured q/k; the score tensor is not a module
boundary anywhere, and no intervention on it is ever needed: analytic
K/V-side patching recomputes scores from cached q/k.

K/V capture requires ``attn_implementation="eager"`` at pipeline load
(causalab's ``LMPipeline`` default): fused sdpa/flash kernels do not
materialize what pyvene's attention units need, and eager is cheap at
analysis scale. Construction verifies this.

Design notes for generalization beyond GPT-2 (agreed direction; deliberately
NOT implemented here):

* **Rotary models**: a cached key vector is position-entangled — RoPE has
  already rotated it by its source position's angle. Patching a key along a
  path therefore requires rotating the key *delta* by the patched position's
  angle before re-scoring; raw cache-to-cache key substitution is only valid
  when clean and counterfactual token positions coincide. Pre-rotation keys
  collected via pyvene (``key_output`` is the projection output, before RoPE)
  with the rotation applied in engine arithmetic is pyvene-native — no hooks
  needed.
* **GQA models**: the honest unit for key/value path edges is the *KV head*,
  with effects fanning out to every query head in its group — that is what
  is causally separable in the weights. Per-query-head value semantics would
  require a TransformerLens-style expansion of shared KV heads, which this
  analytic engine could later offer as pure cache arithmetic (duplicate the
  cached k/v per group member, re-average one query head's z); KV-head edges
  should remain the default.
* **Reference twin**: the K/V twin should stay pyvene — replace per-head k/v
  at the patched position via replace interventions (``head_key_output`` /
  ``head_value_output``) and let the model's own attention recompute.
  Analytic scores are reconstructable from cached q/k, so no score-tensor
  intervention is ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    prepare_intervenable_model,
)
from causalab.neural.units import AtomicModelUnit, ComponentIndexer

from .cache import padded_position
from .descriptor import ArchitectureDescriptor

__all__ = ["AttnDetailCache", "build_attn_detail_cache"]


def _require_fused_absolute(desc: ArchitectureDescriptor, what: str) -> None:
    if desc.attention_style != "fused-qkv-absolute":
        raise NotImplementedError(
            f"{what} is only implemented for GPT-2-style attention "
            f"(fused QKV, absolute positions); this model declares "
            f"attention_style={desc.attention_style!r}. Rotary models need "
            f"the key delta rotated by the patched position's angle before "
            f"re-scoring, and GQA models need KV-head (not query-head) edge "
            f"units — see this module's docstring for the agreed design."
        )


@dataclass
class AttnDetailCache:
    """Per-head q/k/v and derived attention scores at one patched position.

    q_pos      (N, L, H, dh)    query vectors at the position
    k_all      (N, L, H, T, dh) key vectors, all (padded) positions
    v_all      (N, L, H, T, dh) value vectors
    scores_pos (N, L, H, T)     pre-softmax masked scores at the position
                                (derived from q_pos/k_all; not captured)
    attention_mask (N, T)
    """

    q_pos: Tensor
    k_all: Tensor
    v_all: Tensor
    scores_pos: Tensor
    attention_mask: Tensor


def _null_indexer() -> ComponentIndexer:
    return ComponentIndexer(lambda _x: [], id="path_patching_explicit")


@torch.no_grad()
def build_attn_detail_cache(
    pipeline: Any,
    desc: ArchitectureDescriptor,
    inputs: Sequence[Any],
    *,
    position_index: int | Sequence[int] = -1,
) -> AttnDetailCache:
    """Capture per-head q/k/v via pyvene and derive masked scores.

    GPT-2-style attention only (see the gate). Runs as a single batch: this
    is validation-scale machinery.
    """
    _require_fused_absolute(desc, "attention-detail capture")
    from .provenance import check_capability

    check_capability(desc.model, ["K/V attention-detail collection"])
    attn_impl = getattr(desc.model.config, "_attn_implementation", "eager")
    if attn_impl != "eager":
        raise RuntimeError(
            f"K/V attention-detail capture requires attn_implementation="
            f"'eager' (got {attn_impl!r}): fused sdpa/flash kernels do not "
            f"materialize what pyvene's attention units need. causalab's "
            f"LMPipeline loads models eager by default."
        )
    inputs = [{"raw_input": x} if isinstance(x, str) else x for x in inputs]
    L, H, DH = desc.n_layers, desc.n_heads, desc.head_dim

    units: list[AtomicModelUnit] = []
    meta: list[tuple[str, int]] = []
    for layer in range(L):
        for quantity, comp in (
            ("q", "query_output"),
            ("k", "key_output"),
            ("v", "value_output"),
        ):
            units.append(
                AtomicModelUnit(
                    layer, comp, _null_indexer(), id=f"pp_kv_{quantity}_l{layer}"
                )
            )
            meta.append((quantity, layer))

    intervenable_model = prepare_intervenable_model(
        pipeline, units, intervention_type="collect"
    )
    try:
        loaded = pipeline.load(list(inputs))
        mask = loaded["attention_mask"]
        bsz, T = mask.shape
        pos = padded_position(mask, position_index)
        all_positions = [list(range(T)) for _ in range(bsz)]
        indices = [all_positions for _ in units]
        location_map = {"sources->base": (indices, indices)}
        result = intervenable_model(loaded, unit_locations=location_map)
        collected = result[0][1]
    finally:
        delete_intervenable_model(intervenable_model)

    def per_head(x: Tensor) -> Tensor:  # (B, T, H*dh) -> (B, H, T, dh)
        return x.reshape(bsz, T, H, DH).permute(0, 2, 1, 3)

    store: dict[tuple[str, int], Tensor] = {}
    for m, act in zip(meta, collected):
        a = act if isinstance(act, Tensor) else torch.cat(list(act))
        store[m] = per_head(a.reshape(bsz, T, -1).float().cpu())

    k_all = torch.stack([store[("k", li)] for li in range(L)], dim=1)
    v_all = torch.stack([store[("v", li)] for li in range(L)], dim=1)
    bidx = torch.arange(bsz)
    pidx = torch.tensor(pos)
    q_pos = torch.stack([store[("q", li)][bidx, :, pidx] for li in range(L)], dim=1)

    # derived arithmetic — gap registry: kv:scores (no-module-boundary);
    # the score tensor is never hooked or intervened on.
    scores = (q_pos.unsqueeze(3) @ k_all.transpose(-1, -2)).squeeze(3) / (DH**0.5)
    mask_c = mask.cpu()
    scores = scores.masked_fill(mask_c[:, None, None, :] == 0, float("-inf"))
    causal = torch.arange(T)[None, :] > pidx[:, None]
    scores = scores.masked_fill(causal[:, None, None, :], float("-inf"))

    return AttnDetailCache(
        q_pos=q_pos,
        k_all=k_all,
        v_all=v_all,
        scores_pos=scores,
        attention_mask=mask_c,
    )
