"""Key/value-side path patching: capture, score reconstruction, KV-head edges.

Extends the analytic engine to edges that end at an attention head's **keys
or values at one patched token position** (Hanna et al. 2023 App. 8: what
feeds the circuit heads' k/v at the YY position), general across the same
four families as the main engine, with three architecture facts handled
explicitly:

* **Rotary embeddings** (Llama, Gemma-2): pyvene's ``key_output`` unit is the
  k-projection's *output*, before rotation, so cached keys are
  position-disentangled. Score reconstruction applies the model's **own
  rotary module** (``desc.rotary_emb()``) to both q and k at their actual
  position ids; a key patched at position ``p`` is rotated by ``p``'s angle.
  Rotation is linear, so rotating the patched key equals rotating the key
  delta — the docstring recipe "rotate the delta" and "rotate the new key"
  are the same operation.
* **Grouped-query attention** (Llama-3.1): the causally separable unit in
  the weights is the **KV head**; edges attach to :class:`KVHead` and the
  engine fans the resulting z-deltas across every query head in the group
  (:meth:`ArchitectureDescriptor.query_heads_of_kv`). Standard multi-head
  models are the special case of group size 1. Per-*query*-head value paths
  do not exist in the weights; a TransformerLens-style expansion (duplicate
  cached k/v per group member, re-average one query head's z) would be pure
  cache arithmetic and is deliberately not offered as a default.
* **Sliding-window attention** (Gemma-2): before patching position ``p`` as
  seen from the end position at layer ``L``, the engine verifies the end
  position can attend to ``p`` within that layer's window (read off the
  attention module, the way the model's own mask construction does) and
  raises :class:`SlidingWindowError` naming the layer and window otherwise.

Support policy (same as the engine): an operation is available iff pyvene's
mapping has the units it needs — ``query_output``/``key_output``/
``value_output`` for collection, ``head_key_output``/``head_value_output``
for the reference twin. A family whose mapping lacks them (gpt_neox: the
fused per-head-interleaved QKV projection has no pyvene q/k/v units) raises
:class:`UnsupportedArchitectureError` at construction. No raw torch hooks.

K/V capture and patching require ``attn_implementation="eager"`` at pipeline
load: fused sdpa/flash kernels do not materialize what pyvene's attention
units hook, and the analytic score reconstruction mirrors the eager
reference path. Construction verifies this and names the loader option
(``LMPipeline(..., attn_implementation="eager")`` /
``attn_implementation: eager`` in the model YAML).

Numerical contract: everything is float32 cache arithmetic through the
model's own modules (pre-norms, q/k/v projections, rotary, post-norms, LM
head run in model dtype at module boundaries, matching the engine). Score
reconstruction mirrors the HF eager path exactly: scores * module scaling,
Gemma-2 attention softcapping **before** masking, softmax in float32.
Construction guard K1 verifies the whole chain empirically: reconstructed
per-head z at the end position must match the cached
``attention_value_output`` on both caches (tight in float32; the bf16 floor
is recorded, as with G4). Patch deltas are formed as
``proj(norm(x+Δ)) − proj(norm(x))`` through the same code path, so a
zero-delta patch is exactly zero in any dtype.

The score tensor itself is never a module boundary and is never intervened
on: analytic K/V patching recomputes scores from cached q/k (gap registry:
kv:scores, no-module-boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    prepare_intervenable_model,
)
from causalab.neural.units import AtomicModelUnit, ComponentIndexer

from .cache import PatchCache, keep_last_dim_on_collects, padded_position
from .descriptor import ArchitectureDescriptor
from .engine import PatchEngine

__all__ = [
    "AttnDetailCache",
    "KVEdge",
    "KVHead",
    "KVPatchEngine",
    "SlidingWindowError",
    "build_attn_detail_cache",
]


class SlidingWindowError(ValueError):
    """The patched position is outside the layer's attention window."""


@dataclass(frozen=True)
class KVHead:
    """A key/value head: the causally separable K/V unit in the weights.

    On grouped-query models this is one of ``n_kv_heads`` heads whose keys
    and values are read by ``kv_group_size`` query heads; on standard
    multi-head attention it coincides with the query head of the same index.
    """

    layer: int
    kv_index: int

    @classmethod
    def for_query_head(
        cls, desc: ArchitectureDescriptor, layer: int, head: int
    ) -> "KVHead":
        """The KV head whose keys/values query head ``head`` reads."""
        return cls(layer, head // desc.kv_group_size)


@dataclass(frozen=True)
class KVEdge:
    """One K/V-side edge: a KV head with which of its inputs are patched."""

    kv: KVHead
    patch_k: bool = False
    patch_v: bool = True

    def __post_init__(self) -> None:
        if not (self.patch_k or self.patch_v):
            raise ValueError("KVEdge patches neither keys nor values")


@dataclass
class AttnDetailCache:
    """Pre-rotation per-head q/k/v for one loaded batch (float32, CPU).

    q          (N, L, H, dh)      query vectors at each named position
                                  (dict: position name -> tensor)
    k_all      (N, L, KV, T, dh)  key vectors, all padded positions,
                                  pre-rotation
    v_all      (N, L, KV, T, dh)  value vectors, all padded positions
    attention_mask (N, T)
    position_ids   (N, T)         the position ids the forward actually used
    positions      name -> list of padded batch indices (one per example)
    """

    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    q: dict[str, Tensor]
    k_all: Tensor
    v_all: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    positions: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return self.k_all.shape[0]

    @property
    def seq_len(self) -> int:
        return self.k_all.shape[3]


def _null_indexer() -> ComponentIndexer:
    return ComponentIndexer(lambda _x: [], id="path_patching_explicit")


@torch.no_grad()
def build_attn_detail_cache(
    pipeline: Any,
    desc: ArchitectureDescriptor,
    inputs: Sequence[Any],
    positions: dict[str, int | Sequence[int]],
    *,
    require_eager: bool = True,
) -> AttnDetailCache:
    """Capture pre-rotation per-head q/k/v via pyvene.

    ``positions`` maps names (e.g. ``"end"``, ``"yy"``) to unpadded
    per-example coordinates, as in :func:`build_patch_cache`. Query vectors
    are stored at each named position; keys and values at every position.
    Runs as a single batch: this is validation-scale machinery, and a single
    load fixes one padding for the whole set.
    """
    from .provenance import check_capability

    check_capability(desc.model, ["K/V attention-detail collection"])
    if require_eager:
        _require_eager(desc)
    if not positions:
        raise ValueError("at least one named position is required")
    inputs = [{"raw_input": x} if isinstance(x, str) else x for x in inputs]
    L, H, KV, DH = desc.n_layers, desc.n_heads, desc.n_kv_heads, desc.head_dim

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
    keep_last_dim_on_collects(intervenable_model)
    try:
        loaded = pipeline.load(list(inputs))
        mask = loaded["attention_mask"]
        bsz, T = mask.shape
        pos_named = {
            name: padded_position(mask, spec) for name, spec in positions.items()
        }
        if "position_ids" in loaded:
            position_ids = loaded["position_ids"]
        else:  # the model's default when none are passed: arange over padded T
            position_ids = (
                torch.arange(T, device=mask.device).unsqueeze(0).expand(bsz, T)
            )
        all_positions = [list(range(T)) for _ in range(bsz)]
        indices = [all_positions for _ in units]
        location_map = {"sources->base": (indices, indices)}
        result = intervenable_model(loaded, unit_locations=location_map)
        collected = result[0][1]
    finally:
        delete_intervenable_model(intervenable_model)

    store: dict[tuple[str, int], Tensor] = {}
    for m, act in zip(meta, collected):
        a = act if isinstance(act, Tensor) else torch.cat(list(act))
        n_heads_here = H if m[0] == "q" else KV
        store[m] = a.reshape(bsz, T, n_heads_here, DH).permute(0, 2, 1, 3).float().cpu()

    k_all = torch.stack([store[("k", li)] for li in range(L)], dim=1)
    v_all = torch.stack([store[("v", li)] for li in range(L)], dim=1)
    bidx = torch.arange(bsz)
    q = {
        name: torch.stack(
            [store[("q", li)][bidx, :, torch.tensor(p)] for li in range(L)], dim=1
        )
        for name, p in pos_named.items()
    }
    return AttnDetailCache(
        n_layers=L,
        n_heads=H,
        n_kv_heads=KV,
        head_dim=DH,
        q=q,
        k_all=k_all,
        v_all=v_all,
        attention_mask=mask.cpu(),
        position_ids=position_ids.cpu(),
        positions=pos_named,
    )


def _require_eager(desc: ArchitectureDescriptor) -> None:
    attn_impl = getattr(desc.model.config, "_attn_implementation", "eager")
    if attn_impl != "eager":
        raise RuntimeError(
            f"K/V-side path patching requires attn_implementation='eager' "
            f"(got {attn_impl!r}): fused sdpa/flash kernels do not "
            f"materialize what pyvene's attention units hook, and the "
            f"analytic score reconstruction mirrors the eager path. Load "
            f"with LMPipeline(..., attn_implementation='eager') or set "
            f"attn_implementation: eager in the model YAML; no re-loading "
            f"or monkey-patching is done on your behalf."
        )


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class KVPatchEngine:
    """Analytic K/V-side patching over detail caches, twin-checked.

    Built from two position views of the same cache pair (``engine_end``
    reads the receiver position, ``engine_patch`` the patched K/V position)
    plus the pre-rotation q/k/v detail caches for both sides.

    Construction refuses non-eager models, refuses families whose pyvene
    mapping lacks the q/k/v units, and runs guard K1 (z reconstruction) on
    both caches.
    """

    def __init__(
        self,
        engine_end: PatchEngine,
        engine_patch: PatchEngine,
        detail_clean: AttnDetailCache,
        detail_cf: AttnDetailCache,
        *,
        position_end: str = "end",
        position_patch: str = "yy",
        run_guards: bool = True,
        guard_tolerances: dict[str, float] | None = None,
    ) -> None:
        if engine_end.desc is not engine_patch.desc:
            raise ValueError("engines must share one resolved descriptor")
        self.desc = engine_end.desc
        self.engine_end = engine_end
        self.engine_patch = engine_patch
        self.detail = {"clean": detail_clean, "cf": detail_cf}
        self.position_end = position_end
        self.position_patch = position_patch
        self.device = engine_end.device
        self.model_dtype = engine_end.model_dtype
        from .provenance import check_capability

        self.capability = check_capability(
            self.desc.model,
            ["K/V attention-detail collection", "K/V reference-twin interventions"],
        )
        _require_eager(self.desc)
        for name, det in self.detail.items():
            for pos in (position_end, position_patch):
                if pos not in det.positions:
                    raise ValueError(
                        f"detail cache ({name}) has no position {pos!r}; build "
                        f"it with positions={{{position_end!r}: ..., "
                        f"{position_patch!r}: ...}}"
                    )
        self._rope: dict[str, tuple[Tensor, Tensor] | None] = {}
        self.guard_report: dict[str, Any] | None = None
        if run_guards:
            self.guard_report = self._run_guards(guard_tolerances)

    # ------------------------------------------------------------------
    # caches and coordinates
    # ------------------------------------------------------------------
    def _cache(self, name: str) -> PatchCache:
        return self.engine_end.clean if name == "clean" else self.engine_end.cf

    def _pos_idx(self, detail: AttnDetailCache, name: str) -> Tensor:
        return torch.tensor(detail.positions[name], dtype=torch.long)

    def _cos_sin(self, name: str) -> tuple[Tensor, Tensor] | None:
        """(cos, sin) over all padded positions, via the model's own rotary
        module on the position ids the forward actually used."""
        if name in self._rope:
            return self._rope[name]
        rotary = self.desc.rotary_emb()
        if rotary is None or self.desc.attention_style == "fused-qkv-absolute":
            self._rope[name] = None
            return None
        det = self.detail[name]
        pos_ids = det.position_ids.to(self.device)
        dummy = torch.zeros(1, dtype=torch.float32, device=self.device)
        cos, sin = rotary(dummy, pos_ids)
        self._rope[name] = (cos.float().cpu(), sin.float().cpu())
        return self._rope[name]

    def _rot(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary rotation. x: (..., T, dh) or (..., dh) with matching
        cos/sin broadcastable to it."""
        return x * cos + _rotate_half(x) * sin

    # ------------------------------------------------------------------
    # score/z reconstruction (the HF eager path, in float32)
    # ------------------------------------------------------------------
    def _z_from_qkv(
        self,
        layer: int,
        q_pos: Tensor,  # (N, H, dh) pre-rotation, at the end position
        k_all: Tensor,  # (N, KV, T, dh) pre-rotation
        v_all: Tensor,  # (N, KV, T, dh)
        detail_name: str,
        *,
        heads: Sequence[int] | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Per-query-head z at the end position, reconstructed the way the
        eager kernel computes it. ``heads`` restricts to a subset of query
        heads (default: all)."""
        desc = self.desc
        det = self.detail[detail_name]
        N, T = det.n_examples, det.seq_len
        heads = list(range(desc.n_heads)) if heads is None else list(heads)
        end_idx = self._pos_idx(det, self.position_end)
        bidx = torch.arange(N)

        rope = self._cos_sin(detail_name)
        if rope is not None:
            cos, sin = rope  # (N, T, dh)
            q_r = self._rot(
                q_pos, cos[bidx, end_idx][:, None, :], sin[bidx, end_idx][:, None, :]
            )
            k_r = self._rot(k_all, cos[:, None, :, :], sin[:, None, :, :])
        else:
            q_r, k_r = q_pos, k_all

        g = desc.kv_group_size
        kv_of = [h // g for h in heads]
        qh = q_r[:, heads]  # (N, |heads|, dh)
        kh = k_r[:, kv_of]  # (N, |heads|, T, dh)
        vh = v_all[:, kv_of]  # (N, |heads|, T, dh)

        scores = torch.einsum("nhd,nhtd->nht", qh, kh) * desc.attn_scaling(layer)
        cap = desc.attn_logit_softcapping()
        if cap is not None:  # Gemma-2: softcap BEFORE masking, as in HF eager
            scores = torch.tanh(scores / cap) * cap

        mask = det.attention_mask  # (N, T)
        neg = torch.finfo(torch.float32).min
        scores = scores.masked_fill(mask[:, None, :] == 0, neg)
        causal = torch.arange(T)[None, :] > end_idx[:, None]  # (N, T)
        scores = scores.masked_fill(causal[:, None, :], neg)
        window = desc.attn_sliding_window(layer)
        if window is not None:
            too_far = (end_idx[:, None] - torch.arange(T)[None, :]) >= window
            scores = scores.masked_fill(too_far[:, None, :], neg)

        weights = torch.softmax(scores.float(), dim=-1)
        z = torch.einsum("nht,nhtd->nhd", weights, vh)
        if return_weights:
            return z, weights
        return z

    def attention_weights(
        self, layer: int, cache: str = "clean", heads: Sequence[int] | None = None
    ) -> Tensor:
        """(N, |heads|, T) reconstructed attention from the end position."""
        det = self.detail[cache]
        _, w = self._z_from_qkv(
            layer,
            det.q[self.position_end][:, layer],
            det.k_all[:, layer],
            det.v_all[:, layer],
            cache,
            heads=heads,
            return_weights=True,
        )
        return w

    def attention_to_patch_position(
        self, layer: int, head: int, cache: str = "clean"
    ) -> Tensor:
        """(N,) attention weight from the end position to the patched
        position (e.g. Hanna et al. Fig 11: a7.h10 end -> YY)."""
        det = self.detail[cache]
        w = self.attention_weights(layer, cache, heads=[head])[:, 0]
        p_idx = self._pos_idx(det, self.position_patch)
        return w[torch.arange(det.n_examples), p_idx]

    # ------------------------------------------------------------------
    # reachability
    # ------------------------------------------------------------------
    def check_reachable(self, layer: int, detail_name: str = "clean") -> None:
        """Refuse a K/V edge whose patched position the end position cannot
        attend to within this layer's sliding window."""
        window = self.desc.attn_sliding_window(layer)
        if window is None:
            return
        det = self.detail[detail_name]
        end_idx = self._pos_idx(det, self.position_end)
        p_idx = self._pos_idx(det, self.position_patch)
        dist = (end_idx - p_idx).max().item()
        if dist >= window:
            raise SlidingWindowError(
                f"K/V edge at layer {layer}: the end position cannot attend "
                f"to patched position {self.position_patch!r} there — layer "
                f"{layer} uses sliding-window attention with window "
                f"{window}, and the largest end-to-patch distance in this "
                f"batch is {dist} (must be < window). Pick a full-attention "
                f"layer or a closer position."
            )

    # ------------------------------------------------------------------
    # patched k/v construction
    # ------------------------------------------------------------------
    def _kv_delta_from_resid(
        self, layer: int, delta_resid: Tensor, base: str
    ) -> tuple[Tensor, Tensor]:
        """(Δk, Δv) at the patched position for every KV head of ``layer``,
        pre-rotation, from a residual delta at that position.

        Formed as ``proj(norm(x+Δ)) − proj(norm(x))`` through the model's own
        pre-norm and projections (direct module calls, model dtype), so a
        zero residual delta gives exactly zero.
        """
        cache = self._cache(base)
        eng = self.engine_patch
        if layer == 0:
            x = eng._at(cache, "embed")
        else:
            x = eng._at(cache, "block_out", layer - 1)
        norm = self.desc.attn_pre_norm(layer)
        d = delta_resid.to(self.device).float()
        normed_new = norm((x + d).to(self.model_dtype))
        normed_old = norm(x.to(self.model_dtype))
        _, k_new, v_new = self.desc.qkv_new(layer, normed_new)
        _, k_old, v_old = self.desc.qkv_new(layer, normed_old)
        KV, DH = self.desc.n_kv_heads, self.desc.head_dim
        dk = (k_new - k_old).float().reshape(-1, KV, DH).cpu()
        dv = (v_new - v_old).float().reshape(-1, KV, DH).cpu()
        return dk, dv

    def _substitution_delta(
        self, edge: KVEdge, base: str
    ) -> tuple[Tensor | None, Tensor | None]:
        """(Δk, Δv) at the patched position from cache-to-cache substitution:
        the other side's captured pre-rotation k/v minus the base side's.
        Valid when clean and counterfactual token positions coincide (token-
        aligned pairs), which is exactly the regime raw substitution is
        correct in."""
        other = "cf" if base == "clean" else "clean"
        det_b, det_o = self.detail[base], self.detail[other]
        L, j = edge.kv.layer, edge.kv.kv_index
        pb = self._pos_idx(det_b, self.position_patch)
        po = self._pos_idx(det_o, self.position_patch)
        bidx = torch.arange(det_b.n_examples)
        dk = dv = None
        if edge.patch_k:
            dk = det_o.k_all[bidx, L, j, po] - det_b.k_all[bidx, L, j, pb]
        if edge.patch_v:
            dv = det_o.v_all[bidx, L, j, po] - det_b.v_all[bidx, L, j, pb]
        return dk, dv

    # ------------------------------------------------------------------
    # the K/V trunk delta
    # ------------------------------------------------------------------
    @torch.no_grad()
    def kv_trunk_delta(
        self,
        edges: Sequence[KVEdge],
        input_deltas: Mapping[int, Tensor] | Tensor | None = None,
        *,
        base_cache: str = "clean",
        patch_q: bool = False,
        rotate_key_delta: bool = True,
    ) -> Tensor:
        """(N, d) change in the edges' layers' attention-branch trunk
        contributions at the end position when the named KV heads' keys
        and/or values at the patched position are patched.

        ``input_deltas``: residual delta(s) at the patched position entering
        each edge's layer — one tensor for all layers or a per-layer map
        (path-patch mode). ``None`` selects cache-to-cache substitution: the
        other side's captured k/v at the position replace the base side's.

        ``patch_q=True`` additionally restores the affected query heads'
        query vectors from the other cache (the same side a substitution
        patch draws k/v from) — the "queries receive all good inputs" case
        of a full-circuit evaluation. The query swap is then part of the
        patch: the base-side z is scored with the base-side q, the patched
        z with the other side's q. Substitution semantics (token-aligned
        pairs), like ``input_deltas=None``.

        ``rotate_key_delta=False`` is the **negative control**: it adds the
        key delta to the already-rotated base keys without rotating it by
        the patched position's angle. On a rotary family this is wrong by
        construction and must disagree with the reference twin; it exists so
        validation can prove the rotation term has teeth.

        Per-layer raw deltas are summed before any branch post-norm
        (Gemma-2), matching the main engine's trunk_delta.
        """
        desc = self.desc
        det_b = self.detail[base_cache]
        other = "cf" if base_cache == "clean" else "clean"
        N = det_b.n_examples
        bidx = torch.arange(N)
        p_idx = self._pos_idx(det_b, self.position_patch)

        # accumulate per-layer z-deltas (N, H, dh) over affected query heads
        per_layer_dz: dict[int, Tensor] = {}
        # group edges per layer so one reconstruction handles several KV heads
        by_layer: dict[int, list[KVEdge]] = {}
        for e in edges:
            by_layer.setdefault(e.kv.layer, []).append(e)

        for layer, layer_edges in sorted(by_layer.items()):
            self.check_reachable(layer, base_cache)
            k_base = det_b.k_all[:, layer].clone()  # (N, KV, T, dh) pre-rot
            v_base = det_b.v_all[:, layer].clone()
            deltas_ready: dict[int, tuple[Tensor | None, Tensor | None]] = {}
            if input_deltas is None:
                for e in layer_edges:
                    deltas_ready[e.kv.kv_index] = self._substitution_delta(
                        e, base_cache
                    )
            else:
                dr = (
                    input_deltas[layer]
                    if isinstance(input_deltas, Mapping)
                    else input_deltas
                )
                dk_all, dv_all = self._kv_delta_from_resid(layer, dr, base_cache)
                for e in layer_edges:
                    j = e.kv.kv_index
                    deltas_ready[j] = (
                        dk_all[:, j] if e.patch_k else None,
                        dv_all[:, j] if e.patch_v else None,
                    )

            k_patched = k_base.clone()
            v_patched = v_base.clone()
            unrotated_extra: dict[int, Tensor] = {}
            affected_q_heads: set[int] = set()
            for e in layer_edges:
                j = e.kv.kv_index
                dk, dv = deltas_ready[j]
                if dk is not None:
                    if rotate_key_delta:
                        k_patched[bidx, j, p_idx] = k_patched[bidx, j, p_idx] + dk
                    else:
                        # negative control: remember the raw delta; it will be
                        # added AFTER rotation below (wrong on rotary models)
                        unrotated_extra[j] = dk
                if dv is not None:
                    v_patched[bidx, j, p_idx] = v_patched[bidx, j, p_idx] + dv
                affected_q_heads.update(desc.query_heads_of_kv(j))

            heads = sorted(affected_q_heads)
            q_base = det_b.q[self.position_end][:, layer]
            q_new = (
                self.detail[other].q[self.position_end][:, layer] if patch_q else q_base
            )
            if unrotated_extra:
                z_new = self._z_with_unrotated_extra(
                    layer,
                    q_new,
                    k_patched,
                    v_patched,
                    base_cache,
                    unrotated_extra,
                    heads,
                )
            else:
                z_new = self._z_from_qkv(
                    layer, q_new, k_patched, v_patched, base_cache, heads=heads
                )
            # base-side z for the same heads, reconstructed the same way so
            # the delta isolates the patch (reconstruction bias cancels)
            z_old = self._z_from_qkv(
                layer, q_base, k_base, v_base, base_cache, heads=heads
            )
            per_layer_dz[layer] = (z_new - z_old, heads)

        # z-deltas -> trunk deltas through W_O (+ branch post-norm)
        eng = self.engine_end
        cache_b = self._cache(base_cache)
        total: Tensor | None = None
        for layer, (dz, heads) in per_layer_dz.items():
            w_o = eng._w_o[layer]
            raw = torch.zeros(N, desc.d_model, device=self.device)
            for i, h in enumerate(heads):
                rows = w_o[h * desc.head_dim : (h + 1) * desc.head_dim]
                raw = raw + dz[:, i].to(self.device) @ rows
            post = desc.attn_post_norm(layer)
            if post is None:
                d = raw
            else:
                cache_key = "clean" if cache_b is eng.clean else "cf"
                pre = eng._attn_pre_norm_clean(cache_key, cache_b, layer)
                d = eng._apply_norm(post, pre + raw) - eng._apply_norm(post, pre)
            total = d if total is None else total + d
        if total is None:
            total = torch.zeros(N, desc.d_model, device=self.device)
        return total

    def _z_with_unrotated_extra(
        self,
        layer: int,
        q_pos: Tensor,
        k_patched: Tensor,
        v_patched: Tensor,
        detail_name: str,
        unrotated_extra: dict[int, Tensor],
        heads: Sequence[int],
    ) -> Tensor:
        """Negative-control scoring: key deltas added after rotation.

        Only exists to prove the rotation term matters; identical to the
        correct path on non-rotary families (where rotation is identity).
        """
        desc = self.desc
        det = self.detail[detail_name]
        N, T = det.n_examples, det.seq_len
        end_idx = self._pos_idx(det, self.position_end)
        p_idx = self._pos_idx(det, self.position_patch)
        bidx = torch.arange(N)
        rope = self._cos_sin(detail_name)
        if rope is not None:
            cos, sin = rope
            q_r = self._rot(
                q_pos, cos[bidx, end_idx][:, None, :], sin[bidx, end_idx][:, None, :]
            )
            k_r = self._rot(k_patched, cos[:, None, :, :], sin[:, None, :, :])
        else:
            q_r, k_r = q_pos, k_patched
        for j, dk in unrotated_extra.items():
            k_r[bidx, j, p_idx] = k_r[bidx, j, p_idx] + dk  # the deliberate bug
        g = desc.kv_group_size
        kv_of = [h // g for h in heads]
        qh, kh, vh = q_r[:, list(heads)], k_r[:, kv_of], v_patched[:, kv_of]
        scores = torch.einsum("nhd,nhtd->nht", qh, kh) * desc.attn_scaling(layer)
        cap = desc.attn_logit_softcapping()
        if cap is not None:
            scores = torch.tanh(scores / cap) * cap
        neg = torch.finfo(torch.float32).min
        scores = scores.masked_fill(det.attention_mask[:, None, :] == 0, neg)
        causal = torch.arange(T)[None, :] > end_idx[:, None]
        scores = scores.masked_fill(causal[:, None, :], neg)
        window = desc.attn_sliding_window(layer)
        if window is not None:
            too_far = (end_idx[:, None] - torch.arange(T)[None, :]) >= window
            scores = scores.masked_fill(too_far[:, None, :], neg)
        weights = torch.softmax(scores.float(), dim=-1)
        return torch.einsum("nht,nhtd->nhd", weights, vh)

    # ------------------------------------------------------------------
    # guards
    # ------------------------------------------------------------------
    def _run_guards(self, tolerances: dict[str, float] | None) -> dict[str, Any]:
        """K1: per-head z at the end position, reconstructed from captured
        pre-rotation q/k/v through the analytic score path, must match the
        cached attention_value_output. This exercises rotation, GQA fan-in,
        scaling, softcapping, masks, and softmax in one check."""
        # bf16 K1 is a sanity bound, not a precision target (same stance as
        # the engine's G4): reconstructing in float32 what the model
        # accumulated in bf16 has a model-size-dependent rounding floor —
        # measured 1.6e-2 on Gemma-2-2B and 4.3e-2 on Llama-3.1-8B, while
        # the same models in float32 close at ~1e-6 (proving the
        # conventions) and genuine wiring errors show up at O(1).
        tol = {
            "z_reconstruction_rel": 1e-4
            if self.model_dtype in (torch.float32, torch.float64)
            else 1e-1,
        }
        if tolerances:
            tol.update(tolerances)
        report: dict[str, Any] = {"tolerances": dict(tol)}
        failures: list[str] = []
        for name in ("clean", "cf"):
            det = self.detail[name]
            cache = self._cache(name)
            worst = 0.0
            for layer in range(self.desc.n_layers):
                z_rec = self._z_from_qkv(
                    layer,
                    det.q[self.position_end][:, layer],
                    det.k_all[:, layer],
                    det.v_all[:, layer],
                    name,
                )
                z_cached = cache.z[self.position_end][:, layer]
                err = (
                    (z_rec.cpu() - z_cached).abs().max()
                    / z_cached.abs().max().clamp_min(1e-12)
                ).item()
                worst = max(worst, err)
            report[f"K1_z_reconstruction_{name}"] = worst
            if worst > tol["z_reconstruction_rel"]:
                failures.append(
                    f"K1 z reconstruction ({name} cache): worst per-layer "
                    f"relative error {worst:.2e} > "
                    f"{tol['z_reconstruction_rel']:.0e}. Reconstructing z at "
                    f"the end position from captured q/k/v does not "
                    f"reproduce the model's own attention output. Likely "
                    f"causes: wrong rotation (position ids), wrong scaling/"
                    f"softcapping, or a mask mismatch."
                )
        if failures:
            from .guards import GuardError

            raise GuardError(
                "K/V construction guards failed:\n- " + "\n- ".join(failures)
            )
        return report
