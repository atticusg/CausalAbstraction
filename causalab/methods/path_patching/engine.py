"""Analytic path patching over cached activations.

Semantics follow Hanna et al. (2023) §3.1 / Fig 3: all receivers act at one
token position, where the residual stream is a sum of component
contributions (verified at construction by the additivity guard, never
assumed). Replacing a sender's contribution along a named edge set, with
every off-path component frozen at its clean value, reduces to delta
arithmetic on cached activations plus direct re-evaluation of the receiver
MLP branches (through the model's real norm modules) and of the model's own
final norm + LM head (+ logit softcapping where the model has one).

Freeze-recipe equivalence: run on clean input; only the sender's output at
the position changes; frozen components pass their clean values through;
receivers recompute on (clean input + accumulated on-path deltas). Verified
against an independent pyvene replace-intervention implementation
(``reference.py``) in the validation suite.

Sender vocabulary:
  ("head", layer, head)        attention head (contribution = z @ W_O slice)
  ("mlp", layer)               MLP block
  ("neuron", layer, index)     single MLP output-projection input channel
  ("neuron_group", layer, [i]) several channels of one MLP
  ("embed",)                   the embedding contribution
  ("group", [senders...])      union of senders (e.g. a circuit complement)

Branch post-norms (Gemma-2): a branch's contribution to the trunk passes
through its post-norm, which is nonlinear, so sender deltas are computed as
``post_norm(branch_clean + raw_delta) - post_norm(branch_clean)`` — exactly
what a live patched forward produces. Raw deltas from senders inside the
same branch are summed *before* the post-norm.

Block order: on parallel-residual models (Pythia), the MLP reads the block
*input*, so a same-layer head never feeds the same-layer MLP and receiver
inputs are derived from ``block_input`` rather than ``block_output - mlp``.
Declared order is verified empirically at construction (guards).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .cache import PatchCache
from .descriptor import ArchitectureDescriptor
from .edges import PathSpec

__all__ = ["PatchEngine", "Sender"]

Sender = tuple  # see module docstring for the shapes


def _is_gemma_norm(module: nn.Module) -> bool:
    return "gemma" in module.__class__.__name__.lower()


class PatchEngine:
    """Analytic patched logits from a clean and a counterfactual cache.

    Construction runs the empirical guards by default (additivity of trunk
    contributions, per-layer branch reconstruction, patch-nothing /
    patch-everything closure); see ``guards.py``. An engine that fails its
    guards refuses to patch.
    """

    def __init__(
        self,
        desc: ArchitectureDescriptor,
        cache_clean: PatchCache,
        cache_cf: PatchCache,
        *,
        position: str = "end",
        run_guards: bool = True,
        guard_tolerances: dict[str, float] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.desc = desc
        self.clean = cache_clean
        self.cf = cache_cf
        self.position = position
        model = desc.model
        self.device = device or next(model.parameters()).device
        self.model_dtype = next(model.parameters()).dtype
        L = desc.n_layers
        self._w_o = [desc.attn_out_weight(li).float() for li in range(L)]
        self._b_o = [
            b.float() if (b := desc.attn_out_bias(li)) is not None else None
            for li in range(L)
        ]
        self._w_out: list[Tensor | None] = [None] * L  # lazy: (d_ff, d) float32
        self._attn_pre_clean: dict[tuple[str, int], Tensor] = {}
        self._mlp_trunk_clean: dict[int, Tensor] = {}
        from .provenance import check_capability, log_provenance

        # capability contract: every pyvene unit the engine and its twin
        # need must exist in the family's mapping, else this raises
        # UnsupportedArchitectureError before anything is patched.
        self.capability = check_capability(
            model,
            ["path-patching cache collection", "reference-twin interventions"],
        )
        #: which pyvene units / module calls this engine uses (pyvene-named /
        #: pyvene-path / direct-module-call); also logged at INFO once.
        self.provenance: list[dict[str, Any]] = log_provenance(desc)
        self.guard_report: dict[str, Any] | None = None
        if run_guards:
            from .guards import run_construction_guards

            self.guard_report = run_construction_guards(
                self, tolerances=guard_tolerances
            )

    # ------------------------------------------------------------------
    # cached-quantity helpers (float32, on device)
    # ------------------------------------------------------------------
    def _at(self, cache: PatchCache, quantity: str, layer: int | None = None) -> Tensor:
        t = getattr(cache, quantity)[self.position]
        if layer is not None:
            t = t[:, layer]
        return t.to(self.device)

    def _mlp_out_weight(self, layer: int) -> Tensor:
        if self._w_out[layer] is None:
            self._w_out[layer] = self.desc.mlp_out_weight(layer).float()
        return self._w_out[layer]

    def _apply_norm(
        self, module: nn.Module | None, x: Tensor, frozen_at: Tensor | None = None
    ) -> Tensor:
        """Apply a norm module (or identity). ``frozen_at`` freezes the
        normalization denominator at another tensor's value (linearized-norm
        diagnostic, rust-circuit style)."""
        if module is None:
            return x
        if frozen_at is None:
            return module(x.to(self.model_dtype)).float()
        xf = x.float()
        ref = frozen_at.float()
        eps = getattr(module, "eps", None) or getattr(module, "variance_epsilon", 1e-6)
        if isinstance(module, nn.LayerNorm):
            mu = xf.mean(-1, keepdim=True)
            var = ref.var(-1, keepdim=True, unbiased=False)
            xn = (xf - mu) / torch.sqrt(var + eps)
            return xn * module.weight.float() + module.bias.float()
        # RMSNorm family (no mean subtraction)
        rms = torch.sqrt(ref.pow(2).mean(-1, keepdim=True) + eps)
        xn = xf / rms
        w = module.weight.float()
        if _is_gemma_norm(module):
            return xn * (1.0 + w)
        return xn * w

    def _attn_pre_norm_clean(
        self, cache_key: str, cache: PatchCache, layer: int
    ) -> Tensor:
        """The attention branch's output before its post-norm (reconstructed
        as z @ W_O + bias; validated by the additivity guard)."""
        key = (cache_key, layer)
        if key not in self._attn_pre_clean:
            z = self._at(cache, "z", layer)  # (N, H, dh)
            flat = z.reshape(z.shape[0], -1)
            out = flat @ self._w_o[layer]
            if self._b_o[layer] is not None:
                out = out + self._b_o[layer]
            self._attn_pre_clean[key] = out
        return self._attn_pre_clean[key]

    def mlp_trunk_contribution(self, cache: PatchCache, layer: int) -> Tensor:
        """The MLP branch's contribution to the trunk (through its post-norm
        where one exists)."""
        pre = self._at(cache, "mlp_branch", layer)
        return self._apply_norm(self.desc.mlp_post_norm(layer), pre)

    def attn_trunk_contribution(self, cache: PatchCache, layer: int) -> Tensor:
        key = "clean" if cache is self.clean else "cf"
        pre = self._attn_pre_norm_clean(key, cache, layer)
        return self._apply_norm(self.desc.attn_post_norm(layer), pre)

    def resid_for_mlp(self, cache: PatchCache, layer: int) -> Tensor:
        """Residual stream entering layer ``layer``'s MLP branch."""
        if self.desc.mlp_input_resid_is_block_input():
            if layer == 0:
                return self._at(cache, "embed")
            return self._at(cache, "block_out", layer - 1)
        return self._at(cache, "block_out", layer) - self.mlp_trunk_contribution(
            cache, layer
        )

    # ------------------------------------------------------------------
    # sender deltas
    # ------------------------------------------------------------------
    def _raw_branch_deltas(
        self,
        sender: Sender,
        from_cache: PatchCache,
        to_cache: PatchCache,
        acc: dict[tuple[str, int], Tensor],
        embed_acc: list[Tensor],
    ) -> None:
        kind = sender[0]
        if kind == "head":
            _, layer, head = sender
            dz = (
                self._at(to_cache, "z", layer)[:, head]
                - self._at(from_cache, "z", layer)[:, head]
            )
            w = self._w_o[layer][
                head * self.desc.head_dim : (head + 1) * self.desc.head_dim
            ]
            self._acc(acc, ("attn", layer), dz @ w)
        elif kind == "mlp":
            _, layer = sender
            d = self._at(to_cache, "mlp_branch", layer) - self._at(
                from_cache, "mlp_branch", layer
            )
            self._acc(acc, ("mlp", layer), d)
        elif kind in ("neuron", "neuron_group"):
            _, layer, idx = sender
            if not from_cache.neuron_acts:
                raise ValueError(
                    "neuron senders need caches built with collect_neuron_acts=True"
                )
            sel = [idx] if kind == "neuron" else list(idx)
            da = (
                self._at(to_cache, "neuron_acts", layer)[:, sel]
                - self._at(from_cache, "neuron_acts", layer)[:, sel]
            )
            self._acc(acc, ("mlp", layer), da @ self._mlp_out_weight(layer)[sel])
        elif kind == "embed":
            embed_acc.append(
                self._at(to_cache, "embed") - self._at(from_cache, "embed")
            )
        elif kind == "group":
            for s in sender[1]:
                self._raw_branch_deltas(s, from_cache, to_cache, acc, embed_acc)
        else:
            raise ValueError(f"unknown sender {sender!r}")

    @staticmethod
    def _acc(acc: dict, key: tuple[str, int], val: Tensor) -> None:
        acc[key] = acc[key] + val if key in acc else val

    @torch.no_grad()
    def trunk_delta(
        self,
        sender: Sender,
        *,
        from_cache: PatchCache | None = None,
        to_cache: PatchCache | None = None,
        freeze_norms: bool = False,
    ) -> Tensor:
        """(N, d) change in the sender's contribution to the trunk when its
        activation moves from ``from_cache`` (default clean) to ``to_cache``
        (default counterfactual). Raw deltas within one branch are summed
        before that branch's post-norm."""
        from_cache = from_cache or self.clean
        to_cache = to_cache or self.cf
        acc: dict[tuple[str, int], Tensor] = {}
        embed_acc: list[Tensor] = []
        self._raw_branch_deltas(sender, from_cache, to_cache, acc, embed_acc)
        cache_key = "clean" if from_cache is self.clean else "cf"

        total: Tensor | None = None
        for (branch, layer), raw in acc.items():
            post = (
                self.desc.attn_post_norm(layer)
                if branch == "attn"
                else self.desc.mlp_post_norm(layer)
            )
            if post is None:
                d = raw
            else:
                pre = (
                    self._attn_pre_norm_clean(cache_key, from_cache, layer)
                    if branch == "attn"
                    else self._at(from_cache, "mlp_branch", layer)
                )
                frozen = pre if freeze_norms else None
                d = self._apply_norm(
                    post, pre + raw, frozen_at=frozen
                ) - self._apply_norm(post, pre, frozen_at=frozen)
            total = d if total is None else total + d
        for e in embed_acc:
            total = e if total is None else total + e
        if total is None:
            n = from_cache.n_examples
            total = torch.zeros(n, self.desc.d_model, device=self.device)
        return total

    # ------------------------------------------------------------------
    # patched forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def receiver_deltas(
        self,
        sender_delta: Tensor,
        spec: PathSpec,
        *,
        base_cache: PatchCache | None = None,
        freeze_norms: bool = False,
    ) -> dict[int, Tensor]:
        """Per-receiver trunk-output deltas at this engine's position.

        Runs the spec's receiver cascade on a precomputed (N, d) sender
        trunk delta and returns each receiver MLP's output delta, without
        assembling final logits — the building block for feeding a receiver
        chain into something other than the logits (e.g. the K/V engine's
        patched positions).
        """
        base = base_cache or self.clean
        sd = sender_delta.to(self.device).float()
        deltas: dict[int, Tensor] = {}
        for k in spec.receivers:
            inp = torch.zeros_like(sd)
            live = False
            if k in spec.sender_to:
                inp = inp + sd
                live = True
            for j, kk in spec.receiver_to_receiver:
                if kk == k and j in deltas:
                    inp = inp + deltas[j]
                    live = True
            if not live:
                deltas[k] = torch.zeros_like(sd)
                continue
            resid = self.resid_for_mlp(base, k)
            freeze_at = resid if freeze_norms else None
            new = self._mlp_branch_fn(k, resid + inp, freeze_at=freeze_at)
            old = self.mlp_trunk_contribution(base, k)
            deltas[k] = new - old
        return deltas

    def _mlp_branch_fn(
        self, layer: int, resid: Tensor, *, freeze_at: Tensor | None = None
    ) -> Tensor:
        """Re-evaluate layer ``layer``'s full MLP branch (pre-norm, MLP,
        post-norm where present) on ``resid``. ``freeze_at`` freezes norm
        denominators at that residual's value (diagnostic)."""
        pre_norm = self.desc.mlp_pre_norm(layer)
        x = self._apply_norm(pre_norm, resid, frozen_at=freeze_at)
        m = self.desc.mlp(layer)(x.to(self.model_dtype)).float()
        post = self.desc.mlp_post_norm(layer)
        if post is None:
            return m
        # freezing the post-norm at the clean branch output value
        frozen = (
            self._at(self.clean, "mlp_branch", layer) if freeze_at is not None else None
        )
        return self._apply_norm(post, m, frozen_at=frozen)

    def tail_logits(
        self, final_resid: Tensor, *, freeze_at: Tensor | None = None
    ) -> Tensor:
        """Final logits from a reassembled final-position residual, through
        the model's own final norm + LM head (+ softcapping).

        Softcapping runs in the LM head's output dtype, mirroring the HF
        forward exactly — applying it in float32 instead rounds differently
        and costs ~1 bf16 ulp at the cap scale (caught by the patch-nothing
        closure guard on Gemma-2)."""
        x = self._apply_norm(self.desc.final_norm(), final_resid, frozen_at=freeze_at)
        logits = self.desc.lm_head()(x.to(self.model_dtype))
        cap = self.desc.final_logit_softcapping
        if cap is not None:
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return logits.float().cpu()

    @torch.no_grad()
    def patched_logits(
        self,
        sender: Sender | Tensor,
        spec: PathSpec | None = None,
        *,
        base_cache: PatchCache | None = None,
        freeze_norms: bool = False,
    ) -> Tensor:
        """Patched final logits (N, vocab) for ``sender`` along ``spec``.

        ``sender`` is a sender tuple (its trunk delta is computed from the
        engine's caches) or a precomputed (N, d) trunk-delta tensor.
        ``base_cache`` is the cache the patched run otherwise follows
        (default clean). ``freeze_norms`` freezes every recomputed norm's
        denominator at its base value (linearized-norm diagnostic; default
        recomputes norms exactly).
        """
        spec = spec or PathSpec.cascade()
        base = base_cache or self.clean
        if isinstance(sender, Tensor):
            sd = sender.to(self.device).float()
        else:
            to_cache = self.cf if base is self.clean else self.clean
            sd = self.trunk_delta(
                sender, from_cache=base, to_cache=to_cache, freeze_norms=freeze_norms
            )

        deltas = self.receiver_deltas(
            sd, spec, base_cache=base, freeze_norms=freeze_norms
        )

        final = base.final_resid(self.position).to(self.device).float()
        for k in spec.receivers_to_logits:
            if k in deltas:
                final = final + deltas[k]
        if spec.sender_to_logits:
            final = final + sd
        freeze_at = (
            base.final_resid(self.position).to(self.device) if freeze_norms else None
        )
        return self.tail_logits(final, freeze_at=freeze_at)

    # ------------------------------------------------------------------
    # circuit evaluation (Hanna et al. §3.2 / Fig 5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def circuit_eval(
        self,
        circuit_heads: Sequence[tuple[int, int]],
        circuit_mlps: Sequence[int],
        direction: str = "sufficiency",
        head_receiver_mlps: Sequence[int]
        | dict[tuple[int, int], Sequence[int]]
        | None = None,
    ) -> Tensor:
        """Patched logits for a circuit evaluation.

        sufficiency: model runs on the counterfactual side; circuit paths
        (heads->MLPs, heads->logits, MLPs->MLPs->logits) restored to clean.
        necessity: model runs on the clean side; circuit paths receive the
        counterfactual. Heads always keep their direct edge to the logits;
        circuit MLPs feed all downstream circuit MLPs and the logits.
        ``head_receiver_mlps`` limits which circuit MLPs receive the heads'
        deltas: one list applying to every head, or a per-head dict.
        """
        if head_receiver_mlps is None:
            head_receiver_mlps = list(circuit_mlps)
        if not isinstance(head_receiver_mlps, dict):
            head_receiver_mlps = {hd: list(head_receiver_mlps) for hd in circuit_heads}
        if direction == "sufficiency":
            base, restore = self.cf, self.clean
        elif direction == "necessity":
            base, restore = self.clean, self.cf
        else:
            raise ValueError(direction)

        # per-(layer, head-subset) trunk deltas, grouped per branch so any
        # branch post-norm applies to the summed raw delta
        def heads_trunk_delta(heads: Sequence[tuple[int, int]]) -> Tensor | None:
            if not heads:
                return None
            return self.trunk_delta(
                ("group", [("head", hl, hh) for hl, hh in heads]),
                from_cache=base,
                to_cache=restore,
            )

        all_heads = list(circuit_heads)
        mlp_deltas: list[tuple[int, Tensor]] = []
        subset_cache: dict[frozenset, Tensor | None] = {}
        for k in sorted(circuit_mlps):
            feeding = [
                (hl, hh)
                for (hl, hh) in all_heads
                if self.desc.head_feeds_mlp(hl, k)
                and k in head_receiver_mlps.get((hl, hh), ())
            ]
            key = frozenset(feeding)
            if key not in subset_cache:
                subset_cache[key] = heads_trunk_delta(feeding)
            inp_parts = [] if subset_cache[key] is None else [subset_cache[key]]
            inp_parts += [d for j, d in mlp_deltas if j < k]
            resid = self.resid_for_mlp(base, k)
            inp = sum(inp_parts, torch.zeros_like(resid))
            new = self._mlp_branch_fn(k, resid + inp)
            old = self.mlp_trunk_contribution(base, k)
            mlp_deltas.append((k, new - old))

        final = base.final_resid(self.position).to(self.device).float()
        full_heads_delta = heads_trunk_delta(all_heads)
        if full_heads_delta is not None:
            final = final + full_heads_delta
        for _, d in mlp_deltas:
            final = final + d
        return self.tail_logits(final)
