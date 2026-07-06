"""pyvene replace-intervention reference patcher (validation twin).

Implements the freeze recipe with *live intervened forward passes*:
run on the clean inputs; replace the sender's raw activation at the patched
position with its counterfactual cached value; freeze every component
strictly downstream of the sender at its clean cached value, except the
receiver MLPs (and the final norm + LM head), which recompute. Edges the
path spec excludes are cancelled by subtracting the corresponding delta at
the receiving module's input:

* an excluded sender edge subtracts the sender's (cached) trunk delta;
* an excluded receiver->receiver or receiver->logits edge subtracts the
  upstream receiver's *live* output delta, recorded earlier in the same
  forward pass (upstream modules run first, so the value is available).

Everything runs through pyvene: static substitutions, freezes, delta
subtraction, and live recording are all constant-source interventions
attached via pyvene's component mappings (dotted-path fallback for norm
inputs/outputs). No raw torch hooks.

Deliberately independent of the analytic engine — real forwards with module
substitution vs delta arithmetic on caches — so their agreement is
informative. Slow; validation only.
"""

from __future__ import annotations

from typing import Any, Sequence

import pyvene as pv
import torch
from torch import Tensor

from .cache import PatchCache, padded_position
from .descriptor import ArchitectureDescriptor
from .edges import PathSpec
from .engine import PatchEngine, Sender

__all__ = ["reference_patched_logits"]


class _ValueSetter(pv.ConstantSourceIntervention):
    """Set the gathered slice (optionally a channel subset) to a stored value."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.payload: Tensor | None = None
        self.channels: list[int] | None = None

    def forward(self, base, source=None, subspaces=None, **kwargs):
        assert self.payload is not None, "payload unset"
        val = self.payload.to(base.dtype).to(base.device)
        if self.channels is None:
            return val.reshape(base.shape)
        out = base.clone()
        out[..., self.channels] = val.reshape(out[..., self.channels].shape)
        return out


class _DeltaSubtract(pv.ConstantSourceIntervention):
    """Subtract a sum of delta terms (static tensors or live-recorded keys)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.static_terms: list[Tensor] = []
        self.live_keys: list[str] = []
        self.store: dict[str, Tensor] | None = None

    def forward(self, base, source=None, subspaces=None, **kwargs):
        out = base
        for t in self.static_terms:
            out = out - t.to(base.dtype).to(base.device).reshape(base.shape)
        for k in self.live_keys:
            assert self.store is not None and k in self.store, (
                f"live delta {k!r} not recorded before use (module order bug)"
            )
            out = out - self.store[k].to(base.dtype).to(base.device).reshape(base.shape)
        return out


class _LiveDeltaRecorder(pv.ConstantSourceIntervention):
    """Record (live - reference) for the gathered slice; pass through."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reference: Tensor | None = None
        self.key: str = ""
        self.store: dict[str, Tensor] | None = None

    def forward(self, base, source=None, subspaces=None, **kwargs):
        assert self.reference is not None and self.store is not None
        self.store[self.key] = (
            base.float() - self.reference.to(base.device).float().reshape(base.shape)
        )
        return base


def _downstream_components(
    desc: ArchitectureDescriptor, sender: Sender
) -> tuple[list[tuple[int]], list[int]]:
    """(attention layers, MLP layers) strictly downstream of the sender."""
    kind = sender[0]
    if kind == "group":
        raise NotImplementedError(
            "the reference twin validates atomic senders; validate group "
            "members individually"
        )
    if kind == "embed":
        return list(range(desc.n_layers)), list(range(desc.n_layers))
    layer = sender[1]
    attn_layers = [l for l in range(desc.n_layers) if l > layer]
    if kind == "head":
        mlp_layers = [
            m for m in range(desc.n_layers) if desc.head_feeds_mlp(layer, m)
        ]
    else:  # mlp / neuron / neuron_group
        mlp_layers = [m for m in range(desc.n_layers) if m > layer]
    return attn_layers, mlp_layers


@torch.no_grad()
def reference_patched_logits(
    pipeline: Any,
    desc: ArchitectureDescriptor,
    inputs_clean: Sequence[Any],
    cache_clean: PatchCache,
    cache_cf: PatchCache,
    sender: Sender,
    spec: PathSpec,
    *,
    position: str = "end",
    position_index: int | Sequence[int] = -1,
    engine: PatchEngine | None = None,
) -> Tensor:
    """Patched final logits (N, vocab) via a live intervened forward.

    ``position_index`` is the patched position in unpadded per-example
    coordinates (matching what the caches were built at for ``position``).
    ``engine`` (optional) supplies the sender's cached trunk delta for
    excluded-sender-edge cancellation; one is built without guards if absent.
    """
    if engine is None:
        engine = PatchEngine(
            desc, cache_clean, cache_cf, position=position, run_guards=False
        )
    inputs_clean = [
        {"raw_input": x} if isinstance(x, str) else x for x in inputs_clean
    ]
    loaded = pipeline.load(list(inputs_clean))
    mask = loaded["attention_mask"]
    pos_padded = padded_position(mask, position_index)
    batch = mask.shape[0]

    store: dict[str, Tensor] = {}
    configs: list[dict[str, Any]] = []
    setups: list[Any] = []  # callables run on the instantiated interventions

    def add(component: str, layer: int, unit: str, cls: type, setup) -> None:
        configs.append(
            {
                "component": component,
                "unit": unit,
                "layer": layer,
                "intervention_type": cls,
            }
        )
        setups.append(setup)

    receivers = set(spec.receivers)
    kind = sender[0]

    # ---- 1. sender substitution (cf value at the position) ----
    if kind == "head":
        _, s_layer, s_head = sender
        z_cf = cache_cf.z[position][:, s_layer, s_head]  # (N, dh)

        def setup_sender(iv, val=z_cf):
            iv.payload = val

        add("head_attention_value_output", s_layer, "h.pos", _ValueSetter, setup_sender)
    elif kind == "mlp":
        _, s_layer = sender
        val = cache_cf.mlp_branch[position][:, s_layer]

        def setup_sender(iv, val=val):
            iv.payload = val

        add("mlp_output", s_layer, "pos", _ValueSetter, setup_sender)
    elif kind in ("neuron", "neuron_group"):
        _, s_layer, idx = sender
        channels = [idx] if kind == "neuron" else list(idx)
        val = cache_cf.neuron_acts[position][:, s_layer][:, channels]

        def setup_sender(iv, val=val, ch=channels):
            iv.payload = val
            iv.channels = ch

        add(
            desc.component_neuron_values(s_layer),
            s_layer,
            "pos",
            _ValueSetter,
            setup_sender,
        )
    elif kind == "embed":
        val = cache_cf.embed[position]

        def setup_sender(iv, val=val):
            iv.payload = val

        add("block_input", 0, "pos", _ValueSetter, setup_sender)
    else:
        raise NotImplementedError(f"reference twin: unsupported sender {sender!r}")

    # ---- 2. freeze downstream components at clean values ----
    attn_layers, mlp_layers = _downstream_components(desc, sender)
    for l in attn_layers:
        z_clean = cache_clean.z[position][:, l].reshape(batch, -1)

        def setup_freeze_attn(iv, val=z_clean):
            iv.payload = val

        add("attention_value_output", l, "pos", _ValueSetter, setup_freeze_attn)
    for m in mlp_layers:
        if m in receivers:
            continue
        val = cache_clean.mlp_branch[position][:, m]

        def setup_freeze_mlp(iv, val=val):
            iv.payload = val

        add("mlp_output", m, "pos", _ValueSetter, setup_freeze_mlp)

    # ---- 3. live recorders on receivers whose outgoing edges are cut ----
    for j in spec.receivers:
        cut_r2r = any(
            (j, k) not in spec.receiver_to_receiver
            for k in spec.receivers
            if k > j
        )
        cut_logits = j not in spec.receivers_to_logits
        if not (cut_r2r or cut_logits):
            continue
        ref = engine.mlp_trunk_contribution(cache_clean, j).cpu()

        def setup_rec(iv, ref=ref, key=f"recv{j}"):
            iv.reference = ref
            iv.key = key
            iv.store = store

        add(
            desc.component_mlp_trunk_output(j),
            j,
            "pos",
            _LiveDeltaRecorder,
            setup_rec,
        )

    # ---- 4. cancellation subtractors ----
    sender_trunk_delta = engine.trunk_delta(sender).cpu()  # (N, d)

    # at each receiver k: subtract the sender's delta if S->k is excluded but
    # the sender is upstream of k; subtract live deltas of excluded (j, k)
    def sender_upstream_of_mlp(k: int) -> bool:
        if kind == "embed":
            return True
        if kind == "head":
            return desc.head_feeds_mlp(sender[1], k)
        return sender[1] < k

    for k in spec.receivers:
        static_terms: list[Tensor] = []
        live_keys: list[str] = []
        if k not in spec.sender_to and sender_upstream_of_mlp(k):
            static_terms.append(sender_trunk_delta)
        for j in spec.receivers:
            if j < k and (j, k) not in spec.receiver_to_receiver:
                live_keys.append(f"recv{j}")
        if not static_terms and not live_keys:
            continue

        def setup_sub(iv, st=static_terms, lk=live_keys):
            iv.static_terms = st
            iv.live_keys = lk
            iv.store = store

        add(
            desc.component_mlp_pre_norm_input(k),
            k,
            "pos",
            _DeltaSubtract,
            setup_sub,
        )

    # at the final norm input: cancel the sender's direct edge and any
    # receiver->logits edge the spec excludes
    static_terms = [] if spec.sender_to_logits else [sender_trunk_delta]
    live_keys = [
        f"recv{j}" for j in spec.receivers if j not in spec.receivers_to_logits
    ]
    if static_terms or live_keys:

        def setup_final(iv, st=static_terms, lk=live_keys):
            iv.static_terms = st
            iv.live_keys = lk
            iv.store = store

        add(
            desc.component_final_norm_input(),
            desc.n_layers - 1,
            "pos",
            _DeltaSubtract,
            setup_final,
        )

    # ---- build the intervenable model, wire payloads, run ----
    iv_config = pv.IntervenableConfig(configs)
    iv_model = pv.IntervenableModel(iv_config, model=pipeline.model)
    iv_model.disable_model_gradients()
    try:
        keys_in_order = list(iv_model.interventions.keys())
        assert len(keys_in_order) == len(setups)
        for key, setup in zip(keys_in_order, setups):
            iv = iv_model.interventions[key]
            iv = iv[0] if isinstance(iv, tuple) else iv
            setup(iv)

        # unit locations: heads use [head, pos] pairs, others positions only
        indices: list[Any] = []
        for cfg in configs:
            if cfg["unit"] == "h.pos":
                head = sender[2]
                indices.append([[[head]] * batch, [[p] for p in pos_padded]])
            else:
                indices.append([[p] for p in pos_padded])
        location_map = {"sources->base": (indices, indices)}
        result = iv_model(loaded, unit_locations=location_map)
        output = result[0][0]
        logits = output.logits.float()
        idx = torch.tensor(pos_padded, device=logits.device)
        out = logits[torch.arange(batch, device=logits.device), idx]
        return out.cpu()
    finally:
        store.clear()
        del iv_model
