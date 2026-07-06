"""Activation cache for the analytic path-patching engine.

One pass over a dataset collects, at each named token position, every
quantity the engine needs:

* ``z``            (N, L, H, head_dim)  per-head attention value output
                   (the out-projection's input)
* ``mlp_branch``   (N, L, d)   MLP module output — *before* any branch
                   post-norm (Gemma-2), so trunk contributions are derived,
                   never assumed
* ``neuron_acts``  (N, L, d_ff)  the MLP output projection's input (post-GELU
                   activations for GPT-2; the gated ``act(gate) * up`` values
                   for Llama/Gemma) — optional
* ``block_out``    (N, L, d)   residual stream after each block
* ``embed``        (N, d)      block 0's input: the embedding contribution to
                   the trunk (absorbs Gemma's embed scaling)
* ``logits``       (N, vocab)  the model's final logits at the position

Collection is pyvene-native: named components (``attention_value_output``,
``mlp_output``, ``block_output``, ``block_input``) resolve through pyvene's
per-family module mappings; the one capture point pyvene has no name for
(the output projection's input) uses pyvene's dotted module-path fallback.
No raw torch hooks.

Each named position is **one token per example**, given in unpadded
per-example coordinates (a non-negative absolute index, or a negative offset
from the true end of each sequence — ``-1`` is the last real token) and
converted to padded batch coordinates using the attention mask, so
variable-length prompts under the pipeline's left padding are handled
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    prepare_intervenable_model,
)
from causalab.neural.units import AtomicModelUnit, ComponentIndexer

from .descriptor import ArchitectureDescriptor

__all__ = ["PatchCache", "build_patch_cache", "padded_position"]


def padded_position(attention_mask: Tensor, position: int | Sequence[int]) -> list[int]:
    """Convert an unpadded token position to padded batch coordinates.

    ``position`` is one index applied to every example, or a per-example
    sequence. A non-negative index counts from the start of the example's
    true (unpadded) token sequence; a negative index counts from its true
    end (``-1`` = last real token). Works for left or right padding by
    reading the attention mask.
    """
    mask = attention_mask.long()
    batch = mask.shape[0]
    true_len = mask.sum(dim=1)
    first_real = mask.argmax(dim=1)  # 0 under right padding

    if isinstance(position, int):
        per_example = [position] * batch
    else:
        per_example = list(position)
        if len(per_example) != batch:
            raise ValueError(f"{len(per_example)} positions for batch of {batch}")

    out: list[int] = []
    for i, p in enumerate(per_example):
        n = int(true_len[i])
        q = p if p >= 0 else n + p
        if not 0 <= q < n:
            raise IndexError(f"position {p} out of range for example {i} (length {n})")
        idx = int(first_real[i]) + q
        if mask[i, idx] != 1:
            raise AssertionError(f"internal error: padded index {idx} falls on padding")
        out.append(idx)
    return out


@dataclass
class PatchCache:
    """Cached activations at named single-token positions (float32, CPU)."""

    n_layers: int
    n_heads: int
    head_dim: int
    d_model: int
    position_names: tuple[str, ...]
    z: dict[str, Tensor]  # (N, L, H, head_dim)
    mlp_branch: dict[str, Tensor]  # (N, L, d)
    block_out: dict[str, Tensor]  # (N, L, d)
    embed: dict[str, Tensor]  # (N, d)
    logits: dict[str, Tensor]  # (N, vocab)
    neuron_acts: dict[str, Tensor] = field(default_factory=dict)  # (N, L, d_ff)

    @property
    def n_examples(self) -> int:
        return next(iter(self.block_out.values())).shape[0]

    def final_resid(self, position: str) -> Tensor:
        return self.block_out[position][:, self.n_layers - 1]

    def save(self, path: str) -> None:
        from safetensors.torch import save_file

        tensors: dict[str, Tensor] = {}
        for name in ("z", "mlp_branch", "block_out", "embed", "logits", "neuron_acts"):
            for pos, t in getattr(self, name).items():
                tensors[f"{name}@{pos}"] = t.contiguous()
        tensors["_meta"] = torch.tensor(
            [self.n_layers, self.n_heads, self.head_dim, self.d_model]
        )
        save_file(tensors, path, metadata={"positions": ",".join(self.position_names)})

    @classmethod
    def load(cls, path: str) -> "PatchCache":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, framework="pt") as f:
            positions = tuple(f.metadata()["positions"].split(","))
        t = load_file(path)
        meta = t.pop("_meta")
        groups: dict[str, dict[str, Tensor]] = {
            k: {}
            for k in ("z", "mlp_branch", "block_out", "embed", "logits", "neuron_acts")
        }
        for key, val in t.items():
            name, pos = key.split("@", 1)
            groups[name][pos] = val
        return cls(
            n_layers=int(meta[0]),
            n_heads=int(meta[1]),
            head_dim=int(meta[2]),
            d_model=int(meta[3]),
            position_names=positions,
            **groups,
        )


def _null_indexer() -> ComponentIndexer:
    # Indices are supplied explicitly at collection time (computed padding-
    # aware in this module); the unit-level indexer is never consulted.
    return ComponentIndexer(lambda _x: [], id="path_patching_explicit")


@torch.no_grad()
def build_patch_cache(
    pipeline: Any,
    desc: ArchitectureDescriptor,
    inputs: Sequence[Any],
    positions: dict[str, int | Sequence[int]],
    *,
    batch_size: int = 32,
    collect_neuron_acts: bool = True,
) -> PatchCache:
    """Collect a :class:`PatchCache` over ``inputs`` at ``positions``.

    ``inputs`` are raw strings or causalab trace dicts (anything
    ``pipeline.load`` accepts). ``positions`` maps a name (e.g. ``"end"``)
    to one token position per example, in unpadded per-example coordinates
    (see :func:`padded_position`).
    """
    if not positions:
        raise ValueError("at least one named position is required")
    pos_names = tuple(positions)
    inputs = [{"raw_input": x} if isinstance(x, str) else x for x in inputs]

    L = desc.n_layers
    units: list[AtomicModelUnit] = []
    unit_meta: list[tuple[str, int]] = []  # (quantity, layer)

    def add_unit(quantity: str, layer: int, component: str) -> None:
        units.append(
            AtomicModelUnit(
                layer, component, _null_indexer(), id=f"pp_{quantity}_l{layer}"
            )
        )
        unit_meta.append((quantity, layer))

    for layer in range(L):
        add_unit("z", layer, desc.component_head_values())
        add_unit("mlp_branch", layer, desc.component_mlp_branch())
        add_unit("block_out", layer, desc.component_block_output())
        if collect_neuron_acts:
            add_unit("neuron_acts", layer, desc.component_neuron_values(layer))
    add_unit("embed", 0, desc.component_block_input())

    intervenable_model = prepare_intervenable_model(
        pipeline, units, intervention_type="collect"
    )

    store: dict[tuple[str, int], list[Tensor]] = {m: [] for m in unit_meta}
    logits_store: list[Tensor] = []
    P = len(pos_names)

    try:
        for start in range(0, len(inputs), batch_size):
            batch = list(inputs[start : start + batch_size])
            loaded = pipeline.load(batch)
            mask = loaded["attention_mask"]
            bsz = mask.shape[0]
            per_name = {
                name: padded_position(
                    mask,
                    spec
                    if isinstance(spec, int)
                    else list(spec)[start : start + bsz]
                    if len(list(spec)) == len(inputs)
                    else spec,
                )
                for name, spec in positions.items()
            }
            flat = [[per_name[name][i] for name in pos_names] for i in range(bsz)]
            indices = [flat for _ in units]

            location_map = {"sources->base": (indices, indices)}
            result = intervenable_model(
                loaded, unit_locations=location_map, output_original_output=True
            )
            collected = result[0][1]
            model_output = result[0][0]

            for meta, act in zip(unit_meta, collected):
                a = act if isinstance(act, Tensor) else torch.cat(list(act))
                a = a.reshape(bsz, P, -1).float().cpu()
                store[meta].append(a)

            logits = model_output.logits.float()
            idx = torch.tensor(flat, device=logits.device)
            gathered = torch.stack([logits[i, idx[i]] for i in range(bsz)])
            logits_store.append(gathered.cpu())  # (B, P, vocab)
    finally:
        delete_intervenable_model(intervenable_model)

    pos_slot = {name: i for i, name in enumerate(pos_names)}

    def split(chunks: list[Tensor]) -> dict[str, Tensor]:
        full = torch.cat(chunks)  # (N, P, dim)
        return {name: full[:, pos_slot[name]] for name in pos_names}

    def stack_layers(quantity: str) -> dict[str, Tensor]:
        per_layer = [split(store[(quantity, layer)]) for layer in range(L)]
        return {
            name: torch.stack([pl[name] for pl in per_layer], dim=1)
            for name in pos_names
        }

    z_flat = stack_layers("z")
    z = {
        name: t.reshape(t.shape[0], L, desc.n_heads, desc.head_dim)
        for name, t in z_flat.items()
    }
    logits_full = torch.cat(logits_store)
    return PatchCache(
        n_layers=L,
        n_heads=desc.n_heads,
        head_dim=desc.head_dim,
        d_model=desc.d_model,
        position_names=pos_names,
        z=z,
        mlp_branch=stack_layers("mlp_branch"),
        block_out=stack_layers("block_out"),
        embed=split(store[("embed", 0)]),
        logits={name: logits_full[:, pos_slot[name]] for name in pos_names},
        neuron_acts=stack_layers("neuron_acts") if collect_neuron_acts else {},
    )
