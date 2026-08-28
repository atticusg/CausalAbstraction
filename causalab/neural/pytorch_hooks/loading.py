"""Model, tokenizer and tensor-bundle loading for the reference backend.

One bundle per (key, revision, dtype, device): the HF causal-LM, its
tokenizer configured for the backend's one padding convention
(**left**-padded, ``pad = eos``), and the model's static metadata
registered into the protocol model registry so canonicalization inside a
run needs no pre-registration.

Attention is forced **eager**: the captured-goldens context pins eager
(SDPA/flash change activations at tolerance-relevant scale), and eager
keeps every module boundary a real module call for the hooks to fire on.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import torch

from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import (
    ModelInfo,
    model_info_from_hf_config,
    register_model,
)

__all__ = ["BundlePoint", "ModelBundle", "TensorBundle", "load_model"]

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


#: The mixer children that mean a layer runs full (softmax) attention, and the
#: ones that mean it runs a linear-attention kernel. Named here so
#: :meth:`ModelBundle.stream_at` and :meth:`ModelBundle.mixer_at` read the same
#: table instead of each carrying its own probe order.
_FULL_ATTENTION_CHILDREN: tuple[str, ...] = ("self_attn", "attn")
_LINEAR_ATTENTION_CHILDREN: tuple[str, ...] = ("linear_attn",)


@dataclasses.dataclass(frozen=True)
class ModelBundle:
    """One loaded model with everything the executor needs."""

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    device: str
    dtype: str

    @property
    def is_gpt2_family(self) -> bool:
        return hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        )

    @property
    def blocks(self) -> Any:
        """The decoder-layer ModuleList, whichever tree this family uses."""
        return (
            self.model.transformer.h if self.is_gpt2_family else self.model.model.layers
        )

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` actually carries.

        A hybrid architecture varies this *per layer*: 📐 on
        ``tiny-random/qwen3.5-moe`` the text tower is
        ``['linear_attention', 'linear_attention', 'linear_attention',
        'full_attention']`` — three Gated DeltaNet blocks and one gated full
        attention. So a boolean family flag cannot answer this, and neither can
        the config alone: it is read off the module that is really there, which
        is what a hook has to attach to.

        Returns one of ``"full_attention"`` (a ``self_attn``/``attn`` child) or
        ``"linear_attention"`` (a ``linear_attn`` child).

        Raises:
            ProtocolError: the block has no recognised mixer child, or has
                children of *both* kinds. The second case is hypothetical — no
                family in the round-1 box map ships it — but probing in a fixed
                order would answer "full_attention" for it silently, and every
                per-layer tap downstream would then attach to the wrong module
                and still produce plausible numbers. A named refusal is the
                same trade this module makes everywhere else.
        """
        block = self.blocks[layer]
        full = [name for name in _FULL_ATTENTION_CHILDREN if hasattr(block, name)]
        linear = [name for name in _LINEAR_ATTENTION_CHILDREN if hasattr(block, name)]
        if full and linear:
            raise ProtocolError(
                "P4",
                f"layer {layer} of {self.key!r} carries both a full-attention "
                f"child ({', '.join(full)}) and a linear-attention child "
                f"({', '.join(linear)}) — the stream of a layer must be one or "
                "the other, so extend the stream table in "
                "pytorch_hooks/loading.py to say which this family means",
            )
        if full:
            return "full_attention"
        if linear:
            return "linear_attention"
        raise ProtocolError(
            "P4",
            f"layer {layer} of {self.key!r} has no recognised mixer child "
            f"(children={sorted(name for name, _ in block.named_children())}) — "
            "extend the stream table in pytorch_hooks/loading.py",
        )

    def mixer_at(self, layer: int) -> Any:
        """The attention/mixer module at ``layer``, whichever stream it is.

        Resolved *through* :meth:`stream_at` rather than by its own probe, so
        the two can never disagree about a block: one answer, one place.
        """
        block = self.blocks[layer]
        names = (
            _FULL_ATTENTION_CHILDREN
            if self.stream_at(layer) == "full_attention"
            else _LINEAR_ATTENTION_CHILDREN
        )
        for name in names:
            child = getattr(block, name, None)
            if child is not None:
                return child
        raise AssertionError("unreachable")  # stream_at only answers if one exists

    @property
    def streams(self) -> tuple[str, ...]:
        """``stream_at`` for every layer — the whole tower's shape at a glance."""
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))


@functools.lru_cache(maxsize=4)
def load_model(
    key: str, revision: str = "main", *, dtype: str = "fp32", device: str = "cpu"
) -> ModelBundle:
    """Load (and cache) one model bundle.

    The tokenizer is set to the backend's single padding convention: left
    padding with ``pad = eos`` when the checkpoint ships no pad token — the
    inherited pipeline contract the oracle tests were captured under.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        key,
        revision=revision,
        dtype=_DTYPES[dtype],
        attn_implementation="eager",
    )
    torch.nn.Module.to(model, torch.device(device))
    model.eval()
    # only featurizer/free params ever train (§2.11); freezing the network
    # keeps training graphs from accumulating gradients into model weights
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    return ModelBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        device=device,
        dtype=dtype,
    )


@dataclasses.dataclass(frozen=True)
class BundlePoint:
    """One producing point's slice of a bundle: the tensors sharing a
    coordinate suffix, plus that entry's stamped record.

    Slicing by suffix rather than selecting each slot on its own is what
    keeps a multi-slot bundle coherent — an SAE's ``enc`` and ``dec`` must
    come from the same fit, not from whichever entries each lookup found.
    """

    tensors: dict[str, torch.Tensor]
    suffix: str
    record: dict[str, Any]
    what: str

    def tensor(self, slot: str) -> torch.Tensor:
        key = f"{slot}{self.suffix}"
        if key not in self.tensors:
            raise ProtocolError(
                "P2",
                f"{self.what}: the bundle has no {key!r} — an entry's slots "
                f"must be complete (has {sorted(self.tensors)})",
            )
        return self.tensors[key]


@dataclasses.dataclass(frozen=True)
class TensorBundle:
    """One loaded ``.safetensors`` file: its tensors plus the ``entries``
    table from the header (§8, :mod:`causalab.protocol.bundles`).

    :meth:`point` is the only way in. A bundle written by a swept document
    holds one entry per point per slot, so asking for a bare slot name would
    either ``KeyError`` or — worse — silently take whichever entry a plain
    dict lookup happened to find.
    """

    tensors: dict[str, torch.Tensor]
    entry_coords: dict[str, Any]

    def point(
        self,
        slot: str,
        want: Any,
        *,
        what: str,
        implicit: bool = False,
    ) -> BundlePoint:
        """The entry for ``slot`` selected by ``want`` (a coordinate
        mapping; ``implicit`` when derived from the consuming point rather
        than authored), as a coherent slice of the bundle."""
        from causalab.protocol.bundles import select_entry

        key = select_entry(
            self.tensors.keys(),
            slot,
            want,
            what=what,
            coords_by_key=self.entry_coords or None,
            implicit=implicit,
        )
        record = self.entry_coords.get(key, {})
        return BundlePoint(
            tensors=self.tensors,
            suffix=key[len(slot) :],
            record=record if isinstance(record, dict) else {},
            what=what,
        )
