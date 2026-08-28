"""Model loading for the nnsight engine.

Wraps :class:`nnsight.modeling.transformers.TransformersModel` in a bundle
exposing the same surface the shared site map and executor base consume
(``model`` / ``tokenizer`` / ``info`` / ``blocks`` / ``stream_at`` /
``mixer_at``), so :func:`causalab.neural.shared.sites.resolve_site` addresses
the envoy tree exactly as it addresses the reference engine's module tree —
one component→module map, never forked (plan §2.3). The ``module`` a resolved
site carries is then an *envoy*, whose ``.input``/``.output`` the trace
executor reads and assigns.

Attention runs under the **model default** implementation (sdpa on the
target families): a document that never touches the scores or the pattern
pays nothing for them. A group whose interior addresses require eager
switches it on around its own trace and restores the default after —
:meth:`TracePointExecutor._switched_implementations` (D5, decided: on
demand). Callers that must compare like against like — the cross-engine
parity suite, whose reference engine loads eager — pin it explicitly with
``attn_implementation="eager"``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import torch

from causalab.neural.shared import streams
from causalab.protocol.registry import (
    ModelInfo,
    model_info_from_hf_config,
    register_model,
)

__all__ = ["NnsightBundle", "load_model"]

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclasses.dataclass(frozen=True)
class NnsightBundle:
    """One loaded nnsight model with everything the executor needs.

    ``model`` is the :class:`TransformersModel`: attribute access on it
    yields envoys mirroring the HF module tree, which is what lets the
    shared site map resolve against it unchanged.
    """

    key: str
    revision: str
    model: Any
    tokenizer: Any
    info: ModelInfo
    #: The device inputs are sent to. nnsight's dispatch places the *model*
    #: itself on first trace; a disagreement surfaces as a loud device
    #: mismatch at the first forward, never as quiet wrong numbers (the same
    #: trade the reference bundle documents).
    device: str
    dtype: str
    quantization: dict[str, Any] | None = None

    @property
    def is_gpt2_family(self) -> bool:
        return hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        )

    @property
    def blocks(self) -> Any:
        """The decoder-layer list (envoys), whichever tree this family uses."""
        return (
            self.model.transformer.h if self.is_gpt2_family else self.model.model.layers
        )

    def stream_at(self, layer: int) -> str:
        """Which mixer stream ``layer`` carries — the shared table's answer
        (:mod:`causalab.neural.shared.streams`), read off the envoy tree."""
        return streams.stream_at(self.blocks, layer, key=self.key)

    def mixer_at(self, layer: int) -> Any:
        """The attention/mixer envoy at ``layer``, whichever stream it is."""
        return streams.mixer_at(self.blocks, layer, key=self.key)

    @property
    def streams(self) -> tuple[str, ...]:
        """``stream_at`` for every layer — the whole tower's shape at a glance."""
        return tuple(self.stream_at(i) for i in range(len(self.blocks)))


@functools.lru_cache(maxsize=4)
def load_model(
    key: str,
    revision: str = "main",
    *,
    dtype: str = "fp32",
    device: str = "cpu",
    attn_implementation: str | None = None,
) -> NnsightBundle:
    """Load (and cache) one nnsight bundle.

    The tokenizer is set to the engines' single padding convention: left
    padding with ``pad = eos`` when the checkpoint ships no pad token — the
    same contract the reference bundle loads under, so both engines encode
    identical batches.

    ``attn_implementation=None`` keeps the checkpoint's own default (D5); the
    executor switches on demand for the traces that need another one. Passing
    one pins it — what the parity suite does to compare like against like.
    """
    from nnsight.modeling.transformers import TransformersModel

    model = TransformersModel(
        key,
        task="text-generation",
        revision=revision,
        dtype=_DTYPES[dtype],
        **(
            {}
            if attn_implementation is None
            else {"attn_implementation": attn_implementation}
        ),
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = model_info_from_hf_config(key, model.config)
    register_model(info)
    return NnsightBundle(
        key=key,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        info=info,
        device=device,
        dtype=dtype,
    )
