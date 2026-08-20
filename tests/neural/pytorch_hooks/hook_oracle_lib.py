"""Backbone-independent PyTorch-hook oracles for causalab's intervention engine.

causalab's intervention engine is currently backed by ``pyvene`` (see
``causalab/neural/activations/``). The helpers here re-implement the *ground
truth* of every intervention — collect, interchange, replace, steer, noise,
mask, interpolation, cross-model patching, two-pass path patching — using only
raw ``torch.nn.Module.register_forward_hook`` / ``register_forward_pre_hook``.
They never touch pyvene.

A test built on these helpers asserts a behavioural contract on causalab's
*public wrappers* (``run_interchange_interventions``, ``collect_features``,
``run_steering_interventions``, …), not on pyvene's internals. So the contract
survives a backbone swap: when pyvene is replaced by nnsight (GH #380), the same
oracle tests re-run unchanged and verify the new backbone reproduces the same
activations and logits. The pyvene→hook→test coverage map lives in
``docs/PYVENE_HOOK_COVERAGE.md``.

Design notes
------------
* Oracles run on the tiny-random CPU pipelines (``tiny_pipeline`` /
  ``tiny_gpt2_pipeline`` from ``tests/neural/conftest.py``); no GPU, no coherent
  English needed — these are equivalence/invariance contracts, not value pins.
* The hook resolvers below mirror pyvene's llama/gpt2 component→module table:
  ``block_output``→``layers[L]`` output, ``block_input``→``layers[L]`` input,
  ``mlp_output``→``layers[L].mlp`` output, ``attention_output``→
  ``layers[L].self_attn`` output, and ``head_attention_value_output`` head ``H``→
  the ``[H*d_head:(H+1)*d_head]`` column slice of ``o_proj``'s input.
* ``run_with_writes`` / ``capture_with_writes`` install several hooks in one
  forward. PyTorch fires hooks in module-execution (forward) order, so a write on
  an upstream module is visible to a capture/write on a downstream module within
  the same pass — exactly the firing-order contract causal tracing and two-pass
  path patching rely on.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Callable

import torch

from causalab.causal.trace import CausalTrace, Mechanism

# The Plan-era LMPipeline is gone; every helper needs only `.hf_model`, so a
# one-field shim satisfies the parameter (tests/neural/pytorch_hooks/conftest.py).
LMPipeline = Any

# Write/capture hooks fire on a module's input ("in", a forward_pre_hook) or its
# output ("out", a forward_hook).
HookKind = str  # Literal["in", "out"]
WriteFn = Callable[[torch.Tensor], None]  # mutates the hooked activation in place
WriteSpec = tuple[torch.nn.Module, HookKind, WriteFn]


# --------------------------------------------------------------------------- #
#  Inputs — lightweight traces and counterfactual examples                    #
# --------------------------------------------------------------------------- #
def make_trace(text: str) -> CausalTrace:
    """Build a single-variable :class:`CausalTrace` from a raw string.

    The tiny-pipeline tests need lightweight inputs that mimic the
    ``raw_input``-only traces produced by simple task harnesses. Inline
    construction keeps the tests readable without coupling them to any
    specific task's :class:`CausalModel`.
    """
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


def cf_example(base_text: str, *cf_texts: str) -> dict[str, Any]:
    """A ``CounterfactualExample``-shaped dict: one base, N counterfactuals."""
    return {
        "input": make_trace(base_text),
        "counterfactual_inputs": [make_trace(t) for t in cf_texts],
    }


# --------------------------------------------------------------------------- #
#  Featurizer test-doubles — non-identity, last-dim, broadcastable            #
# --------------------------------------------------------------------------- #
class DiagFeaturizerModule(torch.nn.Module):
    """A dimension-sensitive (non-identity) featurizer: scales each of the ``d``
    feature dims by a fixed per-feature weight. The ``assert`` makes it fail
    loudly if it is ever handed a folded ``num_pos*d`` vector instead of a
    ``(..., d)`` per-position slice — so a passing multi-token intervention proves
    the span is applied *per position* (``keep_last_dim=True``), not flattened
    into the feature dim. Mirrors how a real DAS/rotation featurizer (weights
    sized for one position's width) would behave, without the training."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(0.5, 1.5, d), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert x.shape[-1] == self.weight.shape[0], (
            f"featurizer received last dim {x.shape[-1]}, expected "
            f"{self.weight.shape[0]} — span was folded into the feature dim"
        )
        return x * self.weight, None


class DiagInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`DiagFeaturizerModule`."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, error: None) -> torch.Tensor:
        return x / self.weight


class RotateFeaturizerModule(torch.nn.Module):
    """Featurizer that rotates into an orthonormal basis ``R`` (``d x d``).

    A rotation *mixes* dimensions, so swapping a feature subspace in this basis
    is a genuinely different operation than swapping raw activation dims — which
    is what makes it a strong oracle target. The ``assert`` fails loudly if the
    module is ever handed a folded ``num_pos*d`` vector instead of a per-position
    ``(..., d)`` slice."""

    def __init__(self, R: torch.Tensor) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert x.shape[-1] == self.R.shape[0], (
            f"featurizer received last dim {x.shape[-1]}, expected "
            f"{self.R.shape[0]} — span was folded into the feature dim"
        )
        return x @ self.R, None


class RotateInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`RotateFeaturizerModule` (``R`` is orthonormal)."""

    def __init__(self, R: torch.Tensor) -> None:
        super().__init__()
        self.R = torch.nn.Parameter(R, requires_grad=False)

    def forward(self, x: torch.Tensor, error: None) -> torch.Tensor:
        return x @ self.R.T


def random_rotation(d: int, *, seed: int = 0) -> torch.Tensor:
    """A reproducible ``d x d`` orthonormal matrix (Q of a QR of a random matrix)."""
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g))[0]


# --------------------------------------------------------------------------- #
#  Module resolvers — the modules the backbone taps, addressed by hand         #
#                                                                              #
#  Two architecture families are covered so the oracle stays backbone- AND     #
#  model-family-independent (GH #380):                                         #
#    * Llama-family (Llama / Qwen3 / Gemma / Mistral): ``model.model.layers[L]``#
#      with separate ``self_attn.{q,k,v,o}_proj`` and ``.mlp``;                #
#    * GPT-2: ``model.transformer.h[L]`` with a fused ``attn.c_attn`` (QKV) and #
#      ``attn.c_proj`` as the output projection.                               #
#  The resolvers dispatch on the module tree; everything above (capture / write #
#  / multi-hook) is written against these resolvers and is therefore family-   #
#  agnostic. ``config.hidden_size`` / ``num_attention_heads`` resolve on GPT-2 #
#  too via ``GPT2Config.attribute_map`` (``n_embd`` / ``n_head``).             #
# --------------------------------------------------------------------------- #
def _is_gpt2(pipeline: LMPipeline) -> bool:
    """True for the GPT-2 family (``transformer.h`` block list, fused QKV)."""
    return hasattr(pipeline.hf_model, "transformer") and hasattr(
        pipeline.hf_model.transformer, "h"
    )


def decoder_block(pipeline: LMPipeline, layer: int) -> torch.nn.Module:
    """The decoder block whose forward output is the residual stream after
    ``layer`` (matches ``ResidualStream(layer, target_output=True)``)."""
    if _is_gpt2(pipeline):
        return pipeline.hf_model.transformer.h[layer]
    return pipeline.hf_model.model.layers[layer]


def _attn_module(pipeline: LMPipeline, layer: int) -> torch.nn.Module:
    """The self-attention submodule of ``layer`` (``attn`` on GPT-2,
    ``self_attn`` on Llama-family)."""
    blk = decoder_block(pipeline, layer)
    return blk.attn if _is_gpt2(pipeline) else blk.self_attn


def o_proj(pipeline: LMPipeline, layer: int) -> torch.nn.Module:
    """The attention output projection. Its *input* is the concatenated per-head
    attention output — the ``head_attention_value_output`` surface (``c_proj`` on
    GPT-2, ``o_proj`` on Llama-family)."""
    attn = _attn_module(pipeline, layer)
    return attn.c_proj if _is_gpt2(pipeline) else attn.o_proj


def embed_module(pipeline: LMPipeline) -> torch.nn.Module:
    """The token-embedding module whose *output* is the ``embeddings`` component
    (``wte`` on GPT-2, ``embed_tokens`` on Llama-family) — what nnterp exposes as
    the settable ``token_embeddings``."""
    if _is_gpt2(pipeline):
        return pipeline.hf_model.transformer.wte
    return pipeline.hf_model.model.embed_tokens


def head_dim(pipeline: LMPipeline) -> int:
    """Per-head width. Honours an explicit ``config.head_dim`` (e.g. Qwen3, which
    decouples it from ``hidden / n_head``); otherwise ``hidden / n_head``."""
    cfg = pipeline.hf_model.config
    return getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )


def head_slice(pipeline: LMPipeline, head: int) -> slice:
    """The ``[head*d_head:(head+1)*d_head]`` column slice of ``o_proj``'s input."""
    d = head_dim(pipeline)
    return slice(head * d, (head + 1) * d)


def component_module(
    pipeline: LMPipeline, layer: int, component: str
) -> tuple[torch.nn.Module, HookKind]:
    """Map a causalab ``component_type`` to ``(module, kind)`` — the module the
    backbone taps and whether it reads/writes the module's input or output.
    Mirrors the backbone's component→module table for both families."""
    if component == "embeddings":
        # Layer-less: the token-embedding output feeding the first block.
        return embed_module(pipeline), "out"
    blk = decoder_block(pipeline, layer)
    attn = _attn_module(pipeline, layer)
    if component == "mlp_activation":
        # The intermediate activation feeding the down-projection. Architecture-
        # specific: SwiGLU Llama exposes the `act_fn` *output* (act(gate), before
        # the up-proj gate-multiply), GPT-2 exposes the `c_proj` *input*.
        if _is_gpt2(pipeline):
            return blk.mlp.c_proj, "in"
        return blk.mlp.act_fn, "out"
    table: dict[str, tuple[torch.nn.Module, HookKind]] = {
        "block_output": (blk, "out"),
        "block_input": (blk, "in"),
        "mlp_output": (blk.mlp, "out"),
        "mlp_input": (blk.mlp, "in"),
        "attention_output": (attn, "out"),
    }
    return table[component]


def hidden_of(out: Any) -> torch.Tensor:
    """A decoder block / attention module returns a tuple ``(hidden, ...)``; an
    MLP / projection returns the tensor directly. Normalise to the tensor."""
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
#  Capture — read an activation with our own hook, no pyvene                   #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def layer_fire_counts(pipeline: LMPipeline) -> Iterator[list[int]]:
    """Raw-hook fire-counters on every decoder block, for the enclosed runs.

    ``counts[L]`` is how many times block ``L``'s forward *completed* while
    the context was open — the ground truth for the early-stop contract
    (``tracer.stop()``, CAP6 #459): a stopped forward leaves every block past
    the deepest tap at zero. The block that carries the deepest tap itself
    still counts (its forward completes before nnsight's output hook raises
    the stop); only the blocks *after* it never run. The counters register
    before any trace opens, so nnsight's one-shot hooks (inserted in
    mediator-index order behind pre-existing hooks) never shadow them.
    """
    blocks = (
        pipeline.hf_model.transformer.h
        if _is_gpt2(pipeline)
        else pipeline.hf_model.model.layers
    )
    counts = [0] * len(blocks)
    handles = []
    for i, block in enumerate(blocks):

        def hook(_m, _i, _o, i=i):
            counts[i] += 1

        handles.append(block.register_forward_hook(hook))
    try:
        yield counts
    finally:
        for h in handles:
            h.remove()


def capture_residual(pipeline: LMPipeline, layer: int, inputs: Mapping) -> torch.Tensor:
    """Residual stream after ``layer`` for ``inputs``, grabbed via our own
    forward hook — no pyvene involved."""
    grabbed: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        grabbed["resid"] = hidden_of(out).detach().clone()

    handle = decoder_block(pipeline, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            pipeline.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
    finally:
        handle.remove()
    return grabbed["resid"]


def capture_component(
    pipeline: LMPipeline, module: torch.nn.Module, kind: HookKind, inputs: Mapping
) -> torch.Tensor:
    """Activation at ``module`` (its output if ``kind=='out'``, else its input)
    for ``inputs`` — grabbed with our own hook, no pyvene."""
    grabbed: dict[str, torch.Tensor] = {}

    if kind == "out":

        def out_hook(_m, _i, out):
            grabbed["v"] = hidden_of(out).detach().clone()

        handle = module.register_forward_hook(out_hook)
    else:

        def pre_hook(_m, args):
            grabbed["v"] = args[0].detach().clone()

        handle = module.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            pipeline.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
    finally:
        handle.remove()
    return grabbed["v"]


def capture_head_value(
    pipeline: LMPipeline, layer: int, head: int, inputs: Mapping
) -> torch.Tensor:
    """Per-head value output (``o_proj`` input, head-``head`` slice) at all positions.

    ``head_attention_value_output`` for head ``H`` is the
    ``[H*d_head:(H+1)*d_head]`` column slice of the *input* to ``o_proj`` (the
    per-head attention output, concatenated, before the output projection)."""
    grabbed: dict[str, torch.Tensor] = {}

    def pre(_module, args):
        grabbed["x"] = args[0].detach().clone()

    handle = o_proj(pipeline, layer).register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            pipeline.hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
    finally:
        handle.remove()
    return grabbed["x"][:, :, head_slice(pipeline, head)]


# --------------------------------------------------------------------------- #
#  Write + run — hand-rolled interventions, return last-position logits         #
# --------------------------------------------------------------------------- #
def next_token_logits(
    pipeline: LMPipeline,
    base_inputs: Mapping,
    layer: int | None = None,
    positions: list[int] | None = None,
    patch_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Last-position logits for ``base_inputs``. When ``patch_values`` is given,
    a forward hook overwrites the residual after ``layer`` at ``positions`` with
    it (raw activation space, shape ``[batch, len(positions), d]``) — a
    hand-rolled interchange. With no patch, returns the clean logits."""
    handle = None
    if patch_values is not None:
        assert positions is not None and layer is not None

        def hook(_module, _inp, out):
            is_tuple = isinstance(out, tuple)
            hidden = (out[0] if is_tuple else out).clone()
            for i, p in enumerate(positions):
                hidden[:, p, :] = patch_values[:, i, :]
            return (hidden, *out[1:]) if is_tuple else hidden

        handle = decoder_block(pipeline, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = pipeline.hf_model(
                input_ids=base_inputs["input_ids"],
                attention_mask=base_inputs["attention_mask"],
            ).logits[:, -1, :]
    finally:
        if handle is not None:
            handle.remove()
    # On CPU so the equivalence comparison holds on GPU too — the causalab
    # wrappers move their scores to CPU; this is a no-op when the model is on CPU.
    return logits.cpu()


def component_written_logits(
    pipeline: LMPipeline,
    base_inputs: Mapping,
    module: torch.nn.Module,
    kind: HookKind,
    write: WriteFn,
) -> torch.Tensor:
    """Next-token logits after running the base with a hook whose ``write(hidden)``
    mutates the component's activation in place (the hand-rolled intervention)."""
    return run_with_writes(pipeline, base_inputs, [(module, kind, write)])


def head_patched_next_logits(
    pipeline: LMPipeline,
    layer: int,
    head: int,
    positions: list[int],
    base_inputs: Mapping,
    patch_slice: torch.Tensor,
) -> torch.Tensor:
    """Base run with a forward_pre_hook overwriting ``o_proj``'s head-``head`` slice
    at ``positions`` with ``patch_slice`` (``[batch, len(positions), d_head]``)."""
    sl = head_slice(pipeline, head)

    def write(x: torch.Tensor) -> None:
        for i, p in enumerate(positions):
            x[:, p, sl] = patch_slice[:, i, :]

    return run_with_writes(
        pipeline, base_inputs, [(o_proj(pipeline, layer), "in", write)]
    )


# --------------------------------------------------------------------------- #
#  Multi-hook — several writes / a capture in one forward (firing order)        #
# --------------------------------------------------------------------------- #
def _install(module: torch.nn.Module, kind: HookKind, write: WriteFn):
    """Register an in-place ``write`` on ``module``'s input/output; return handle."""
    if kind == "out":

        def out_hook(_m, _i, out):
            hidden = hidden_of(out).clone()
            write(hidden)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

        return module.register_forward_hook(out_hook)

    def pre_hook(_m, args):
        x = args[0].clone()
        write(x)
        return (x, *args[1:])

    return module.register_forward_pre_hook(pre_hook)


def run_with_writes(
    pipeline: LMPipeline, base_inputs: Mapping, writes: list[WriteSpec]
) -> torch.Tensor:
    """Last-position logits after applying every ``(module, kind, write)`` in one
    forward. PyTorch fires the hooks in module-execution order, so an upstream
    write is visible to a downstream write — the forward-order contract causal
    tracing and two-pass path patching depend on."""
    handles = [_install(m, kind, write) for (m, kind, write) in writes]
    try:
        with torch.no_grad():
            logits = pipeline.hf_model(
                input_ids=base_inputs["input_ids"],
                attention_mask=base_inputs["attention_mask"],
            ).logits[:, -1, :]
    finally:
        for h in handles:
            h.remove()
    # On CPU so the equivalence comparison holds on GPU too — the causalab
    # wrappers move their scores to CPU; this is a no-op when the model is on CPU.
    return logits.cpu()


def capture_with_writes(
    pipeline: LMPipeline,
    base_inputs: Mapping,
    capture_module: torch.nn.Module,
    capture_kind: HookKind,
    writes: list[WriteSpec],
) -> torch.Tensor:
    """Capture ``capture_module``'s activation in a single forward while ``writes``
    are active. Because hooks fire in forward order, the captured value reflects
    every *upstream* write already applied — this is how two-pass path patching's
    PASS 1 reads a receiver's value under an upstream sender+restorer
    intervention (no pyvene)."""
    grabbed: dict[str, torch.Tensor] = {}

    if capture_kind == "out":

        def cap(_m, _i, out):
            grabbed["v"] = hidden_of(out).detach().clone()

        cap_handle = capture_module.register_forward_hook(cap)
    else:

        def cap_pre(_m, args):
            grabbed["v"] = args[0].detach().clone()

        cap_handle = capture_module.register_forward_pre_hook(cap_pre)

    write_handles = [_install(m, kind, write) for (m, kind, write) in writes]
    try:
        with torch.no_grad():
            pipeline.hf_model(
                input_ids=base_inputs["input_ids"],
                attention_mask=base_inputs["attention_mask"],
            )
    finally:
        cap_handle.remove()
        for h in write_handles:
            h.remove()
    return grabbed["v"]
