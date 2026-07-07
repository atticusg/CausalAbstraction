"""Regression tests for the transformers ``attn_implementation`` semantics that
causalab's ``LMPipeline(..., attn_implementation=...)`` pass-through relies on.

Verified empirically on transformers 4.57.1 / torch 2.9.0 (CPU, float32).
Context: causalab used to select eager attention by flipping
``config._attn_implementation`` *after* ``from_pretrained`` (the legacy
``eager_attn`` flag). On 4.57.1 that flip happens to be self-consistent for
decoder-only models because both the kernel dispatch and the causal-mask
builder read the live config at forward time — but the correctness is an
accident of transformers version and model family:

* ``GPT2Model.__init__`` stores a frozen copy of the implementation choice
  (``self._attn_implementation``); cross-attention encoder-mask preparation
  reads the frozen copy, so a flipped model runs eager kernels with
  sdpa-format encoder masks (a mixed state).
* On older transformers (≤ 4.4x) attention module *classes* were chosen at
  init, so the flip changed nothing at all.

These tests pin the 4.57-line behavior so an upgrade that changes any of it
fails loudly. All models are tiny-random; no network, CPU-only.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    LlamaConfig,
    PreTrainedModel,
)

# ---------------------------------------------------------------------------
# tiny-random model factories (no network)
# ---------------------------------------------------------------------------


def _tiny_gpt2_config(**overrides: object) -> GPT2Config:
    kwargs = dict(
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=64,
        vocab_size=257,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    kwargs.update(overrides)
    return GPT2Config(**kwargs)


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=257,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )


@pytest.fixture(scope="module")
def tiny_model_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("tiny_attn_models")
    dirs = {}
    for name, config in (
        ("gpt2", _tiny_gpt2_config()),
        ("llama", _tiny_llama_config()),
        ("gpt2_cross", _tiny_gpt2_config(add_cross_attention=True)),
    ):
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        path = root / name
        model.save_pretrained(path)
        dirs[name] = str(path)
    return dirs


def _load(path: str, **kwargs: object) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, **kwargs)
    model.eval()
    return model


def _flip_to_eager(model: PreTrainedModel) -> PreTrainedModel:
    """The legacy post-load flip (what the removed ``eager_attn`` flag did)."""
    model.config._attn_implementation = "eager"
    return model


def _attn_modules(model: PreTrainedModel) -> list[torch.nn.Module]:
    if hasattr(model, "transformer"):  # GPT-2
        return [block.attn for block in model.transformer.h]
    return [layer.self_attn for layer in model.model.layers]  # Llama


def _batches(pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    unpadded = torch.randint(1, 257, (2, 8))
    padded = unpadded.clone()
    padded[0, :3] = pad_id
    pad_mask = torch.ones(2, 8, dtype=torch.long)
    pad_mask[0, :3] = 0
    return unpadded, padded, pad_mask


def _capture_forward_param(
    modules: list[torch.nn.Module], param_name: str
) -> tuple[list[object], list[torch.utils.hooks.RemovableHandle]]:
    """Capture the named forward parameter on each module via pre-hooks."""
    captured: list = []
    handles = []

    def hook(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        captured.append(bound.arguments.get(param_name, "<param-missing>"))

    for m in modules:
        handles.append(m.register_forward_pre_hook(hook, with_kwargs=True))
    return captured, handles


def _force_param_none(
    modules: list[torch.nn.Module], param_name: str
) -> list[torch.utils.hooks.RemovableHandle]:
    """Force the named forward parameter to None (simulates an sdpa-produced
    None reaching a kernel that does not expect it)."""
    handles = []

    def hook(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ):
        sig = inspect.signature(module.forward)
        bound = sig.bind(*args, **kwargs)
        if param_name in bound.arguments:
            bound.arguments[param_name] = None
        new_kwargs = {}
        for name, value in bound.arguments.items():
            if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                new_kwargs.update(value)
            else:
                new_kwargs[name] = value
        return (), new_kwargs

    for m in modules:
        handles.append(m.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def _remove(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


@torch.no_grad()
def _logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits


# ---------------------------------------------------------------------------
# C1: default resolution is sdpa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_default_resolution_is_sdpa(tiny_model_dirs: dict[str, str], name: str) -> None:
    model = _load(tiny_model_dirs[name])
    assert model.config._attn_implementation == "sdpa"


# ---------------------------------------------------------------------------
# C2: on this transformers version, the legacy post-load flip is
# self-consistent for decoder-only models (matches load-time eager exactly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_postload_flip_matches_load_time_eager_logits(
    tiny_model_dirs: dict[str, str], name: str
) -> None:
    m_flip = _flip_to_eager(_load(tiny_model_dirs[name]))
    m_eager = _load(tiny_model_dirs[name], attn_implementation="eager")
    unpadded, padded, pad_mask = _batches()

    assert torch.equal(_logits(m_flip, unpadded), _logits(m_eager, unpadded))
    assert torch.equal(
        _logits(m_flip, padded, pad_mask), _logits(m_eager, padded, pad_mask)
    )


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_postload_flip_delivers_float_additive_mask(
    tiny_model_dirs: dict[str, str], name: str
) -> None:
    """The mask builder reads the live config: after the flip, attention
    layers receive an eager-format float additive mask, not sdpa's None/bool."""
    m_flip = _flip_to_eager(_load(tiny_model_dirs[name]))
    unpadded, padded, pad_mask = _batches()

    for ids, mask in ((unpadded, None), (padded, pad_mask)):
        captured, handles = _capture_forward_param(
            _attn_modules(m_flip), "attention_mask"
        )
        _logits(m_flip, ids, mask)
        _remove(handles)
        got = captured[0]
        assert isinstance(got, torch.Tensor)
        assert got.dtype.is_floating_point
        assert got.max().item() == 0.0
        assert got.min().item() < 0.0


# ---------------------------------------------------------------------------
# C3: the frozen per-module copy — the mixed state the flip creates
# ---------------------------------------------------------------------------


def test_gpt2_frozen_copy_diverges_from_live_config_after_flip(
    tiny_model_dirs: dict[str, str],
) -> None:
    m_flip = _flip_to_eager(_load(tiny_model_dirs["gpt2"]))
    assert m_flip.config._attn_implementation == "eager"
    # GPT2Model.__init__ froze the load-time choice; the flip does not reach it.
    assert m_flip.transformer._attn_implementation == "sdpa"


def test_gpt2_cross_attention_encoder_mask_follows_frozen_copy(
    tiny_model_dirs: dict[str, str],
) -> None:
    """Where the frozen copy is live: encoder-mask preparation. A flipped
    cross-attention model prepares the encoder mask down the sdpa branch while
    kernels run eager — unlike a load-time-eager model, which gets the
    eager-format additive mask."""
    unpadded, _, _ = _batches()
    torch.manual_seed(11)
    enc_hidden = torch.randn(2, 5, 32)
    enc_mask_padded = torch.ones(2, 5, dtype=torch.long)
    enc_mask_padded[0, 3:] = 0

    def encoder_mask_seen(
        model: PreTrainedModel, enc_mask: torch.Tensor | None
    ) -> object:
        cross = [b.crossattention for b in model.transformer.h]
        captured, handles = _capture_forward_param(cross, "encoder_attention_mask")
        _logits(
            model,
            unpadded,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
        )
        _remove(handles)
        return captured[0]

    m_flip = _flip_to_eager(_load(tiny_model_dirs["gpt2_cross"]))
    m_eager = _load(tiny_model_dirs["gpt2_cross"], attn_implementation="eager")

    # Load-time eager: invert_attention_mask -> float additive, both cases.
    for enc_mask in (None, enc_mask_padded):
        got = encoder_mask_seen(m_eager, enc_mask)
        assert isinstance(got, torch.Tensor) and got.dtype.is_floating_point

    # Flipped: the frozen-"sdpa" branch elides the all-ones encoder mask
    # entirely — the eager cross-attention kernel receives None.
    assert encoder_mask_seen(m_flip, None) is None

    # Padded encoder: both branches materialize a float mask, but in the
    # sdpa-prepared shape (batch, 1, q_len, kv_len) vs the eager-inverted
    # (batch, 1, 1, kv_len) — the formats diverge even when values agree.
    flip_mask = encoder_mask_seen(m_flip, enc_mask_padded)
    eager_mask = encoder_mask_seen(m_eager, enc_mask_padded)
    assert flip_mask.shape != eager_mask.shape


# ---------------------------------------------------------------------------
# C4: sdpa mask formats — None (is_causal deferral) unpadded, boolean padded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_sdpa_mask_is_none_for_unpadded_full_causal_batch(
    tiny_model_dirs: dict[str, str], name: str
) -> None:
    m_sdpa = _load(tiny_model_dirs[name])
    assert m_sdpa.config._attn_implementation == "sdpa"
    unpadded, _, _ = _batches()

    captured, handles = _capture_forward_param(_attn_modules(m_sdpa), "attention_mask")
    _logits(m_sdpa, unpadded)
    _remove(handles)
    assert captured[0] is None


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_sdpa_mask_is_boolean_for_left_padded_batch(
    tiny_model_dirs: dict[str, str], name: str
) -> None:
    m_sdpa = _load(tiny_model_dirs[name])
    _, padded, pad_mask = _batches()

    captured, handles = _capture_forward_param(_attn_modules(m_sdpa), "attention_mask")
    _logits(m_sdpa, padded, pad_mask)
    _remove(handles)
    got = captured[0]
    assert isinstance(got, torch.Tensor)
    assert got.dtype is torch.bool


# ---------------------------------------------------------------------------
# C5: eager mask is always a materialized float additive mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["gpt2", "llama"])
def test_eager_mask_is_always_float_additive(
    tiny_model_dirs: dict[str, str], name: str
) -> None:
    m_eager = _load(tiny_model_dirs[name], attn_implementation="eager")
    unpadded, padded, pad_mask = _batches()

    for ids, mask in ((unpadded, None), (padded, pad_mask)):
        captured, handles = _capture_forward_param(
            _attn_modules(m_eager), "attention_mask"
        )
        _logits(m_eager, ids, mask)
        _remove(handles)
        got = captured[0]
        assert isinstance(got, torch.Tensor)
        assert got.dtype.is_floating_point
        assert got.max().item() == 0.0
        assert got.min().item() <= torch.finfo(torch.float32).min / 2


# ---------------------------------------------------------------------------
# C8: GPT-2's eager kernel stays causal even if the mask goes missing
# (its registered bias buffer applies causal masking unconditionally)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _future_token_influence(model: PreTrainedModel, force_none: bool) -> float:
    """Change the final token; max |logit diff| at earlier positions."""
    torch.manual_seed(13)
    ids_a = torch.randint(1, 257, (1, 8))
    ids_b = ids_a.clone()
    ids_b[0, -1] = (ids_a[0, -1] + 1) % 256 + 1

    handles = (
        _force_param_none(_attn_modules(model), "attention_mask") if force_none else []
    )
    la = model(input_ids=ids_a).logits
    lb = model(input_ids=ids_b).logits
    _remove(handles)
    return (la[:, :-1] - lb[:, :-1]).abs().max().item()


def test_gpt2_eager_kernel_stays_causal_when_mask_is_none(
    tiny_model_dirs: dict[str, str],
) -> None:
    m_gpt2 = _load(tiny_model_dirs["gpt2"], attn_implementation="eager")
    assert _future_token_influence(m_gpt2, force_none=True) == 0.0


def test_llama_eager_kernel_is_not_causal_when_mask_is_none(
    tiny_model_dirs: dict[str, str],
) -> None:
    """The counterpoint that makes C8 meaningful: an eager kernel with no
    bias-buffer fallback (Llama) goes non-causal if the mask goes missing.
    This cannot happen on the normal path (mask construction and kernel
    dispatch read the same live config); it is the failure mode a
    mixed-state model risks."""
    m_llama = _load(tiny_model_dirs["llama"], attn_implementation="eager")
    assert _future_token_influence(m_llama, force_none=False) == 0.0  # control
    assert _future_token_influence(m_llama, force_none=True) > 0.0
