"""Tests for the logit lens method (causalab.methods.logit_lens).

The pure-tensor tests validate the projection math and accessors with a stub
module and run anywhere. The gpt2 integration test (marked ``slow``) is the
real correctness check: at the *last* layer, with the final norm applied, the
lens must reproduce the model's own next-token argmax — proving the
final-norm + unembedding wiring is faithful.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.logit_lens import (
    compute_logit_lens,
    get_final_norm,
    project_to_logits,
    save_logit_lens_results,
)

import pytest


# --------------------------------------------------------------------------- #
# Pure-tensor unit tests (no model download)                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_get_final_norm_finds_gpt2_style_path():
    """A transformer.ln_f chain (GPT-2 family) is resolved."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_f = nn.LayerNorm(8)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _Inner()

    norm = get_final_norm(_Model())
    assert isinstance(norm, nn.LayerNorm)


@pytest.mark.unit
def test_get_final_norm_raises_when_absent():
    class _Model(nn.Module):
        pass

    with pytest.raises(ValueError, match="final layer norm"):
        get_final_norm(_Model())


@pytest.mark.numerical_unit
def test_project_to_logits_matches_manual_norm_then_unembed():
    """project_to_logits == final_norm then lm_head, computed in float32."""
    torch.manual_seed(0)
    hidden_size, vocab_size, n = 8, 17, 5
    final_norm = nn.LayerNorm(hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    hidden = torch.randn(n, hidden_size)
    logits = project_to_logits(hidden, final_norm, lm_head, apply_final_norm=True)

    expected = lm_head(final_norm(hidden))
    assert logits.shape == (n, vocab_size)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.numerical_unit
def test_project_to_logits_skips_norm_when_disabled():
    torch.manual_seed(0)
    final_norm = nn.LayerNorm(8)
    lm_head = nn.Linear(8, 11, bias=False)
    hidden = torch.randn(3, 8)

    logits = project_to_logits(hidden, final_norm, lm_head, apply_final_norm=False)
    torch.testing.assert_close(logits, lm_head(hidden), atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# gpt2 integration test (real weights) — the faithful-wiring check            #
# --------------------------------------------------------------------------- #


def _trace(prompt: str) -> CausalTrace:
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": prompt},
    )


@pytest.fixture(scope="module")
def gpt2_pipeline():
    from causalab.neural.pipeline import LMPipeline

    try:
        return LMPipeline("gpt2", max_new_tokens=1, device="cpu")
    except Exception as exc:  # pragma: no cover - offline / no weights
        pytest.skip(f"Could not load gpt2: {exc}")


@pytest.mark.numerical_unit
def test_compute_logit_lens_last_layer_matches_model(gpt2_pipeline):
    from causalab.neural.token_positions import TokenPosition, get_last_token_index

    pipeline = gpt2_pipeline
    prompts = [
        "The capital of France is",
        "Two plus two equals",
        "The opposite of hot is",
    ]
    dataset = [{"input": _trace(p)} for p in prompts]

    last_layer = pipeline.model.config.num_hidden_layers - 1
    last_pos = TokenPosition(
        indexer=lambda inp: get_last_token_index(inp, pipeline),
        pipeline=pipeline,
        id="last",
    )

    top_k = 5
    # compute_logit_lens collects through the position-corrected forward
    # (collect.py supplies left-pad position_ids), so the lens reads gpt2's
    # activations at the correct positions. The ground-truth forward below must do
    # the same (ensure_position_ids) to compare like-for-like — otherwise it would
    # number positions from the pad tokens and disagree on this left-padded batch.
    result = compute_logit_lens(
        dataset,
        pipeline,
        layers=[last_layer],
        token_positions=[last_pos],
        top_k=top_k,
        batch_size=len(prompts),
    )

    key = (last_layer, "last")
    assert key in result["top_k_by_unit"]
    payload = result["top_k_by_unit"][key]
    assert payload["token_ids"].shape == (len(prompts), top_k)
    assert payload["probs"].shape == (len(prompts), top_k)
    assert len(payload["tokens"]) == len(prompts)
    # Probabilities are a valid descending top-k slice.
    assert torch.all(payload["probs"] >= 0)
    assert torch.all(payload["probs"][:, :-1] >= payload["probs"][:, 1:] - 1e-6)

    # Ground truth: the model's own next-token argmax over the SAME batch,
    # position-corrected to match how the lens collected its activations.
    from causalab.neural.pipeline import ensure_position_ids

    lens_top1 = payload["token_ids"][:, 0]
    loaded = ensure_position_ids(pipeline.load([_trace(p) for p in prompts]))
    with torch.no_grad():
        out = pipeline.model(**loaded)
    model_argmax = out.logits[:, -1].argmax(dim=-1)
    for i, prompt in enumerate(prompts):
        assert lens_top1[i].item() == model_argmax[i].item(), (
            f"Logit lens top-1 at the last layer should equal the model's "
            f"next-token argmax for prompt {prompt!r}: "
            f"got {lens_top1[i].item()} vs {model_argmax[i].item()}"
        )


@pytest.mark.numerical_unit
def test_compute_logit_lens_target_track_and_save(gpt2_pipeline, tmp_path):
    from causalab.neural.token_positions import TokenPosition, get_last_token_index

    pipeline = gpt2_pipeline
    dataset = [{"input": _trace(p)} for p in ["Paris is in", "Rome is in"]]
    last_pos = TokenPosition(
        indexer=lambda inp: get_last_token_index(inp, pipeline),
        pipeline=pipeline,
        id="last",
    )
    layers = [0, pipeline.model.config.num_hidden_layers - 1]

    # Track the probability mass on a couple of arbitrary answer tokens.
    answer_ids = (
        pipeline.tokenizer.encode(" France")[:1]
        + pipeline.tokenizer.encode(" Italy")[:1]
    )

    result = compute_logit_lens(
        dataset,
        pipeline,
        layers=layers,
        token_positions=[last_pos],
        top_k=3,
        batch_size=2,
        target_token_ids=answer_ids,
    )

    assert result["target_track"] is not None
    for layer in layers:
        cell = result["target_track"][(layer, "last")]
        assert cell["answer_mass"].shape == (len(dataset),)
        assert 0.0 <= cell["answer_mass_mean"] <= 1.0

    # Method metadata is intrinsic-only — run-level tagging (experiment_type,
    # task, model, seed) is the analysis layer's job (docs/CODEBASE.md inv. 4).
    assert "experiment_type" not in result["metadata"]
    assert result["metadata"]["num_samples"] == len(dataset)
    assert result["metadata"]["vocab_size"] > 0

    paths = save_logit_lens_results(result, str(tmp_path))
    import os

    assert os.path.exists(paths["metadata_path"])
    assert os.path.isdir(paths["top_k_dir"])
    assert os.path.isdir(paths["target_track_dir"])
    # One safetensors + one json per (layer, position) cell in top_k/.
    st = [f for f in os.listdir(paths["top_k_dir"]) if f.endswith(".safetensors")]
    js = [f for f in os.listdir(paths["top_k_dir"]) if f.endswith(".json")]
    assert len(st) == len(layers)
    assert len(js) == len(layers)
