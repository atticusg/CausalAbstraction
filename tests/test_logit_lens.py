"""Tests for the logit lens method (causalab.methods.logit_lens).

Two layers of checks:

* **Pure-tensor** — the projection math (:func:`project_to_logits`) with a stub
  module, runs anywhere.
* **nnterp-routed** — on the tiny-random Llama, that
  :func:`resolve_final_norm_and_unembed` / :func:`project_on_vocab` find and
  apply the *same* final-norm + unembedding the removed ``_FINAL_NORM_PATHS``
  probe did (parity), and that the nnterp trace wrappers (:func:`logit_lens`,
  :func:`get_topk_closest_tokens`) return valid distributions — the last
  logit-lens layer reproduces the model's own next-token distribution.

The gpt2 integration test (real weights) is the end-to-end faithful-wiring check
for :func:`compute_logit_lens`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.logit_lens import (
    compute_logit_lens,
    get_topk_closest_tokens,
    logit_lens,
    project_on_vocab,
    project_to_logits,
    resolve_final_norm_and_unembed,
    save_logit_lens_results,
)

import pytest


# --------------------------------------------------------------------------- #
# Pure-tensor unit tests (no model download)                                  #
# --------------------------------------------------------------------------- #


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
# nnterp-routed projection on the tiny-random Llama (no big download)         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_pipeline():
    from causalab.neural.pipeline import LMPipeline
    from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

    try:
        return LMPipeline(
            TINY_RANDOM_MODEL_NAME, max_new_tokens=1, device="cpu", dtype="float32"
        )
    except Exception as exc:  # pragma: no cover - offline / no weights
        pytest.skip(f"Could not load tiny-random: {exc}")


@pytest.mark.numerical_unit
def test_project_on_vocab_parity_with_bespoke_final_norm_path(tiny_pipeline):
    """nnterp resolves the *same* modules the removed _FINAL_NORM_PATHS did.

    The deleted path probed ``("model", "norm")`` for the final norm and
    ``get_output_embeddings()`` for the unembedding. nnterp's standardized
    ``ln_final`` / ``lm_head`` must be those exact modules, so ``project_on_vocab``
    reproduces the old projection bit-for-bit (project_to_logits is unchanged).
    """
    pipe = tiny_pipeline
    hf = pipe.hf_model

    norm, head = resolve_final_norm_and_unembed(pipe)
    # Identity, not just numerical equality: nnterp found the same objects.
    assert norm is hf.model.norm
    assert head is hf.get_output_embeddings()

    hidden = torch.randn(4, pipe.model.hidden_size, dtype=torch.float32)
    got = project_on_vocab(pipe, hidden)
    expected = project_to_logits(
        hidden, hf.model.norm, hf.get_output_embeddings(), apply_final_norm=True
    )
    assert got.shape == (4, pipe.model.config.vocab_size)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    # apply_final_norm=False path also routes through the same unembedding.
    raw = project_on_vocab(pipe, hidden, apply_final_norm=False)
    raw_expected = project_to_logits(
        hidden, hf.model.norm, hf.get_output_embeddings(), apply_final_norm=False
    )
    torch.testing.assert_close(raw, raw_expected, atol=1e-5, rtol=1e-4)


@pytest.mark.numerical_unit
def test_logit_lens_wrapper_last_layer_matches_model(tiny_pipeline):
    """nnterp logit_lens: valid per-layer distributions; last layer == model."""
    pipe = tiny_pipeline
    prompts = ["The cat sat on the"]

    probs = logit_lens(pipe, prompts)
    n_layers = pipe.model.config.num_hidden_layers
    vocab = pipe.model.config.vocab_size
    assert probs.shape == (len(prompts), n_layers, vocab)
    # Valid probability distributions per (prompt, layer).
    assert torch.all(probs >= 0)
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(len(prompts), n_layers), atol=1e-4, rtol=1e-4
    )

    # The last logit-lens layer projects the final residual through the same
    # final norm + unembedding as the model itself, so it reproduces the model's
    # own next-token distribution.
    enc = pipe.tokenizer(prompts, return_tensors="pt")
    with torch.no_grad():
        model_logits = pipe.hf_model(**enc).logits
    model_probs = model_logits[:, -1].softmax(-1)
    torch.testing.assert_close(probs[:, -1], model_probs, atol=1e-4, rtol=1e-3)


@pytest.mark.unit
def test_get_topk_closest_tokens_shapes(tiny_pipeline):
    pipe = tiny_pipeline
    hidden_size = pipe.model.hidden_size

    single = get_topk_closest_tokens(pipe, torch.randn(hidden_size), k=3)
    assert isinstance(single, dict) and len(single) == 3

    batched = get_topk_closest_tokens(pipe, torch.randn(2, hidden_size), k=4)
    assert isinstance(batched, list) and len(batched) == 2
    assert all(isinstance(d, dict) and len(d) == 4 for d in batched)


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
