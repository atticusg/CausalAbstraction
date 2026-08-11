"""
Logit Lens — project intermediate residual-stream activations to vocabulary logits.

The logit lens (nostalgebraist, 2020) reads a transformer's intermediate
computation by applying the model's *final layer norm* followed by the
*unembedding head* (``lm_head``) to a hidden vector taken from some layer's
residual stream. The resulting vocabulary distribution shows which token the
model is "leaning toward" at that depth, giving a training-free view of how a
prediction forms across layers.

This is a **read-only projection**, so it is implemented as a functional method
(mirroring ``causalab/methods/pca.py``) rather than a ``Featurizer`` — there is
no invertible encode/decode and it never participates in interventions.

Reuse, not reinvention:
- ``build_residual_stream_sites`` builds the (layer × token) site grid.
- ``collect_features`` runs a single forward pass and returns the hidden vectors.
- ``class_probabilities`` (``methods/metric.py``) computes prob mass over a set
  of token ids for the optional target-token track.
- The **final layer norm + unembedding** are resolved by nnterp's
  ``StandardizedTransformer`` (its cross-architecture rename scheme), not by a
  bespoke per-family attribute probe — see :func:`resolve_final_norm_and_unembed`.
  This mirrors nnterp's own ``project_on_vocab``; causalab keeps the float32 /
  device-aware matmul (:func:`project_to_logits`) for numerically stable topk.

This module also re-exposes nnterp's trace-based analysis primitives as thin,
causalab-facing wrappers over a ``pipeline``: :func:`project_on_vocab`,
:func:`logit_lens`, :func:`patchscope_lens`, :func:`patchscope_generate`, and
:func:`get_topk_closest_tokens` (plus the patchscope prompt scaffolds
``TargetPrompt`` / ``repeat_prompt`` / ``it_repeat_prompt`` /
``TargetPromptBatch``). These are analysis-neutral primitives (no disk layout,
no Hydra) — the ``analyses/`` layer owns the research question and artifacts.

Sharp edges:
- The final layer norm is **mandatory and architecture-specific**; skipping it
  (or grabbing the wrong module) yields garbage distributions. nnterp's rename
  scheme locates it (``ln_final``) across families; ``apply_final_norm`` toggles
  it on/off to compare with/without.
- Logit lens is systematically **biased on non-GPT-2 models** (intermediate
  layers live in a rotated/shifted basis).
- Only top-``k`` ids/probs are retained per (sample, layer) — full
  ``vocab_size`` logits are never stored, which bounds artifact size.

Output Structure (via ``save_logit_lens_results``):
================
output_dir/
├── metadata.json                       # Experiment configuration
├── top_k/                              # Top-k tokens per (layer, position)
│   ├── {layer}__{pos_id}.safetensors   # token_ids (n,k), probs (n,k)
│   └── {layer}__{pos_id}.json          # decoded token strings + summary
└── target_track/                       # Optional answer-token probability track
    └── {layer}__{pos_id}.safetensors   # answer_mass (n,)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnterp import StandardizedTransformer

# Patchscope prompt scaffolds are re-exported so callers can build patchscope
# target prompts without importing nnterp directly.
from nnterp.interventions import (  # noqa: F401 — re-exported convenience
    TargetPrompt,
    TargetPromptBatch,
    it_repeat_prompt,
    repeat_prompt,
)

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.io.artifacts import (
    save_experiment_metadata as _save_experiment_metadata,
    save_tensor_results as _save_tensor_results,
    save_json_results as _save_json_results,
)
from causalab.methods.metric import class_probabilities
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.site_grids import build_residual_stream_sites
from causalab.neural.pipeline import Pipeline
from causalab.neural.token_positions import TokenPosition

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model-internals accessors (nnterp-standardized)                             #
# --------------------------------------------------------------------------- #


def _unwrap_envoy(obj: Any) -> nn.Module:
    """Return the underlying ``nn.Module`` for an nnsight/nnterp ``Envoy``.

    nnterp's own ``ModuleAccessor`` reaches the raw module via ``._module``; a
    plain ``nn.Module`` (already unwrapped) passes through unchanged.
    """
    module = getattr(obj, "_module", obj)
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"expected an nn.Module (or an Envoy wrapping one), got "
            f"{type(module).__name__}"
        )
    return module


def resolve_final_norm_and_unembed(pipeline: Pipeline) -> tuple[nn.Module, nn.Module]:
    """The final layer norm + unembedding head, resolved via nnterp.

    Replaces causalab's bespoke per-family attribute probing (the removed
    ``_FINAL_NORM_PATHS``): nnterp's ``StandardizedTransformer`` locates the
    final norm (``ln_final``) and unembedding (``lm_head``) across 16+ decoder
    families through its rename scheme, so onboarding a new architecture is
    nnterp's job, not a causalab code change. The raw ``nn.Module`` for each is
    returned so they can be applied to *collected* (out-of-trace) hidden states
    — the logit-lens collect-then-project flow.

    ``pipeline.model`` is normally the validated ``StandardizedTransformer``;
    for a raw-HF pipeline we standardize names on the underlying module via
    nnterp's ``ModuleAccessor`` rather than re-introducing the attribute probe.
    """
    model = pipeline.model
    if isinstance(model, StandardizedTransformer):
        # The wrapper already resolved these under any custom rename_config it
        # was built with — reuse them rather than re-deriving.
        return _unwrap_envoy(model.ln_final), _unwrap_envoy(model.lm_head)

    # Raw-HF fallback (no standardized wrapper): standardize names on the raw
    # module with nnterp's ModuleAccessor — the documented primitive for
    # "standardized submodule access outside a trace".
    from nnterp.nnsight_utils import ModuleAccessor

    accessor = ModuleAccessor(pipeline.hf_model)
    return accessor.get_unembed_norm(), accessor.get_unembed()


def project_to_logits(
    hidden: torch.Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """Project hidden vectors ``(n, hidden_size)`` to logits ``(n, vocab_size)``.

    Applies the final layer norm (optionally) then the unembedding head. The
    matmul runs in float32 for numerically stable downstream softmax/topk, and
    tensors are moved onto the norm's / unembedding's own device so that
    ``device_map="auto"`` shards (where the head may live on a specific GPU)
    work without manual placement.
    """
    if apply_final_norm:
        norm_param = next(final_norm.parameters(), None)
        if norm_param is not None:
            hidden = final_norm(
                hidden.to(device=norm_param.device, dtype=norm_param.dtype)
            )
        else:
            hidden = final_norm(hidden)

    weight = lm_head.weight
    hidden = hidden.to(device=weight.device, dtype=torch.float32)
    logits = hidden @ weight.to(torch.float32).t()
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias.to(device=logits.device, dtype=torch.float32)
    return logits


def project_on_vocab(
    pipeline: Pipeline,
    hidden: torch.Tensor,
    *,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """Project hidden vectors to vocabulary logits — the core logit-lens primitive.

    causalab's counterpart to nnterp's ``project_on_vocab(model, h)``: resolves
    the standardized final norm + unembedding via :func:`resolve_final_norm_and_unembed`
    (nnterp), then applies them in float32 with shard-aware device placement
    (:func:`project_to_logits`). Unlike nnterp's in-trace helper this operates on
    already-collected tensors and pins float32, so downstream softmax/topk is
    numerically stable even when the model runs in bf16.

    Args:
        pipeline: Loaded pipeline (real weights required).
        hidden: ``(..., hidden_size)`` residual-stream vectors.
        apply_final_norm: Apply the final layer norm before unembedding (the
            faithful logit lens); ``False`` inspects the raw projection.

    Returns:
        Float32 logits ``(..., vocab_size)``.
    """
    final_norm, lm_head = resolve_final_norm_and_unembed(pipeline)
    return project_to_logits(hidden, final_norm, lm_head, apply_final_norm)


# --------------------------------------------------------------------------- #
# nnterp trace-based analysis primitives (thin causalab-facing wrappers)       #
# --------------------------------------------------------------------------- #


def _standardized_model(pipeline: Pipeline) -> StandardizedTransformer:
    """Return the pipeline's ``StandardizedTransformer`` (nnterp) for tracing."""
    model = pipeline.model
    if not isinstance(model, StandardizedTransformer):
        raise TypeError(
            "nnterp trace-based lenses need a StandardizedTransformer-backed "
            f"pipeline; got pipeline.model of type {type(model).__name__}. Load "
            "the pipeline with real weights (load_weights=True)."
        )
    return model


def logit_lens(
    pipeline: Pipeline,
    prompts: list[str] | str,
    *,
    return_inv_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-layer next-token probabilities for the last token (nnterp).

    Thin wrapper over ``nnterp.interventions.logit_lens`` — one forward pass that
    projects every layer's last-token residual through the final norm +
    unembedding. Returns ``(num_prompts, num_layers, vocab)`` CPU probabilities
    (a ``(probs, inv_probs)`` pair when ``return_inv_logits``). For the richer
    (layer × token-position) grid over a counterfactual dataset, use
    :func:`compute_logit_lens`.
    """
    from nnterp.interventions import logit_lens as _nnterp_logit_lens

    return _nnterp_logit_lens(
        _standardized_model(pipeline), prompts, return_inv_logits=return_inv_logits
    )


def patchscope_lens(
    pipeline: Pipeline,
    source_prompts: list[str] | str | None = None,
    target_patch_prompts: (
        "TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None"
    ) = None,
    layers: int | list[int] | None = None,
    latents: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patchscope decoding of intermediate states (nnterp).

    Thin wrapper over ``nnterp.interventions.patchscope_lens``: patches each
    source prompt's per-layer last-token hidden state into a target (repeat)
    prompt and reads the next-token distribution — the patchscopes-paper lens
    causalab otherwise lacks. Returns ``(num_sources, num_layers, vocab)`` CPU
    probabilities. Build ``target_patch_prompts`` with :func:`repeat_prompt` /
    :func:`it_repeat_prompt` (defaults to ``repeat_prompt()``).
    """
    from nnterp.interventions import patchscope_lens as _nnterp_patchscope_lens

    return _nnterp_patchscope_lens(
        _standardized_model(pipeline),
        source_prompts=source_prompts,
        target_patch_prompts=target_patch_prompts,
        layers=layers,
        latents=latents,
    )


def patchscope_generate(
    pipeline: Pipeline,
    prompts: list[str] | str,
    target_patch_prompt: "TargetPrompt",
    *,
    max_length: int = 50,
    layers: list[int] | None = None,
    max_batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """Patchscope generation per layer (nnterp).

    Thin wrapper over ``nnterp.interventions.patchscope_generate``: for each
    layer, patch the source hidden state into ``target_patch_prompt`` and let the
    model generate. Returns ``{layer: generated token ids}`` (CPU).
    """
    from nnterp.interventions import patchscope_generate as _nnterp_patchscope_generate

    return _nnterp_patchscope_generate(
        _standardized_model(pipeline),
        prompts,
        target_patch_prompt,
        max_length=max_length,
        layers=layers,
        max_batch_size=max_batch_size,
    )


def get_topk_closest_tokens(
    pipeline: Pipeline,
    hidden: torch.Tensor,
    k: int = 5,
) -> dict[str, float] | list[dict[str, float]]:
    """Nearest vocabulary tokens for a hidden state (nnterp).

    Projects ``hidden`` on the vocabulary (final norm + unembedding) and returns
    the top-``k`` ``{token: prob}`` map (a list of maps for a 2-D batch) — a
    quick nearest-token readout for any residual-stream vector. Delegates to
    ``StandardizedTransformer.get_topk_closest_tokens``.
    """
    return _standardized_model(pipeline).get_topk_closest_tokens(hidden, k=k)


def _top_k_tokens(
    logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    """Return ``(token_ids (n,k), probs (n,k), token_strings)`` for ``logits``.

    Mirrors the top-k decode pattern in ``causalab/analyses/baseline/main.py``.
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
    tokens = [tokenizer.convert_ids_to_tokens(row.tolist()) for row in top_ids.cpu()]
    return top_ids.cpu(), top_probs.cpu(), tokens


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def compute_logit_lens(
    dataset: list[CounterfactualExample],
    pipeline: Pipeline,
    layers: List[int],
    token_positions: List[TokenPosition],
    top_k: int,
    *,
    batch_size: int,
    target_token_ids: Optional[List[int]] = None,
    apply_final_norm: bool = True,
) -> Dict[str, Any]:
    """Run the logit lens over a (layer × token_position) grid.

    Collects residual-stream activations at each requested ``(layer, position)``
    in a single forward pass, projects them through the final norm + unembedding,
    and records the top-``k`` predicted tokens per example. Optionally tracks the
    total probability mass on a set of "answer" token ids across layers.

    Args:
        dataset: Counterfactual examples to run the lens on.
        pipeline: Loaded ``LMPipeline`` (real weights required).
        layers: Residual-stream layer indices. Follows the codebase convention:
            ``block_output`` at layer L (the stream *after* block L); ``-1`` maps
            to the embedding output (``block_input`` at layer 0).
        token_positions: Token positions to read the lens at (e.g. last token).
        top_k: Number of top tokens to retain per (sample, layer, position).
        batch_size: Forward-pass batch size for activation collection.
        target_token_ids: Optional flat list of "answer" vocab ids. When given,
            the returned ``target_track`` reports the full-vocab-softmax
            probability mass on those ids per (layer, position) — the standard
            "does the answer emerge with depth?" curve.
        apply_final_norm: Apply the model's final layer norm before unembedding
            (the faithful logit lens). Set False to inspect the raw projection.

    Returns:
        Dict with:
            - ``top_k_by_unit``: ``{(layer, pos_id): {"token_ids" (n,k),
              "probs" (n,k), "tokens": list[list[str]]}}``
            - ``target_track``: ``{(layer, pos_id): {"answer_mass" (n,),
              "answer_mass_mean": float}}`` or ``None``
            - ``layers``, ``token_position_ids``, ``metadata``

    Raises:
        ValueError: if ``pipeline`` has no usable unembedding head.
    """
    # nnterp resolves the final norm + unembedding across architectures (its
    # rename scheme), replacing causalab's bespoke per-family probe. The raw
    # nn.Modules are returned so they apply to *collected* hidden states outside
    # any trace (this collect-then-project flow); project_to_logits pins float32.
    final_norm, lm_head = resolve_final_norm_and_unembed(pipeline)

    # Build one residual-stream site per (layer, position) and flatten to a
    # single list for a single-pass collection (same approach as pca.py).
    grid = build_residual_stream_sites(
        pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
        target_output=True,
    )
    spec_key_to_cell: Dict[str, Tuple[Any, ...]] = {}
    all_specs = []
    for key, groups in grid.items():
        for group in groups:
            for spec in group:
                all_specs.append(spec)
                spec_key_to_cell[spec.key] = key

    logger.info(
        "Logit lens: collecting %d residual-stream units over %d examples",
        len(all_specs),
        len(dataset),
    )
    features_by_unit = collect_features(
        dataset, pipeline, all_specs, batch_size=batch_size
    )
    assert isinstance(features_by_unit, dict)

    top_k_by_unit: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    target_track: Dict[Tuple[Any, ...], Dict[str, Any]] | None = (
        {} if target_token_ids else None
    )

    for spec_key, hidden in features_by_unit.items():
        key = spec_key_to_cell[spec_key]
        logits = project_to_logits(hidden, final_norm, lm_head, apply_final_norm)

        token_ids, probs, tokens = _top_k_tokens(logits, pipeline.tokenizer, top_k)
        top_k_by_unit[key] = {
            "token_ids": token_ids,
            "probs": probs,
            "tokens": tokens,
        }

        if target_track is not None:
            assert target_token_ids is not None
            # Treat the answer tokens as a single class; full-vocab softmax gives
            # true P(any answer token). Shape (n, 1) -> (n,).
            answer_mass = class_probabilities(
                logits, [target_token_ids], full_vocab_softmax=True
            ).squeeze(-1)
            target_track[key] = {
                "answer_mass": answer_mass.cpu(),
                "answer_mass_mean": float(answer_mass.mean().item()),
            }

    token_position_ids = [tp.id for tp in token_positions]
    # Intrinsic method metadata only — run-level tagging (experiment_type, task,
    # model, seed, …) is an analysis-layer concern (docs/CODEBASE.md invariant 4).
    metadata: Dict[str, Any] = {
        "num_samples": len(dataset),
        "layers": list(layers),
        "token_position_ids": token_position_ids,
        "top_k": top_k,
        "apply_final_norm": apply_final_norm,
        "has_target_track": target_track is not None,
        "vocab_size": int(getattr(pipeline.model.config, "vocab_size", -1)),
        "batch_size": batch_size,
    }

    return {
        "top_k_by_unit": top_k_by_unit,
        "target_track": target_track,
        "layers": list(layers),
        "token_position_ids": token_position_ids,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Save helper (artifact serialization policy)                                 #
# --------------------------------------------------------------------------- #


def _key_to_str(key: Tuple[Any, ...]) -> str:
    """Convert a (layer, pos_id) key to a filename-safe string."""
    return "__".join(str(k) for k in key)


def save_logit_lens_results(
    result: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Persist ``compute_logit_lens`` output: metadata, per-unit top-k, target track.

    Tensors go to ``.safetensors`` and decoded strings/summaries to ``.json``
    via the canonical ``causalab.io.artifacts`` helpers (no raw ``torch.save``).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {
        "metadata_path": _save_experiment_metadata(result["metadata"], output_dir),
    }

    top_k_dir = os.path.join(output_dir, "top_k")
    os.makedirs(top_k_dir, exist_ok=True)
    for key, payload in result["top_k_by_unit"].items():
        stem = _key_to_str(key)
        _save_tensor_results(
            {"token_ids": payload["token_ids"], "probs": payload["probs"]},
            top_k_dir,
            f"{stem}.safetensors",
        )
        _save_json_results(
            {
                "layer": key[0],
                "token_position": key[1] if len(key) > 1 else None,
                "tokens": payload["tokens"],
            },
            top_k_dir,
            f"{stem}.json",
        )
    paths["top_k_dir"] = top_k_dir

    if result["target_track"] is not None:
        track_dir = os.path.join(output_dir, "target_track")
        os.makedirs(track_dir, exist_ok=True)
        for key, payload in result["target_track"].items():
            stem = _key_to_str(key)
            _save_tensor_results(
                {"answer_mass": payload["answer_mass"]},
                track_dir,
                f"{stem}.safetensors",
            )
        # Compact per-cell mean summary for quick inspection / downstream plots.
        _save_json_results(
            {
                _key_to_str(key): payload["answer_mass_mean"]
                for key, payload in result["target_track"].items()
            },
            track_dir,
            "answer_mass_mean.json",
        )
        paths["target_track_dir"] = track_dir

    return paths
