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
- ``build_residual_stream_targets`` builds the (layer × token) unit grid.
- ``collect_features`` runs a single forward pass and returns the hidden vectors.
- ``class_probabilities`` (``methods/metric.py``) computes prob mass over a set
  of token ids for the optional target-token track.

Sharp edges:
- The final layer norm is **mandatory and architecture-specific**; skipping it
  (or grabbing the wrong module) yields garbage distributions. See
  ``get_final_norm`` for the arch-aware accessor and ``apply_final_norm`` to
  compare with/without.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.io.artifacts import (
    save_experiment_metadata as _save_experiment_metadata,
    save_tensor_results as _save_tensor_results,
    save_json_results as _save_json_results,
)
from causalab.methods.metric import class_probabilities
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.targets import build_residual_stream_targets
from causalab.neural.pipeline import Pipeline
from causalab.neural.token_positions import TokenPosition

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model-internals accessors (architecture-aware)                              #
# --------------------------------------------------------------------------- #

# Candidate attribute chains for the final layer norm across HF decoder families.
# Tried in order; first one that resolves to a Module wins.
_FINAL_NORM_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "norm"),  # Llama, Mistral, Qwen2/3, Gemma
    ("model", "final_layernorm"),  # Phi, some others
    ("transformer", "ln_f"),  # GPT-2, GPT-J, Falcon
    ("gpt_neox", "final_layer_norm"),  # GPT-NeoX, Pythia
    ("transformer", "norm_f"),  # MPT
    ("norm",),  # bare decoder (no outer LM head wrapper)
)


def _resolve_attr_chain(obj: Any, chain: Sequence[str]) -> Any | None:
    """Follow a dotted attribute chain, returning None if any hop is missing."""
    cur = obj
    for name in chain:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def get_final_norm(model: nn.Module) -> nn.Module:
    """Return the model's final layer norm (the one applied before ``lm_head``).

    There is no universal HuggingFace API for this, so we probe the known
    attribute chains for each decoder family. Raises a clear error if none
    match — better to fail loudly than to silently project an unnormalized
    hidden state and produce a garbage distribution.
    """
    for chain in _FINAL_NORM_PATHS:
        candidate = _resolve_attr_chain(model, chain)
        if isinstance(candidate, nn.Module):
            return candidate
    raise ValueError(
        "Could not locate the final layer norm on this model "
        f"({type(model).__name__}). Tried: "
        f"{['.'.join(c) for c in _FINAL_NORM_PATHS]}. Add this architecture's "
        "norm path to _FINAL_NORM_PATHS in causalab/methods/logit_lens.py."
    )


def get_lm_head(model: nn.Module) -> nn.Module:
    """Return the unembedding head, going through ``get_output_embeddings()``.

    This is the correct entry point because it transparently handles weight
    tying (where ``lm_head`` shares storage with the input embeddings).
    """
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError(
            f"Model {type(model).__name__}.get_output_embeddings() returned None; "
            "logit lens requires an unembedding head (real weights must be loaded)."
        )
    return lm_head


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
    model = pipeline.model
    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)

    # Build one residual-stream unit per (layer, position) and flatten to a
    # single list for a single-pass collection (same approach as pca.py).
    targets = build_residual_stream_targets(
        pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
        target_output=True,
    )
    unit_to_key: Dict[str, Tuple[Any, ...]] = {}
    all_units = []
    for key, target in targets.items():
        for unit in target.flatten():
            all_units.append(unit)
            unit_to_key[unit.id] = key

    logger.info(
        "Logit lens: collecting %d residual-stream units over %d examples",
        len(all_units),
        len(dataset),
    )
    features_by_unit = collect_features(
        dataset, pipeline, all_units, batch_size=batch_size
    )
    assert isinstance(features_by_unit, dict)

    top_k_by_unit: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    target_track: Dict[Tuple[Any, ...], Dict[str, Any]] | None = (
        {} if target_token_ids else None
    )

    for unit_id, hidden in features_by_unit.items():
        key = unit_to_key[unit_id]
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
        "vocab_size": int(getattr(model.config, "vocab_size", -1)),
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
