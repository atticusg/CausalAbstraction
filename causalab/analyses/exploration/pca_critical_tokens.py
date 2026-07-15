"""``pca`` mode: PCA over the residual stream at each critical token, many inputs.

For every essential-token slot and every layer, this collects the residual
stream across a large input sample (~10k), fits centered PCA, and saves the
projection of all inputs onto the top-``n_components`` principal components.
Those saved projections (plus per-PC explained variance) are everything the
report and web app need to draw, per critical token at each layer, a scatter in
any chosen pair/triple of PCs — colored by whatever per-input label schemes the
caller supplies. Reuses ``build_residual_stream_targets`` + ``collect_features``
+ ``compute_svd`` directly.

Memory is bounded by processing one token at a time (all its layers, then
freed), which is also the natural fan unit across shards via
``exploration.pca.tokens`` (see ``causalab/runner/fanout.py``).

Config (``cfg.exploration.pca``):

* ``inputs`` — JSON list whose elements are either a prompt string or an object
  ``{"input": "2 + 3 =", "positions": {"<label>": <int>}}`` giving each
  essential-token slot's per-input position (use when the slot moves between
  inputs, e.g. variable-width operands).
* ``essential_tokens`` — JSON list of
  ``{"label": str, "index": <int>?, "text": <substring>?, "occurrence": <int>?}``.
  Per-input position resolution, in priority order: that input's
  ``positions[label]``; the slot's fixed ``index``; the last token overlapping
  the slot's ``text`` substring.
* ``labels`` (optional) — JSON list of per-input flat mappings (color schemes),
  saved verbatim for the web app's color-by dropdown.
* ``n_components`` / ``layers`` / ``tokens`` / ``batch_size``.
"""

from __future__ import annotations

import json
import os

from omegaconf import DictConfig

from causalab.io.artifacts import (
    save_experiment_metadata,
    save_json_results,
    save_tensors_with_meta,
)
from causalab.methods.pca import compute_svd
from causalab.neural.activations.collect import collect_features
from causalab.neural.activations.targets import build_residual_stream_targets
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition, get_substring_token_ids


def _load_inputs(path: str):
    """Return (prompts, positions_per_input) from the inputs file.

    positions_per_input[i] is a dict {label: int} or {} when the input was a
    bare string.
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError("exploration.pca.inputs must be a non-empty JSON list")
    prompts: list[str] = []
    positions: list[dict] = []
    for el in raw:
        if isinstance(el, str):
            prompts.append(el)
            positions.append({})
        elif isinstance(el, dict) and "input" in el:
            prompts.append(el["input"])
            positions.append(el.get("positions", {}) or {})
        else:
            raise ValueError(
                "Each inputs element must be a string or an object with an "
                f"'input' key; got: {el!r}"
            )
    return prompts, positions


def _make_token_position(
    token: dict, prompts, positions, pipeline: LMPipeline
) -> TokenPosition:
    """Build a per-example TokenPosition resolving this slot in every input.

    The resolved position depends only on the prompt string, so a lookup keyed
    by prompt text is safe: identical prompts share the same slot position.
    """
    label = token["label"]
    fixed_index = token.get("index")
    substring = token.get("text")
    occurrence = int(token.get("occurrence", 0))

    lut: dict[str, int] = {}
    for prompt, pos in zip(prompts, positions):
        if label in pos:
            idx = int(pos[label])
        elif fixed_index is not None:
            idx = int(fixed_index)
        elif substring is not None:
            ids = get_substring_token_ids(
                prompt, substring, pipeline, occurrence=occurrence
            )
            if not ids:
                raise ValueError(
                    f"substring {substring!r} for token {label!r} not found in {prompt!r}"
                )
            idx = ids[-1]  # last token overlapping the slot
        else:
            raise ValueError(
                f"token {label!r}: provide per-input 'positions', a fixed 'index', "
                "or a 'text' substring to locate it"
            )
        lut[prompt] = idx

    def indexer(inp, _lut=lut):
        # collect_features passes each example["input"] to the indexer, which is
        # the {"raw_input": prompt} wrapper; fall back to a bare string.
        key = inp["raw_input"] if isinstance(inp, dict) else inp
        return [_lut[key]]

    safe_label = label.replace("/", "_").replace(" ", "_")
    return TokenPosition(indexer=indexer, pipeline=pipeline, id=f"crit_{safe_label}")


def run(pipeline, acfg: DictConfig, out_dir: str) -> str:
    prompts, positions = _load_inputs(acfg.inputs)
    with open(acfg.essential_tokens) as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(
            "exploration.pca.essential_tokens must be a non-empty JSON list"
        )

    labels = None
    if acfg.get("labels"):
        with open(acfg.labels) as f:
            labels = json.load(f)
        if not isinstance(labels, list) or len(labels) != len(prompts):
            raise ValueError(
                f"exploration.pca.labels must be a JSON list with one row per input "
                f"({len(prompts)}), got "
                f"{len(labels) if isinstance(labels, list) else type(labels)}"
            )

    tokens_arg = acfg.get("tokens")
    token_subset = (
        [int(i) for i in str(tokens_arg).split(",")]
        if tokens_arg
        else list(range(len(tokens)))
    )
    n_components = int(acfg.n_components)
    batch_size = int(acfg.batch_size)

    n_layers = pipeline.model.config.num_hidden_layers
    layers_arg = acfg.get("layers")
    layers = (
        [int(x) for x in str(layers_arg).split(",")]
        if layers_arg
        else list(range(n_layers))
    )
    model_name = getattr(pipeline.model, "name_or_path", "")

    data = [{"input": {"raw_input": p}} for p in prompts]

    # Shared top-level files the web app reads (identical regardless of which
    # token subset this run owns). Only the shard owning token 0 emits them, so a
    # per-token fan-out (`exploration.pca.tokens` sharded via
    # causalab/runner/fanout.py) yields exactly one authoritative copy to collect;
    # the single-job default path (no `tokens` override) always writes them.
    if not tokens_arg or token_subset[0] == 0:
        save_experiment_metadata(
            {
                "experiment_type": "pca_critical_tokens",
                "model": model_name,
                "num_inputs": len(prompts),
                "n_components": n_components,
                "layers": layers,
                "tokens": [
                    {
                        "label": t["label"],
                        **{k: t[k] for k in ("index", "text") if k in t},
                    }
                    for t in tokens
                ],
                "label_columns": sorted({k for row in labels for k in row})
                if labels
                else [],
                "batch_size": batch_size,
            },
            out_dir,
        )
        save_json_results({"inputs": prompts}, out_dir, "inputs.json")
        if labels is not None:
            save_json_results({"rows": labels}, out_dir, "labels.json")

    for ti in token_subset:
        token = tokens[ti]
        label = token["label"]
        tp = _make_token_position(token, prompts, positions, pipeline)
        targets = build_residual_stream_targets(
            pipeline, layers=layers, token_positions=[tp], mode="one_target_per_unit"
        )

        # One forward pass collects this token's residual stream at every layer.
        features_by_unit = collect_features(
            data,
            pipeline,
            [u for t in targets.values() for u in t.flatten()],
            batch_size=batch_size,
        )
        assert isinstance(features_by_unit, dict)

        safe_label = label.replace("/", "_").replace(" ", "_")
        token_dir = os.path.join(out_dir, safe_label)
        for (layer, _pos_id), target in targets.items():
            unit = target.flatten()[0]
            feats = features_by_unit[unit.id].float()  # (n_inputs, hidden)
            svd = compute_svd(
                {"f": feats}, n_components=n_components, preprocess="center"
            )["f"]
            mean = svd["mean"]  # (1, hidden)
            rotation = svd["rotation"]  # (hidden, n_components)
            projections = (feats - mean) @ rotation  # (n_inputs, n_components)
            save_tensors_with_meta(
                {
                    "projections": projections.contiguous(),
                    "rotation": rotation.contiguous(),
                    "mean": mean.contiguous(),
                },
                {
                    "token": label,
                    "layer": int(layer),
                    "n_components": int(svd["n_components"]),
                    "num_inputs": projections.shape[0],
                    "explained_variance_ratio": [
                        float(x) for x in svd["explained_variance_ratio"]
                    ],
                },
                token_dir,
                f"L{layer}",
            )
        print(f"[pca] token {ti} {label!r}: {len(layers)} layers -> {token_dir}")
        del features_by_unit

    return out_dir
