"""``knockout`` mode: component-ablation knockout sweeps over raw prompts.

Knocks out model components and measures, for each cell, how much the model's
behavior moves when a component's output is replaced across the configured token
span. Because exploration is task-less, both metrics are graded against the
model's *own* un-ablated output for each input — every cell is scored under
**two** metrics from one set of generations:

* ``match_drop`` — fraction of inputs whose greedy output *changed* (the binary
  behavioral drop; ``1 - match-the-base``). Coarse but interpretable.
* ``logit_diff`` — mean drop in the base-predicted token's logit
  (``base_logit[pred] - ablated_logit[pred]`` at the first generated position).
  Graded/continuous, so a cell that suppresses the original prediction without
  flipping the argmax still registers.

Larger is more disruptive for both. This is the task-less analogue of the
accuracy drop the ``ablation`` analysis reports against a task label.

Two component families, both reusing the ``causalab.methods.ablation`` primitives
(``run_ablation_scan_multi`` / ``run_ablation_combo_multi``) — no parallel hooking
system:

* **attention_head** — the full ``(layer × head)`` grid (one drop per head),
  *plus* whole-sublayer layer-bands: every head in a contiguous layer band
  ablated jointly, swept over ``head_band_widths``. A width-1 band knocks out a
  whole attention sublayer (all heads in one layer).
* **mlp** — sliding contiguous layer-bands of each configured ``mlp_widths``.
  Width 1 is the per-layer scan; wider bands ablate every layer in the window
  *jointly*, giving a drop-vs-band-start strip per width.

For both families the band sweep is a **necessity** measure ("does behavior need
this band?"). With ``complement=True`` each band is additionally run inverted —
ablate every *other* configured layer, keeping the band — a **sufficiency**
measure ("can behavior survive on just this band?"), stored under
``complement_widths``.

Both ablation references are run when ``ablation_modes`` lists both: ``zero``
(drop the contribution) and ``mean`` (replace with the corpus-mean activation).
Each reference writes its own ``<ablation_mode>/results.json`` under the mode's
output dir, carrying a per-metric grid for each family; the optional web app
turns the saved grids into per-width / per-family tabs with a metric selector.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from omegaconf import DictConfig

from causalab.analyses.exploration.pca_critical_tokens import (
    _load_inputs,  # pyright: ignore[reportPrivateUsage]
    _make_token_position,  # pyright: ignore[reportPrivateUsage]
)
from causalab.causal.trace import CausalTrace, Mechanism
from causalab.methods.ablation import (
    make_mean_vectors,
    make_zero_vectors,
    run_ablation_combo_multi,
    run_ablation_scan_multi,
)
from causalab.methods.metric import InterchangeMetric, compute_base_outputs
from causalab.neural.activations.targets import (
    build_attention_head_targets,
    build_mlp_targets,
)
from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    TokenPosition,
    get_all_tokens,
    get_last_token_index,
)
from causalab.neural.units import InterchangeTarget

logger = logging.getLogger(__name__)

_COMPONENTS = ("attention_head", "mlp")
_ABLATION_MODES = ("zero", "mean")


def _trace(text: str) -> CausalTrace:
    """Minimal single-input trace; only ``example["input"]`` is used for ablation."""
    return CausalTrace(
        mechanisms={
            "raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])
        },
        inputs={"raw_input": text},
    )


# Per-metric display info for the web app's metric selector. Both metrics are
# oriented so that *larger = more disruptive* (a 0-anchored sequential scale).
_METRIC_INFO: dict[str, dict[str, str]] = {
    "match_drop": {
        "label": "Behavioral drop",
        "description": "fraction of inputs whose greedy output changed",
        "colorbar": "behavioral drop",
    },
    "logit_diff": {
        "label": "Predicted-token logit drop",
        "description": "mean base−ablated logit of the base-predicted token",
        "colorbar": "logit drop",
    },
}


def _match_drop_metric() -> InterchangeMetric:
    """Binary behavioral-drop metric: score 1.0 when the knockout *changed* the
    greedy output vs the un-ablated base, 0.0 when it left it unchanged.

    Averaged over inputs this is the share the knockout moved — already oriented
    as a drop (larger = more disruptive), so the caller stores it directly."""

    def fn(
        intervention_output: dict[str, Any],
        _expected: dict[str, Any],
        original: dict[str, Any],
    ) -> float:
        return float(
            str(intervention_output.get("string", "")).strip()
            != str(original.get("string", "")).strip()
        )

    return InterchangeMetric(
        fn=fn, needs_causal_expected=False, needs_original_output=True
    )


def _logit_diff_metric() -> InterchangeMetric:
    """Graded knockout metric: drop in the base-predicted token's logit.

    For each input the base model's greedy first token is the "predicted token";
    this returns ``base_logit[pred] - ablated_logit[pred]`` at the first generated
    position, so a larger value means the knockout suppressed the original
    prediction more (it is positive when the component supported that token).
    Reads raw logits from both the ablated run (``intervention_output["scores"]``)
    and the base run (``original["scores"]``), so the scan must request full-vocab
    ``output_scores=True`` — the base-predicted token may fall outside the ablated
    run's top-k otherwise."""

    def fn(
        intervention_output: dict[str, Any],
        _expected: dict[str, Any],
        original: dict[str, Any],
    ) -> float:
        ablated_scores = intervention_output.get("scores")
        base_scores = original.get("scores")
        if ablated_scores is None or base_scores is None:
            raise ValueError(
                "logit_diff metric needs full-vocab scores from both the ablated "
                "run (output_scores=True) and the base run (compute_base_outputs)."
            )
        idx = intervention_output["example_idx"]
        base_logits = base_scores[0]  # (vocab,) first generated position
        ablated_logits = ablated_scores[0][idx]  # (vocab,)
        pred = int(base_logits.argmax())
        return float(base_logits[pred] - ablated_logits[pred])

    return InterchangeMetric(
        fn=fn,
        needs_causal_expected=False,
        needs_original_output=True,
        needs_scores=True,
    )


def _build_metrics() -> dict[str, InterchangeMetric]:
    """The metric set scored for every knockout cell, in display order."""
    return {
        "match_drop": _match_drop_metric(),
        "logit_diff": _logit_diff_metric(),
    }


def _resolve_span(
    acfg: DictConfig, prompts: list[str], positions: list[dict], pipeline: LMPipeline
) -> TokenPosition:
    """Resolve the ``span`` config into a single ``TokenPosition``.

    ``all`` (every non-pad token) and ``last`` (final token) need no token file;
    ``essential`` unions the positions of every slot in ``essential_tokens`` into
    one span, reusing the PCA mode's per-slot resolver so the index/text/per-input
    resolution rules stay identical across the two modes.
    """
    span = acfg.get("span", "all")
    if span == "all":
        return get_all_tokens(_trace(prompts[0]), pipeline)
    if span == "last":
        return TokenPosition(
            lambda inp: get_last_token_index(inp, pipeline), pipeline, id="last_token"
        )
    if span == "essential":
        path = acfg.get("essential_tokens")
        if not path:
            raise ValueError(
                "knockout.span='essential' requires knockout.essential_tokens "
                "(path to an essential_tokens.json listing the essential token slots)."
            )
        with open(path) as f:
            tokens = json.load(f)
        slot_positions = [
            _make_token_position(tok, prompts, positions, pipeline) for tok in tokens
        ]

        def union_indexer(inp, _slots=slot_positions):
            idxs: set[int] = set()
            for slot in _slots:
                idxs.update(slot.index(inp))
            return sorted(idxs)

        return TokenPosition(indexer=union_indexer, pipeline=pipeline, id="essential")
    raise ValueError(
        f"knockout.span must be 'all', 'last', or 'essential'; got {span!r}"
    )


def _reference_vectors(
    ablation_mode: str,
    pipeline: LMPipeline,
    dataset: list[dict[str, Any]],
    units: list[Any],
    batch_size: int,
) -> dict[str, Any]:
    """Zero or corpus-mean reference vector per unit id, built once for the whole
    grid (one ``make_*`` call over a combined one-unit-per-group target)."""
    combined = InterchangeTarget([[u] for u in units])
    if ablation_mode == "zero":
        return make_zero_vectors(combined)
    return make_mean_vectors(pipeline, dataset, combined, batch_size=batch_size)


def _band_sweep(
    *,
    family: str,
    layers: list[int],
    widths: list[int],
    units_for_layers: Callable[[list[int]], list[Any]],
    complement: bool,
    pipeline: LMPipeline,
    dataset: list[dict[str, Any]],
    vectors: dict[str, Any],
    metrics: dict[str, InterchangeMetric],
    base_outputs: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Sliding contiguous layer-band joint ablation, per metric.

    For each width ``W`` (``<= len(layers)``) a band slides over the sorted
    configured ``layers``. The band's components — supplied by
    ``units_for_layers`` (one MLP unit per layer, or every head in each layer) —
    are ablated **jointly** in one forward pass, keyed by the band's start layer.

    With ``complement=False`` this ablates the band itself (the *necessity*
    sweep: "does behavior need this band?"). With ``complement=True`` it ablates
    every *other* configured layer instead, keeping the band (the *sufficiency*
    sweep: "can behavior survive on just this band?"). A full-width complement
    band has no layers left to ablate and is skipped.

    Returns ``{metric_name: {width_str: {band_start_str: drop}}}`` — the per-metric
    nesting the family block stores directly.
    """
    metric_names = list(metrics)
    out: dict[str, dict[str, dict[str, float]]] = {name: {} for name in metric_names}
    for width in widths:
        if width > len(layers):
            logger.info(
                "Skipping %s band width %d (> %d configured layers).",
                family,
                width,
                len(layers),
            )
            continue
        per_start: dict[str, dict[str, float]] = {name: {} for name in metric_names}
        for i in range(0, len(layers) - width + 1):
            band = layers[i : i + width]
            ablated_layers = (
                [layer for layer in layers if layer not in band] if complement else band
            )
            if not ablated_layers:  # complement of a full-width band: nothing left
                continue
            scored = run_ablation_combo_multi(
                units_for_layers(ablated_layers),
                dataset,
                pipeline,
                vectors,
                metrics=metrics,
                batch_size=batch_size,
                output_scores=True,
                original_outputs=base_outputs,
            )
            for name in metric_names:
                per_start[name][str(band[0])] = scored[name]
        for name in metric_names:
            out[name][str(width)] = per_start[name]
    return out


def _knockout_one_mode(
    *,
    ablation_mode: str,
    pipeline: LMPipeline,
    dataset: list[dict[str, Any]],
    metrics: dict[str, InterchangeMetric],
    base_outputs: list[dict[str, Any]],
    head_targets: dict | None,
    mlp_targets: dict | None,
    layers: list[int],
    heads: list[int],
    mlp_widths: list[int],
    head_band_widths: list[int],
    complement: bool,
    span_id: Any,
    batch_size: int,
    out_dir: str,
) -> dict[str, Any]:
    """Run every configured knockout for one ablation reference and persist it.

    Each cell is scored under every metric in ``metrics`` from a single set of
    generations; the per-family grids are nested under ``metrics`` so the web app
    can switch which one it renders. Both metrics are stored already oriented as
    drops (larger = more disruptive), so there is no per-metric transform here."""
    mode_dir = os.path.join(out_dir, ablation_mode)
    os.makedirs(mode_dir, exist_ok=True)
    metric_names = list(metrics)

    # One reference-vector dict covers every grid unit; scan/combo slice per target.
    grid_units = []
    if head_targets is not None:
        grid_units += [u for t in head_targets.values() for u in t.flatten()]
    if mlp_targets is not None:
        grid_units += [u for t in mlp_targets.values() for u in t.flatten()]
    vectors = _reference_vectors(
        ablation_mode, pipeline, dataset, grid_units, batch_size
    )

    def add_band_sweeps(
        family: str,
        family_metrics: dict[str, dict[str, Any]],
        units_for_layers: Callable[[list[int]], list[Any]],
        widths: list[int],
    ) -> None:
        """Attach the necessity ``widths`` (and, if enabled, the sufficiency
        ``complement_widths``) band sweeps to a family's per-metric block."""
        nec = _band_sweep(
            family=family,
            layers=layers,
            widths=widths,
            units_for_layers=units_for_layers,
            complement=False,
            pipeline=pipeline,
            dataset=dataset,
            vectors=vectors,
            metrics=metrics,
            base_outputs=base_outputs,
            batch_size=batch_size,
        )
        for name in metric_names:
            family_metrics[name]["widths"] = nec[name]
        if complement:
            comp = _band_sweep(
                family=family,
                layers=layers,
                widths=widths,
                units_for_layers=units_for_layers,
                complement=True,
                pipeline=pipeline,
                dataset=dataset,
                vectors=vectors,
                metrics=metrics,
                base_outputs=base_outputs,
                batch_size=batch_size,
            )
            for name in metric_names:
                family_metrics[name]["complement_widths"] = comp[name]

    components: dict[str, Any] = {}

    if head_targets is not None:
        # Fine-grained per-(layer, head) grid; full-vocab scores for logit_diff.
        scored = run_ablation_scan_multi(
            head_targets,
            dataset,
            pipeline,
            vectors,
            metrics=metrics,
            batch_size=batch_size,
            output_scores=True,
            original_outputs=base_outputs,
        )
        head_metrics: dict[str, dict[str, Any]] = {
            name: {"drop_grid": {f"{k[0]}|{k[1]}": scored[k][name] for k in scored}}
            for name in metric_names
        }
        # Whole-sublayer bands: every head in a layer band ablated jointly. A
        # width-1 band therefore knocks out a whole attention sublayer (all heads
        # in one layer) — coarser than the per-head grid above, and the direct
        # analogue of the MLP per-layer band.
        head_units_by_layer: dict[int, list[Any]] = {}
        for (layer, _head), target in head_targets.items():
            head_units_by_layer.setdefault(layer, []).extend(target.flatten())
        add_band_sweeps(
            "attention_head",
            head_metrics,
            lambda lyrs: [u for layer in lyrs for u in head_units_by_layer[layer]],
            head_band_widths,
        )
        components["attention_head"] = {
            "layers": layers,
            "heads": heads,
            "metrics": head_metrics,
        }

    if mlp_targets is not None:
        unit_by_layer = {
            layer: mlp_targets[(layer, span_id)].flatten()[0] for layer in layers
        }
        mlp_metrics: dict[str, dict[str, Any]] = {name: {} for name in metric_names}
        add_band_sweeps(
            "mlp",
            mlp_metrics,
            lambda lyrs: [unit_by_layer[layer] for layer in lyrs],
            mlp_widths,
        )
        components["mlp"] = {"layers": layers, "metrics": mlp_metrics}

    results = {
        "ablation_mode": ablation_mode,
        "span_id": span_id,
        "base_match": 1.0,  # base output trivially matches itself
        "n_inputs": len(dataset),
        "complement": complement,
        "metrics": {name: _METRIC_INFO[name] for name in metric_names},
        "components": components,
    }
    with open(os.path.join(mode_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("[knockout/%s] wrote %s/results.json", ablation_mode, mode_dir)
    return results


def run(pipeline: LMPipeline, acfg: DictConfig, out_dir: str) -> dict[str, Any]:
    """Run the knockout sweeps over the prompts in ``acfg.inputs``."""
    prompts, positions = _load_inputs(acfg.inputs)
    if not prompts:
        raise ValueError("knockout.inputs resolved to an empty prompt list")
    dataset = [{"input": _trace(p)} for p in prompts]

    batch_size = int(acfg.get("batch_size", 16))
    base_outputs = compute_base_outputs(dataset, pipeline, batch_size=batch_size)
    metrics = _build_metrics()
    span = _resolve_span(acfg, prompts, positions, pipeline)

    config = pipeline.model.config
    layers = (
        sorted(int(x) for x in acfg.layers)
        if acfg.get("layers") is not None
        else list(range(config.num_hidden_layers))
    )
    heads = (
        [int(x) for x in acfg.heads]
        if acfg.get("heads") is not None
        else list(range(config.num_attention_heads))
    )
    mlp_widths = [int(w) for w in acfg.get("mlp_widths", [1, 3, 5, 10])]
    head_band_widths = [int(w) for w in acfg.get("head_band_widths", [1, 3, 5, 10])]
    complement = bool(acfg.get("complement", False))

    requested = list(acfg.get("components", list(_COMPONENTS)))
    for comp in requested:
        if comp not in _COMPONENTS:
            raise ValueError(
                f"knockout.components entries must be in {_COMPONENTS}; got {comp!r}"
            )
    head_targets = (
        build_attention_head_targets(pipeline, layers, heads, span)
        if "attention_head" in requested
        else None
    )
    mlp_targets = (
        build_mlp_targets(pipeline, layers, [span]) if "mlp" in requested else None
    )

    ablation_modes = [str(m) for m in acfg.get("ablation_modes", list(_ABLATION_MODES))]
    for m in ablation_modes:
        if m not in _ABLATION_MODES:
            raise ValueError(
                f"knockout.ablation_modes entries must be in {_ABLATION_MODES}; got {m!r}"
            )

    mode_results = {
        m: _knockout_one_mode(
            ablation_mode=m,
            pipeline=pipeline,
            dataset=dataset,
            metrics=metrics,
            base_outputs=base_outputs,
            head_targets=head_targets,
            mlp_targets=mlp_targets,
            layers=layers,
            heads=heads,
            mlp_widths=mlp_widths,
            head_band_widths=head_band_widths,
            complement=complement,
            span_id=span.id,
            batch_size=batch_size,
            out_dir=out_dir,
        )
        for m in ablation_modes
    }

    metadata = {
        "analysis": "exploration",
        "mode": "knockout",
        "model": getattr(pipeline.model, "name_or_path", None),
        "span": acfg.get("span", "all"),
        "span_id": span.id,
        "components": requested,
        "ablation_modes": ablation_modes,
        "metrics": list(metrics),
        "complement": complement,
        "layers": layers,
        "heads": heads if "attention_head" in requested else None,
        "mlp_widths": mlp_widths,
        "head_band_widths": head_band_widths if "attention_head" in requested else None,
        "n_inputs": len(dataset),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[knockout] complete; modes=%s -> %s", ablation_modes, out_dir)
    return {"output_dir": out_dir, "metadata": metadata, "results": mode_results}
