"""Single-example residual stream tracing via interchange interventions."""

import logging
from typing import Dict, Any, List

from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import TokenPosition
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.activations.targets import build_residual_stream_targets

logger = logging.getLogger(__name__)


def run_residual_stream_tracing(
    pipeline: LMPipeline,
    prompt: str,
    counterfactual_prompt: str,
    token_positions: List[TokenPosition],
    layers: List[int] | None = None,
    verbose: bool = True,
    source_pipeline: LMPipeline | None = None,
) -> Dict[str, Any]:
    """Trace information flow through residual stream interventions.

    Runs interchange interventions at each (layer, token_position) combination
    using a single prompt/counterfactual pair.  Returns raw intervention outputs
    (no scoring or metric).

    Args:
        pipeline: Target LMPipeline where interventions are applied.
        prompt: Original prompt text.
        counterfactual_prompt: Counterfactual prompt text.
        token_positions: List of TokenPosition objects to analyze.
        layers: Specific layers to analyze (default: all layers including -1).
        verbose: Whether to print progress information.
        source_pipeline: If provided, collect activations from this pipeline
            instead of the target pipeline.  Enables cross-model patching.

    Returns:
        Dictionary containing:
            - intervention_results: raw intervention outputs keyed by (layer, pos_id)
            - metadata: experiment configuration
    """
    if verbose:
        logging.getLogger("causalab").setLevel(logging.DEBUG)

    counterfactual_dataset: list[CounterfactualExample] = [  # type: ignore[assignment]
        {
            "input": {"raw_input": prompt},
            "counterfactual_inputs": [{"raw_input": counterfactual_prompt}],
        }
    ]

    num_layers = pipeline.model.config.num_hidden_layers
    if layers is None:
        layers = [-1] + list(range(num_layers))

    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    intervention_results = {}
    for (layer, pos_id), target in tqdm(
        targets.items(), desc="Running interventions", disable=not verbose
    ):
        outputs = run_interchange_interventions(
            pipeline=pipeline,
            counterfactual_dataset=counterfactual_dataset,
            interchange_target=target,
            batch_size=1,
            output_scores=False,
            source_pipeline=source_pipeline,
        )
        intervention_results[(layer, pos_id)] = outputs

    model_name = getattr(pipeline, "model_or_name", None)
    token_position_ids = [pos.id for pos in token_positions]
    metadata = {
        "experiment_type": "tracing",
        "model": model_name,
        "prompt": prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "num_layers": num_layers,
        "layers_used": layers,
        "num_token_positions": len(token_positions),
        "token_position_ids": token_position_ids,
        "total_interventions": len(layers) * len(token_positions),
    }

    return {
        "intervention_results": intervention_results,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Save helpers                                                                #
# --------------------------------------------------------------------------- #

import os as _os  # noqa: E402
import torch as _torch  # noqa: E402
from causalab.io.artifacts import (  # noqa: E402
    save_experiment_metadata as _save_experiment_metadata,
    save_json_results as _save_json_results,
    save_tensor_results as _save_tensor_results,
)
from causalab.io.plots.string_heatmap import (  # noqa: E402
    plot_residual_stream_intervention_heatmap as _plot_rs_heatmap,
)


def save_residual_stream_results(
    result: Dict[str, Any],
    output_dir: str,
    generate_visualization: bool = True,
    pipeline: LMPipeline | None = None,
    token_positions: List[TokenPosition] | None = None,
    prompt: str | None = None,
    layers: List[int] | None = None,
) -> Dict[str, str]:
    """Save run_residual_stream_tracing results: JSON + safetensors + heatmap."""
    intervention_results = result["intervention_results"]
    metadata = result["metadata"]
    output_paths: Dict[str, str] = {}

    results_dir = _os.path.join(output_dir, "results")
    _os.makedirs(results_dir, exist_ok=True)

    results_json: Dict[str, Any] = {}
    results_tensors: Dict[str, Any] = {}
    for (layer, pos), res in intervention_results.items():
        key = f"{layer}__{pos}"
        results_json[key] = {"string": res.get("string", [])}
        if "sequences" in res and res["sequences"]:
            results_tensors[key] = _torch.cat(res["sequences"], dim=0)

    output_paths["results_json_path"] = _save_json_results(
        data=results_json,
        output_dir=results_dir,
        filename="intervention_results.json",
    )
    if results_tensors:
        output_paths["results_tensors_path"] = _save_tensor_results(
            tensors=results_tensors,
            output_dir=results_dir,
            filename="intervention_results.safetensors",
        )
    output_paths["results_dir"] = results_dir

    if generate_visualization and pipeline is not None and token_positions is not None:
        heatmap_path = _os.path.join(output_dir, "residual_stream_heatmap.png")
        _plot_rs_heatmap(
            intervention_results=intervention_results,
            prompt=prompt or "",
            layers=layers or sorted(set(k[0] for k in intervention_results)),
            token_positions=token_positions,
            pipeline=pipeline,
            correct_answer=None,
            title="Residual Stream Tracing: Output Tokens After Intervention",
            save_path=heatmap_path,
            color_by_frequency=True,
        )
        output_paths["heatmap_path"] = heatmap_path

    output_paths["metadata_path"] = _save_experiment_metadata(
        metadata=metadata,
        output_dir=output_dir,
    )
    return output_paths
