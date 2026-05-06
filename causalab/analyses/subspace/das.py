"""DAS (Distributed Alignment Search) subspace discovery.

Trains a linear subspace via interchange interventions, then collects
features through the learned featurizer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from safetensors.torch import save_file

from causalab.causal.causal_model import CausalModel
from causalab.analyses.subspace._visualization import save_features_visualization
from causalab.neural.units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline
from causalab.neural.activations.collect import collect_features

logger = logging.getLogger(__name__)


def find_das_subspace(
    target: InterchangeTarget,
    train_dataset: list,
    test_dataset: list,
    pipeline: LMPipeline,
    causal_model: CausalModel,
    k_features: int,
    batch_size: int = 32,
    output_dir: str = "",
    metric: Callable | None = None,
    loss_config: dict | None = None,
    intervention_variable: str | None = None,
    embeddings: dict[str, Callable] | None = None,
    target_variable_group: tuple[str, ...] = ("raw_output",),
    colormap: str | None = None,
    variable_values: list[str] | None = None,
    detailed_hover: bool = False,
    max_hover_chars: int = 50,
    figure_format: str = "pdf",
) -> dict[str, Any]:
    """Train a DAS subspace and collect features through it.

    Calls ``train_interventions`` directly (no grid/heatmap) to learn a
    linear subspace, then runs a forward pass to collect projected features.

    Args:
        target: Interchange target (single unit).
        train_dataset: Training counterfactual examples.
        test_dataset: Test counterfactual examples.
        pipeline: LM pipeline.
        causal_model: Causal model for intervention training.
        k_features: Subspace dimension.
        batch_size: Batch size.
        output_dir: Where to save DAS artifacts + features.
        metric: Intervention success metric.
        loss_config: DAS training config overrides (training_epoch, init_lr).
        intervention_variable: Variable name for visualization coloring.
        embeddings: Embedding functions for visualization.
        target_variable_group: Variable group for training.

    Returns:
        Dict with keys:
        - ``das_result``: training result dict from ``train_interventions``
        - ``features``: (N, k) projected features tensor
    """
    from causalab.methods.trained_subspace.train import train_interventions
    from causalab.methods.trained_subspace.train import save_train_results
    from causalab.configs.train_config import merge_with_defaults

    loss_config = loss_config or {}
    das_config = merge_with_defaults(
        {
            "intervention_type": "interchange",
            "DAS": {"n_features": k_features},
            "train_batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            **loss_config,
        }
    )

    das_dir = os.path.join(output_dir, "das") if output_dir else "das"
    das_config["log_dir"] = os.path.join(das_dir, "logs")
    os.makedirs(das_config["log_dir"], exist_ok=True)

    # Materialize datasets in memory (accept either list or path for back-compat)
    train_examples = (
        _load_dataset(train_dataset, causal_model)
        if isinstance(train_dataset, str)
        else train_dataset
    )
    test_examples = (
        _load_dataset(test_dataset, causal_model)
        if isinstance(test_dataset, str)
        else test_dataset
    )

    das_result = train_interventions(
        causal_model=causal_model,
        interchange_targets=target,
        train_dataset=train_examples,
        test_dataset=test_examples,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        config=das_config,
        metric=metric,
    )
    save_train_results(das_result, das_dir)
    logger.info(
        "DAS training complete. Test score: %s",
        das_result.get("avg_test_score", "N/A"),
    )

    # Collect features through the learned featurizer
    unit = target.flatten()[0]
    features_dict = collect_features(
        dataset=train_examples,
        pipeline=pipeline,
        model_units=[unit],
        batch_size=batch_size,
    )
    features = features_dict[unit.id].detach()
    logger.info("Collected %d-dim features: %s", k_features, features.shape)

    if output_dir:
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        save_file(
            {"features": features.contiguous()},
            os.path.join(features_dir, "training_features.safetensors"),
        )

        save_features_visualization(
            features,
            train_examples,
            output_dir,
            intervention_variable,
            embeddings,
            colormap=colormap,
            variable_values=variable_values,
            detailed_hover=detailed_hover,
            max_hover_chars=max_hover_chars,
            figure_format=figure_format,
        )

    return {
        "das_result": das_result,
        "features": features,
    }


def _load_dataset(path: str, causal_model: CausalModel) -> list:
    from causalab.io.counterfactuals import load_counterfactual_examples

    return load_counterfactual_examples(path, causal_model)
