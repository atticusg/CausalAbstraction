"""
Generic Intervention Training Script (DBM and DAS)

This module provides a generic function to train interventions on any InterchangeTarget.
Supports both:
- DBM (Desiderata-Based Masking): Learns binary masks over units
- DAS (Distributed Alignment Search): Learns linear subspaces containing causal variables

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── models/                     # Trained models per key
│   ├── 0__first_token/
│   ├── 0__last_token/
│   └── ...
├── training/                   # Training-specific artifacts
│   └── feature_indices.json
├── train_eval/                 # Training set evaluation
│   ├── scores.json
│   └── raw_results.json
└── test_eval/                  # Test set evaluation
    ├── scores.json
    └── raw_results.json
"""

import copy
import logging
import os
from typing import Dict, Any, Callable, Union, Tuple

from tqdm import tqdm

from causalab.neural.pipeline import LMPipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.featurizers import Featurizer
from causalab.neural import SubspaceFeaturizer
from causalab.neural.pyvene_core.interchange import (
    train_interventions as train_interventions_pyvene,
    run_interchange_interventions,
)
from causalab.experiments.configs.train_config import (
    ExperimentConfig,
    ExperimentConfigInput,
    merge_with_defaults,
)
from causalab.causal.causal_model import CausalModel
from causalab.experiments.metric import (
    LM_loss_and_metric_fn,
    causal_score_intervention_outputs,
)
from causalab.experiments.io import (
    save_intervention_results,
    save_training_artifacts,
    save_aggregate_metadata,
)
from causalab.causal.causal_utils import load_counterfactual_examples

logger = logging.getLogger(__name__)


def train_interventions(
    causal_model: CausalModel,
    interchange_targets: Union[
        Dict[Tuple[Any, ...], InterchangeTarget], InterchangeTarget
    ],
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    config: ExperimentConfigInput = None,
    save_results: bool = True,
    source_pipeline: LMPipeline | None = None,
) -> Dict[str, Any]:
    """
    Train interventions (DBM or DAS) on one or more InterchangeTargets.

    Handles featurizer initialization based on config:
    - intervention_type="mask": Uses Featurizer with tie_masks from config
    - intervention_type="interchange": Uses SubspaceFeaturizer with n_features from config

    This function trains interventions, then evaluates using run_interventions() on both
    train and test datasets.

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_targets: Either a dict mapping keys to targets, or a single target
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: Target LMPipeline where interventions are applied
        target_variable_group: Tuple of target variable names to evaluate jointly
                              (e.g., ("answer",) or ("answer", "position"))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        config: Training configuration dict. Should contain:
                - intervention_type: "mask" for DBM, "interchange" for DAS (default: "mask")
                - featurizer_kwargs: {"tie_masks": bool} for mask interventions (required for DBM)
                - DAS: {"n_features": int} for interchange interventions (default: 32)
                - train_batch_size: batch size for training
                (default: DEFAULT_CONFIG with appropriate settings)
        save_results: Whether to save metadata and models to disk (default: True)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline. Enables cross-model patching where you train
            to find features in pipeline that align with activations from
            source_pipeline.

    Note:
        Units with pre-initialized featurizers (id != "null") are preserved.
        This allows using PCA/SVD-initialized featurizers without overwriting them.

    Returns:
        Dictionary containing:
            - results_by_key: dict mapping keys to per-target results with:
                - train_score: training accuracy
                - test_score: test accuracy
                - feature_indices: feature indices dict
                - train_eval: full train evaluation results
                - test_eval: full test evaluation results
            - avg_train_score: average training accuracy across targets
            - avg_test_score: average test accuracy across targets
            - metadata: experiment configuration and summary
            - output_paths: paths to saved files and directories

    Raises:
        ValueError: If invalid intervention_type
        FileNotFoundError: If dataset paths do not exist
    """

    # Setup configuration - merge with defaults
    config = merge_with_defaults(config)

    intervention_type = config.get("intervention_type", "mask")

    if intervention_type not in ["mask", "interchange"]:
        raise ValueError(
            f"Invalid intervention_type: {intervention_type}. "
            f"Must be 'mask' (DBM) or 'interchange' (DAS)."
        )

    # Validate required config for DBM
    if intervention_type == "mask" and "featurizer_kwargs" not in config:
        raise ValueError(
            "config['featurizer_kwargs'] is required for mask interventions. "
            "Set config['featurizer_kwargs'] = {'tie_masks': True} for one mask per unit, "
            "or {'tie_masks': False} for separate masks per feature dimension."
        )

    # Wrap single target in dict
    if isinstance(interchange_targets, InterchangeTarget):
        interchange_targets = {("single",): interchange_targets}

    # Load training dataset and deserialize CausalTraces
    train_dataset = load_counterfactual_examples(train_dataset_path, causal_model)

    # Ensure log_dir is set
    config["log_dir"] = os.path.join(output_dir, "logs")
    os.makedirs(config["log_dir"], exist_ok=True)

    # Initialize featurizers based on config (skips pre-initialized featurizers)
    _initialize_featurizers(interchange_targets, config)

    # Load test dataset and deserialize CausalTraces
    test_dataset = load_counterfactual_examples(test_dataset_path, causal_model)

    # Label training dataset
    labeled_train_dataset = causal_model.label_counterfactual_data(
        copy.deepcopy(train_dataset), list(target_variable_group)
    )

    results_by_key = {}
    eval_batch_size = config.get("evaluation_batch_size", 32)

    # Outer progress bar for all targets
    pbar = tqdm(
        interchange_targets.items(),
        desc="Training targets",
        disable=not logger.isEnabledFor(logging.DEBUG),
        total=len(interchange_targets),
    )

    for key, target in pbar:
        # Train this target (inner progress bar handled by training loop)
        train_interventions_pyvene(
            pipeline=pipeline,
            interchange_target=target,
            counterfactual_dataset=labeled_train_dataset,  # type: ignore[arg-type]
            intervention_type=intervention_type,
            config=config,  # type: ignore[arg-type]
            loss_and_metric_fn=lambda p, m, b, t, sp, sim: LM_loss_and_metric_fn(
                p, m, b, t, metric, source_pipeline=sp, source_intervenable_model=sim
            ),
            source_pipeline=source_pipeline,
        )

        # Get feature indices
        feature_indices = target.get_feature_indices()

        # Run interventions on train data
        train_raw_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=train_dataset,
                interchange_target=target,
                batch_size=eval_batch_size,
                output_scores=False,
                source_pipeline=source_pipeline,
            )
        }

        # Score train results
        train_eval = causal_score_intervention_outputs(
            raw_results=train_raw_results,
            dataset=train_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )

        # Run interventions on test data
        test_raw_results = {
            key: run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=test_dataset,
                interchange_target=target,
                batch_size=eval_batch_size,
                output_scores=False,
                source_pipeline=source_pipeline,
            )
        }

        # Score test results
        test_eval = causal_score_intervention_outputs(
            raw_results=test_raw_results,
            dataset=test_dataset,
            causal_model=causal_model,
            target_variable_groups=[target_variable_group],
            metric=metric,
        )

        results_by_key[key] = {
            "train_score": train_eval["results_by_key"][key]["avg_score"],
            "test_score": test_eval["results_by_key"][key]["avg_score"],
            "feature_indices": feature_indices,
            "train_eval": train_eval["results_by_key"][key],
            "test_eval": test_eval["results_by_key"][key],
            "trained_target": target,  # For model saving
        }

    pbar.close()

    # Compute averages
    avg_train = float(
        sum(r["train_score"] for r in results_by_key.values()) / len(results_by_key)
    )
    avg_test = float(
        sum(r["test_score"] for r in results_by_key.values()) / len(results_by_key)
    )

    # Count selected units/features (for DBM)
    num_selected = sum(
        1
        for result in results_by_key.values()
        for indices in result["feature_indices"].values()
        if indices and len(indices) > 0
    )

    # Create metadata
    experiment_name = "DBM" if intervention_type == "mask" else "DAS"
    model_name = getattr(pipeline, "model_or_name", None)

    training_config: Dict[str, Any] = {
        "train_batch_size": config.get("train_batch_size"),
        "evaluation_batch_size": config.get("evaluation_batch_size"),
        "training_epoch": config.get("training_epoch"),
        "init_lr": config.get("init_lr"),
    }
    metadata: Dict[str, Any] = {
        "experiment_type": experiment_name,
        "intervention_type": intervention_type,
        "model": model_name,
        "train_dataset_path": train_dataset_path,
        "test_dataset_path": test_dataset_path,
        "target_variable_group": target_variable_group,
        "num_train_examples": len(train_dataset),
        "num_test_examples": len(test_dataset),
        "num_targets": len(interchange_targets),
        "avg_train_score": float(avg_train),
        "avg_test_score": float(avg_test),
        "training_config": training_config,
    }

    # Add intervention-specific metadata
    if intervention_type == "mask":
        metadata["num_selected_units"] = num_selected
        training_config["regularization_coefficient"] = config.get("masking", {}).get(
            "regularization_coefficient"
        )
        training_config["tie_masks"] = config.get("featurizer_kwargs", {}).get(
            "tie_masks"
        )
    else:
        training_config["n_features"] = config.get("DAS", {}).get("n_features")

    # Save results
    output_paths = {}
    if save_results:
        # Save train evaluation results
        train_eval_paths = save_intervention_results(
            {k: v["train_eval"] for k, v in results_by_key.items()},
            output_dir=output_dir,
            prefix="train_eval",
        )
        output_paths.update({f"train_{k}": v for k, v in train_eval_paths.items()})

        # Save test evaluation results
        test_eval_paths = save_intervention_results(
            {k: v["test_eval"] for k, v in results_by_key.items()},
            output_dir=output_dir,
            prefix="test_eval",
        )
        output_paths.update({f"test_{k}": v for k, v in test_eval_paths.items()})

        # Save training artifacts (feature indices and models)
        training_paths = save_training_artifacts(
            results_by_key,
            output_dir=output_dir,
        )
        output_paths.update(training_paths)

        # Save metadata
        metadata_path = save_aggregate_metadata(metadata, output_dir)
        output_paths["metadata_path"] = metadata_path

    result = {
        "results_by_key": results_by_key,
        "avg_train_score": avg_train,
        "avg_test_score": avg_test,
        "metadata": metadata,
        "output_paths": output_paths,
    }

    # Add intervention-specific results
    if intervention_type == "mask":
        result["num_selected_units"] = num_selected

    return result


def _initialize_featurizers(
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
    config: ExperimentConfig,
) -> None:
    """
    Initialize featurizers on all units based on config.

    Skips units that already have a non-placeholder featurizer (id != "null"),
    allowing pre-initialized featurizers (e.g., from PCA/SVD) to be preserved.

    For intervention_type="mask": Uses Featurizer with tie_masks
    For intervention_type="interchange": Uses SubspaceFeaturizer with n_features
    """
    intervention_type = config.get("intervention_type", "mask")

    for _key, target in interchange_targets.items():
        for unit in target.flatten():
            # Skip units with pre-initialized featurizers
            if unit.featurizer.id != "null":
                continue

            unit_shape = unit.shape
            if unit_shape is None:
                raise ValueError(f"Unit {unit.id} has no shape defined")
            if intervention_type == "mask":
                tie_masks = config.get("featurizer_kwargs", {}).get("tie_masks", True)
                unit.set_featurizer(
                    Featurizer(
                        n_features=unit_shape[0],
                        tie_masks=tie_masks,
                        id=f"mask_{unit.id}",
                    )
                )
            else:  # "interchange" (DAS or subspace tracing)
                n_features = config.get("DAS", {}).get("n_features", 32)
                unit.set_featurizer(
                    SubspaceFeaturizer(
                        shape=(unit_shape[0], n_features),
                        trainable=True,
                        id=f"DAS_{unit.id}",
                    )
                )
