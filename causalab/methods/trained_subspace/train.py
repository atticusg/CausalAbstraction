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

import collections
import copy
import logging
import random
from typing import Dict, Any, Callable, Union, Tuple

import numpy as np
import torch
from torch import Tensor
import transformers
from tqdm import tqdm

from causalab.causal.causal_model import CausalModel
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.featurizer import Featurizer
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.methods.metric import (
    LM_loss_and_metric_fn,
    causal_score_intervention_outputs,
)
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.activations.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.pipeline import Pipeline, LMPipeline
from causalab.neural.units import InterchangeTarget
from causalab.io.artifacts import (
    save_intervention_results,
    save_training_artifacts,
    save_aggregate_metadata,
)

logger = logging.getLogger(__name__)


def train_interventions(
    causal_model: CausalModel,
    interchange_targets: Union[
        Dict[Tuple[Any, ...], InterchangeTarget], InterchangeTarget
    ],
    train_dataset: list[CounterfactualExample],
    test_dataset: list[CounterfactualExample],
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    metric: Callable[[Any, Any], bool],
    config: dict[str, Any],
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
        train_dataset: In-memory training counterfactual examples
        test_dataset: In-memory test counterfactual examples
        pipeline: Target LMPipeline where interventions are applied
        target_variable_group: Tuple of target variable names to evaluate jointly
                              (e.g., ("answer",) or ("answer", "position"))
        metric: Function to compare neural output with expected output
        config: Fully-resolved training configuration dict. All keys must be
                present — callers merge with DEFAULT_CONFIG before calling.
                Must include: intervention_type, train_batch_size,
                evaluation_batch_size, training_epoch, init_lr, log_dir,
                featurizer_kwargs (for mask), DAS (for interchange), masking, etc.
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

    Raises:
        ValueError: If invalid intervention_type
        KeyError: If required config keys are missing
    """

    intervention_type = config["intervention_type"]

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

    # Initialize featurizers based on config (skips pre-initialized featurizers)
    _initialize_featurizers(interchange_targets, config)

    # Label training dataset
    labeled_train_dataset = causal_model.label_counterfactual_data(
        copy.deepcopy(train_dataset), list(target_variable_group)
    )

    results_by_key = {}
    eval_batch_size = config["evaluation_batch_size"]

    # Outer progress bar for all targets
    pbar = tqdm(
        interchange_targets.items(),
        desc="Training targets",
        disable=not logger.isEnabledFor(logging.DEBUG),
        total=len(interchange_targets),
    )

    loss_fn = lambda p, m, b, t, sp, sim: LM_loss_and_metric_fn(
        p, m, b, t, metric, source_pipeline=sp, source_intervenable_model=sim
    )

    for key, target in pbar:
        # Train this target (inner progress bar handled by training loop)
        _run_training_loop(
            pipeline=pipeline,
            interchange_target=target,
            counterfactual_dataset=labeled_train_dataset,  # type: ignore[arg-type]
            intervention_type=intervention_type,
            config=config,  # type: ignore[arg-type]
            loss_and_metric_fn=loss_fn,
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

    training_config: Dict[str, Any] = {
        "train_batch_size": config["train_batch_size"],
        "evaluation_batch_size": config["evaluation_batch_size"],
        "training_epoch": config["training_epoch"],
        "init_lr": config["init_lr"],
    }
    metadata: Dict[str, Any] = {
        "intervention_type": intervention_type,
        "target_variable_group": list(target_variable_group),
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
        training_config["regularization_coefficient"] = config["masking"][
            "regularization_coefficient"
        ]
        training_config["tie_masks"] = config["featurizer_kwargs"]["tie_masks"]
    else:
        training_config["n_features"] = config["DAS"]["n_features"]

    result = {
        "results_by_key": results_by_key,
        "avg_train_score": avg_train,
        "avg_test_score": avg_test,
        "metadata": metadata,
    }

    # Add intervention-specific results
    if intervention_type == "mask":
        result["num_selected_units"] = num_selected

    return result


def _initialize_featurizers(
    interchange_targets: Dict[Tuple[Any, ...], InterchangeTarget],
    config: dict[str, Any],
) -> None:
    """
    Initialize featurizers on all units based on config.

    Skips units that already have a non-placeholder featurizer (id != "null"),
    allowing pre-initialized featurizers (e.g., from PCA/SVD) to be preserved.

    For intervention_type="mask": Uses Featurizer with tie_masks
    For intervention_type="interchange": Uses SubspaceFeaturizer with n_features
    """
    intervention_type = config["intervention_type"]

    for _key, target in interchange_targets.items():
        for unit in target.flatten():
            # Skip units with pre-initialized featurizers
            if unit.featurizer.id != "null":
                continue

            unit_shape = unit.shape
            if unit_shape is None:
                raise ValueError(f"Unit {unit.id} has no shape defined")
            if intervention_type == "mask":
                tie_masks = config["featurizer_kwargs"]["tie_masks"]
                unit.set_featurizer(
                    Featurizer(
                        n_features=unit_shape[0],
                        tie_masks=tie_masks,
                        id=f"mask_{unit.id}",
                    )
                )
            else:  # "interchange" (DAS or subspace tracing)
                n_features = config["DAS"]["n_features"]
                unit.set_featurizer(
                    SubspaceFeaturizer(
                        shape=(unit_shape[0], n_features),
                        trainable=True,
                        id=f"DAS_{unit.id}",
                    )
                )


def _run_training_loop(
    pipeline: Pipeline,
    interchange_target: InterchangeTarget,
    counterfactual_dataset: list[CounterfactualExample],
    intervention_type: str,
    config: dict[str, Any],
    loss_and_metric_fn: Callable[..., tuple[Tensor, dict[str, Any], dict[str, Any]]],
    source_pipeline: Pipeline | None = None,
) -> str:
    """
    Train intervention models on a counterfactual dataset.

    This function implements the training loop for neural network interventions,
    supporting both "interchange" and "mask" intervention types. It optimizes
    intervention parameters while keeping the base model frozen.

    Args:
        pipeline: Target pipeline where interventions are applied
        interchange_target: InterchangeTarget containing model components to intervene on,
                           where groups share counterfactual inputs
        counterfactual_dataset: List of counterfactual examples
        intervention_type: Type of intervention ("interchange" or "mask")
        config: Configuration parameters including:
            - batch_size: Number of examples per batch
            - training_epoch: Maximum number of training epochs
            - init_lr: Initial learning rate
            - regularization_coefficient: Weight for sparsity regularization (mask only)
            - log_dir: Directory for TensorBoard logs
            - temperature_schedule: Start and end temperature for mask annealing
            - temperature_annealing_fraction: Fraction of training steps to anneal
            - patience: Epochs without improvement before early stopping
            - scheduler_type: Learning rate scheduler type
            - memory_cleanup_freq: Batch frequency for memory cleanup
            - shuffle: Whether to shuffle data
        loss_and_metric_fn: Function computing loss and metrics for a batch
                           with signature (pipeline, model, examples, units, source_pipeline) ->
                           (loss, metrics_dict, logging_info)
        source_pipeline: If provided, collect activations from this pipeline instead
            of the target pipeline during training. Enables cross-model patching.

    Returns:
        str: Summary string with final metrics
    """
    # ----- Model Initialization ----- #
    intervenable_model = prepare_intervenable_model(
        pipeline, interchange_target, intervention_type=intervention_type
    )
    intervenable_model.disable_model_gradients()
    intervenable_model.eval()

    # Create source intervenable model if doing cross-model patching
    source_intervenable_model = None
    if source_pipeline is not None:
        source_intervenable_model = prepare_intervenable_model(
            source_pipeline, interchange_target, intervention_type="collect"
        )

    # ----- Data Preparation ----- #
    train_batch_size = config["train_batch_size"]
    shuffle = config["shuffle"]
    num_batches = -(-len(counterfactual_dataset) // train_batch_size)

    # ----- Configuration ----- #
    num_epoch = config["training_epoch"]
    regularization_coefficient = config["masking"]["regularization_coefficient"]
    memory_cleanup_freq = config["memory_cleanup_freq"]
    patience = config["patience"]
    scheduler_type = config["scheduler_type"]

    # ----- Early Stopping Setup ----- #
    best_loss = float("inf")
    patience_counter = 0
    early_stopping_enabled = patience is not None

    # ----- Optimizer Configuration ----- #
    optimizer_params = []
    for k, v in intervenable_model.interventions.items():
        optimizer_params += list(v.parameters())

    optimizer = torch.optim.AdamW(
        optimizer_params, lr=config["init_lr"], weight_decay=0
    )

    scheduler = transformers.get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_epoch * num_batches,
    )

    # Track step count manually instead of accessing scheduler._step_count
    current_step = 0

    # ----- Temperature Scheduling for Mask Interventions ----- #
    temperature_schedule = None
    interventions = None
    if intervention_type == "mask":
        temperature_start, temperature_end = config["masking"]["temperature_schedule"]
        temperature_annealing_fraction = config["masking"][
            "temperature_annealing_fraction"
        ]

        # Calculate number of steps for annealing
        total_steps = num_epoch * num_batches
        annealing_steps = int(total_steps * temperature_annealing_fraction)

        # Create schedule: anneal for first fraction of steps, then stay constant
        annealing_schedule = torch.linspace(
            temperature_start, temperature_end, annealing_steps + 1
        )
        constant_schedule = torch.full(
            (total_steps - annealing_steps,), temperature_end
        )
        temperature_schedule = torch.cat([annealing_schedule, constant_schedule])
        temperature_schedule = temperature_schedule.to(pipeline.model.dtype).to(
            pipeline.model.device
        )

        # Set initial temperature for all mask interventions
        interventions = intervenable_model.interventions
        assert interventions is not None
        for k, v in interventions.items():
            # pyvene's intervention dict values are dynamically typed
            init_intervention = v[0] if isinstance(v, tuple) else v
            init_intervention.set_temperature(temperature_schedule[current_step])  # pyright: ignore[reportCallIssue]

    # ----- Training Loop ----- #
    postfix_dict: dict[str, str] = {}  # Initialize to avoid unbound error
    train_iterator = tqdm(
        range(0, int(num_epoch)),
        desc=f"Training {str(interchange_target)[:100]}...",
        leave=False,
    )
    for epoch in train_iterator:
        # Shuffle indices for this epoch if requested
        indices = list(range(len(counterfactual_dataset)))
        if shuffle:
            random.shuffle(indices)

        aggregated_stats = collections.defaultdict(list)

        epoch_iterator = tqdm(
            range(0, len(indices), train_batch_size),
            desc=f"Epoch: {epoch}",
            position=1,
            leave=False,
        )
        for step, start in enumerate(epoch_iterator):
            examples = [
                counterfactual_dataset[i]
                for i in indices[start : start + train_batch_size]
            ]

            # Run training step
            loss, eval_metrics, _logging_info = loss_and_metric_fn(
                pipeline,
                intervenable_model,
                examples,
                interchange_target,
                source_pipeline,
                source_intervenable_model,
            )

            # Add sparsity loss for mask interventions
            if intervention_type == "mask":
                assert temperature_schedule is not None
                assert interventions is not None
                temp = temperature_schedule[current_step]

                # Collect sparsity losses and mask sizes for normalization
                total_sparsity: Tensor = torch.tensor(0.0, device=loss.device)
                total_mask_elements = 0
                for k, v in interventions.items():
                    # pyvene's intervention dict values are dynamically typed
                    mask_intervention = v[0] if isinstance(v, tuple) else v
                    total_sparsity = (
                        total_sparsity + mask_intervention.get_sparsity_loss()  # pyright: ignore[reportCallIssue]
                    )
                    total_mask_elements += mask_intervention.mask.numel()  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
                    mask_intervention.set_temperature(temp)  # pyright: ignore[reportCallIssue]

                # Normalize by total mask elements so regularization_coefficient
                # has consistent meaning regardless of number of features/units
                if total_mask_elements > 0:
                    loss = loss + regularization_coefficient * (
                        total_sparsity / total_mask_elements
                    )

            # Update statistics
            aggregated_stats["loss"].append(loss.item())
            aggregated_stats["metrics"].append(eval_metrics)

            # Update progress bar
            postfix = {"loss": round(np.mean(aggregated_stats["loss"]), 2)}
            for k, v in eval_metrics.items():
                postfix[k] = round(np.mean(v), 2)
            epoch_iterator.set_postfix(postfix)

            # Optimization step
            loss.backward()
            optimizer.step()
            # get_scheduler's return type includes ReduceLROnPlateau which needs metrics,
            # but we only use schedulers that don't require it
            scheduler.step()  # pyright: ignore[reportCallIssue]
            current_step += 1
            intervenable_model.set_zero_grad()

            # Periodic memory cleanup
            if step % memory_cleanup_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update progress bar with epoch summary
        epoch_avg_loss = np.mean(aggregated_stats["loss"])
        postfix_dict = {"loss": f"{epoch_avg_loss:.4f}"}

        if aggregated_stats["metrics"]:
            # Aggregate metrics across all batches in the epoch
            all_metrics = {}
            for batch_metrics in aggregated_stats["metrics"]:
                for k, v in batch_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
            # Add metrics to postfix
            for k, v in all_metrics.items():
                postfix_dict[k] = f"{np.mean(v):.4f}"

        train_iterator.set_postfix(postfix_dict)

        # Early stopping check at end of epoch
        if early_stopping_enabled:
            epoch_avg_loss = np.mean(aggregated_stats["loss"])
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # ----- Feature Selection for Mask Interventions ----- #
    if intervention_type == "mask":
        # Flatten InterchangeTarget to get all model units
        model_units = interchange_target.flatten()
        assert intervenable_model.interventions is not None

        for kv, model_unit in zip(
            intervenable_model.interventions.items(),
            model_units,
        ):
            k, v = kv
            # pyvene's intervention dict values are dynamically typed
            intervention = v[0] if isinstance(v, tuple) else v

            if config["featurizer_kwargs"]["tie_masks"]:
                # If masks are tied, use the average mask across all units
                if torch.sigmoid(intervention.mask[0]) > 0.5:  # pyright: ignore[reportIndexIssue,reportArgumentType]
                    indices = None
                else:
                    indices = []
            else:
                # Get binary mask and indices
                mask_binary = (torch.sigmoid(intervention.mask) > 0.5).float().cpu()  # pyright: ignore[reportArgumentType]
                indices = torch.nonzero(mask_binary).numpy().flatten().tolist()

            # Update model unit
            model_unit.set_feature_indices(indices)

    # ----- Cleanup ----- #
    delete_intervenable_model(intervenable_model)
    if source_intervenable_model is not None:
        delete_intervenable_model(source_intervenable_model)

    summary = f"Trained intervention for {str(interchange_target)[:200]}"
    summary += "\nFinal metrics: " + " ".join(
        [f"{k}: {v}" for k, v in postfix_dict.items()]
    )
    return summary


def save_train_results(result: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Save train_interventions results to disk."""
    results_by_key = result["results_by_key"]
    metadata = result["metadata"]
    output_paths: Dict[str, str] = {}

    train_eval_paths = save_intervention_results(
        {k: v["train_eval"] for k, v in results_by_key.items()},
        output_dir=output_dir,
        prefix="train_eval",
    )
    output_paths.update({f"train_{k}": v for k, v in train_eval_paths.items()})

    test_eval_paths = save_intervention_results(
        {k: v["test_eval"] for k, v in results_by_key.items()},
        output_dir=output_dir,
        prefix="test_eval",
    )
    output_paths.update({f"test_{k}": v for k, v in test_eval_paths.items()})

    training_paths = save_training_artifacts(results_by_key, output_dir=output_dir)
    output_paths.update(training_paths)

    metadata_path = save_aggregate_metadata(metadata, output_dir)
    output_paths["metadata_path"] = metadata_path

    return output_paths
