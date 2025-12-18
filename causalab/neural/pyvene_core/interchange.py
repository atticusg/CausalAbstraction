"""
interchange.py
==============
Core utilities for running interchange intervention experiments.

This module provides functions for running interventions on neural networks using
the pyvene library. It focuses on interchange interventions where activations are
swapped between base and counterfactual inputs, as well as training intervention
models like DAS and DBM.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Callable

import torch
from torch import Tensor
import transformers
import numpy as np
from tqdm import tqdm
from pyvene import IntervenableModel

from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.neural.pipeline import Pipeline
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pyvene_core.intervenable_model import (
    prepare_intervenable_model,
    delete_intervenable_model,
)
from torch.utils.data import DataLoader

# Configure logging
logger = logging.getLogger(__name__)


def prepare_intervenable_inputs(
    pipeline: Pipeline,
    batch: dict[str, Any],
    interchange_target: InterchangeTarget,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[list[list[int] | None]],
]:
    """
    Prepare the inputs for the intervenable model.

    This function loads the base and counterfactual inputs, and prepares the indices
    for the model units.

    Args:
        pipeline: The pipeline containing the model
        batch: The batch of data containing the base and counterfactual inputs.
            The batch should contain "input" and "counterfactual_inputs" keys.
            The "counterfactual_inputs" key should contain a list of lists with shape,
            (batch_size, num_counterfactuals).
        interchange_target: InterchangeTarget containing the model units to be intervened on.
            Groups in the target correspond to counterfactual inputs.

    Returns:
        Tuple of (batched_base, batched_counterfactuals, inv_locations, feature_indices)
    """
    batched_base = batch["input"]
    # Change the shape of the counterfactual inputs from (batch_size, num_counterfactuals)
    # to (num_counterfactuals, batch_size)
    batched_counterfactuals = list(zip(*batch["counterfactual_inputs"]))

    # shape: (num_model_units, batch_size, num_component_indices)
    base_indices = [
        model_unit.index_component(batched_base, batch=True, is_original=True)
        for group in interchange_target
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_component_indices)
    counterfactual_indices = [
        model_unit.index_component(
            batched_counterfactual, batch=True, is_original=False
        )
        for group, batched_counterfactual in zip(
            interchange_target, batched_counterfactuals
        )
        for model_unit in group
    ]

    # shape: (num_model_units, batch_size, num_feature_indices)
    feature_indices = [
        [model_unit.get_feature_indices() for _ in range(len(batched_base))]
        for group in interchange_target
        for model_unit in group
    ]

    batched_base = pipeline.load(batched_base)
    batched_counterfactuals = [
        pipeline.load(batched_counterfactual)
        for batched_counterfactual in batched_counterfactuals
    ]

    inv_locations = {"sources->base": (counterfactual_indices, base_indices)}
    return batched_base, batched_counterfactuals, inv_locations, feature_indices


def batched_interchange_intervention(
    pipeline: Pipeline,
    intervenable_model: IntervenableModel,
    batch: dict[str, Any],
    interchange_target: InterchangeTarget,
    output_scores: bool | int = True,
) -> dict[str, Any]:
    """
    Perform interchange interventions on batched inputs using an intervenable model.

    This function executes the core intervention logic by:
    1. Preparing the base and counterfactual inputs for intervention
    2. Running the model with interventions at specified locations
    3. Moving tensors back to CPU to free GPU memory

    Args:
        pipeline: Neural model pipeline that handles tokenization and generation
        intervenable_model: PyVENE model with preset intervention locations
        batch: Batch of data containing "input" and "counterfactual_inputs"
        interchange_target: InterchangeTarget containing model components to intervene on
        output_scores: Whether to include scores in output dictionary (default: True)

    Returns:
        dict: Dictionary with 'sequences' and optionally 'scores' keys
    """
    # Prepare inputs for intervention
    batched_base, batched_counterfactuals, inv_locations, feature_indices = (
        prepare_intervenable_inputs(pipeline, batch, interchange_target)
    )

    # Execute the intervention via the pipeline
    gen_kwargs = {"output_scores": output_scores}
    output = pipeline.intervenable_generate(
        intervenable_model,
        batched_base,
        batched_counterfactuals,
        inv_locations,
        feature_indices,
        **gen_kwargs,
    )

    # Move tensors to CPU to free GPU memory
    for batched in [batched_base] + batched_counterfactuals:
        for k, v in batched.items():
            if isinstance(v, Tensor):
                batched[k] = v.cpu()

    return output


def _convert_to_top_k(
    outputs: list[dict[str, Any]], pipeline: Pipeline, k: int
) -> list[dict[str, Any]]:
    """
    Convert full vocabulary logits to top-k format to reduce memory usage.

    This processes outputs to extract only the top-k logits, indices, and tokens,
    dramatically reducing memory footprint (e.g., from ~256K to 10 values per token).

    Args:
        outputs: List of output dictionaries with 'scores' (list of tensors on GPU)
        pipeline: Pipeline with tokenizer for decoding tokens
        k: Number of top logits to keep (must be > 0)

    Returns:
        Modified outputs where 'scores' contains top-k structured data
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    converted_outputs = []
    for batch_dict in outputs:
        converted_batch = {
            "sequences": batch_dict.get("sequences"),
            "string": batch_dict.get("string"),
        }

        # Convert scores to top-k format if present
        scores = batch_dict.get("scores")
        if scores:
            top_k_scores = []
            for position_logits in scores:
                # position_logits shape: (batch_size, vocab_size)
                vocab_size = position_logits.shape[1]
                k_actual = min(k, vocab_size)

                # Get top-k values and indices for entire batch
                top_k_values, top_k_indices = torch.topk(
                    position_logits, k=k_actual, dim=1
                )

                # Decode tokens for entire batch - flatten and batch decode for efficiency
                batch_size = position_logits.shape[0]
                flat_indices = top_k_indices.flatten().tolist()
                flat_tokens = pipeline.tokenizer.batch_decode(
                    [[idx] for idx in flat_indices], skip_special_tokens=False
                )

                # Reshape back to (batch_size, k)
                top_k_tokens = [
                    flat_tokens[i * k_actual : (i + 1) * k_actual]
                    for i in range(batch_size)
                ]

                top_k_scores.append(
                    {
                        "top_k_logits": top_k_values,  # Tensor[batch, k]
                        "top_k_indices": top_k_indices,  # Tensor[batch, k]
                        "top_k_tokens": top_k_tokens,  # List[batch][k]
                    }
                )

            converted_batch["scores"] = top_k_scores

        converted_outputs.append(converted_batch)

    return converted_outputs


def _move_outputs_to_cpu(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Move all tensors in outputs to CPU and detach from computation graph.
    Handles nested structures including top-K formatted scores.

    Args:
        outputs: List of output dictionaries

    Returns:
        Same structure with all tensors moved to CPU and detached
    """

    def move_to_cpu(value: Any) -> Any:
        """Recursively move value to CPU, handling various types."""
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu()
        elif isinstance(value, dict):
            return {k: move_to_cpu(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(move_to_cpu(item) for item in value)
        else:
            # Primitives (str, int, float, etc.) - return as-is
            return value

    return [move_to_cpu(batch_dict) for batch_dict in outputs]


def run_interchange_interventions(
    pipeline: Pipeline,
    counterfactual_dataset: CounterfactualDataset,
    interchange_target: InterchangeTarget,
    batch_size: int = 32,
    output_scores: bool | int = True,
) -> dict[str, list[Any]]:
    """
    Run interchange interventions on a full counterfactual dataset in batches.

    This function:
    1. Prepares an intervenable model configured for interchange interventions
    2. Processes the dataset in batches, applying interventions to each batch
    3. Converts scores to top-k format if requested (for memory efficiency)
    4. Moves all outputs to CPU to free GPU memory
    5. Collects and returns results from all batches

    Args:
        pipeline: Neural model pipeline that handles tokenization and generation
        counterfactual_dataset: Dataset containing inputs and their counterfactuals
        interchange_target: InterchangeTarget containing model components to intervene on,
                           where groups share counterfactual inputs
        batch_size: Number of examples to process in each batch
        output_scores: Controls score output format:
            - False: No scores
            - True: Full vocabulary scores (on CPU)
            - int (e.g., 10): Top-k scores (on CPU, memory efficient)

    Returns:
        List[dict]: List of dictionaries, each with 'sequences' (on CPU) and optionally
                   'scores' keys (on CPU, in top-k format if int was provided)
    """
    # Initialize intervenable model with interchange intervention type
    intervenable_model = prepare_intervenable_model(
        pipeline, interchange_target, intervention_type="interchange"
    )

    # Create data loader for batch processing
    def shallow_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {key: [item[key] for item in batch] for key in batch[0].keys()}

    dataloader = DataLoader(
        counterfactual_dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,  # Maintain dataset order
        collate_fn=shallow_collate_fn,
    )
    all_outputs = []

    # Process each batch with progress tracking
    for batch in tqdm(
        dataloader,
        desc="Processing batches",
        disable=not logger.isEnabledFor(logging.DEBUG),
        leave=False,
    ):
        with torch.no_grad():  # Disable gradient tracking for inference
            # Perform interchange interventions on the batch - returns dict
            output_dict = batched_interchange_intervention(
                pipeline,
                intervenable_model,
                batch,
                interchange_target,
                output_scores=output_scores,
            )

            # Collect outputs from this batch
            all_outputs.append(output_dict)

    # Clean up the intervenable model to free GPU memory
    delete_intervenable_model(intervenable_model)

    # Convert to top-k format if requested (while still on GPU for efficiency)
    if not isinstance(output_scores, bool) and output_scores > 0:
        all_outputs = _convert_to_top_k(all_outputs, pipeline, k=output_scores)

    # Move all outputs to CPU
    all_outputs = _move_outputs_to_cpu(all_outputs)

    # remove batch structure from outputs
    all_outputs = {
        k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()
    }

    return all_outputs


def train_interventions(
    pipeline: Pipeline,
    interchange_target: InterchangeTarget,
    counterfactual_dataset: CounterfactualDataset,
    intervention_type: str,
    config: dict[str, Any],
    loss_and_metric_fn: Callable[..., tuple[Tensor, dict[str, Any], dict[str, Any]]],
) -> str:
    """
    Train intervention models on a counterfactual dataset.

    This function implements the training loop for neural network interventions,
    supporting both "interchange" and "mask" intervention types. It optimizes
    intervention parameters while keeping the base model frozen.

    Args:
        pipeline: Neural model pipeline for tokenization and model execution
        interchange_target: InterchangeTarget containing model components to intervene on,
                           where groups share counterfactual inputs
        counterfactual_dataset: Dataset containing original inputs and their counterfactuals
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
                           with signature (pipeline, model, batch, units) ->
                           (loss, metrics_dict, logging_info)

    Returns:
        str: Summary string with final metrics
    """
    # ----- Model Initialization ----- #
    intervenable_model = prepare_intervenable_model(
        pipeline, interchange_target, intervention_type=intervention_type
    )
    intervenable_model.disable_model_gradients()
    intervenable_model.eval()

    # ----- Data Preparation ----- #
    def shallow_collate_fn_train(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {key: [item[key] for item in batch] for key in batch[0].keys()}

    dataloader = DataLoader(
        counterfactual_dataset,  # type: ignore[arg-type]
        batch_size=config["train_batch_size"],
        shuffle=config.get("shuffle", True),
        collate_fn=shallow_collate_fn_train,
    )

    # ----- Configuration ----- #
    num_epoch = config["training_epoch"]
    regularization_coefficient = config.get("masking", {}).get(
        "regularization_coefficient", 1e-4
    )
    memory_cleanup_freq = config.get("memory_cleanup_freq", 50)
    patience = config.get("patience", None)  # Default to no early stopping
    scheduler_type = config.get("scheduler_type", "constant")

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
        num_training_steps=num_epoch * len(dataloader),
    )

    # Track step count manually instead of accessing scheduler._step_count
    current_step = 0

    # ----- Temperature Scheduling for Mask Interventions ----- #
    temperature_schedule = None
    interventions = None
    if intervention_type == "mask":
        temperature_start, temperature_end = config.get("masking", {}).get(
            "temperature_schedule", (1.0, 0.01)
        )
        temperature_annealing_fraction = config.get("masking", {}).get(
            "temperature_annealing_fraction", 0.5
        )

        # Calculate number of steps for annealing
        total_steps = num_epoch * len(dataloader)
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
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch: {epoch}", position=1, leave=False
        )

        aggregated_stats = collections.defaultdict(list)

        for step, batch in enumerate(epoch_iterator):
            # Move batch data to device
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(pipeline.model.device)

            # Run training step
            loss, eval_metrics, _logging_info = loss_and_metric_fn(
                pipeline, intervenable_model, batch, interchange_target
            )

            # Add sparsity loss for mask interventions
            if intervention_type == "mask":
                masks_list: list[Tensor] = []
                assert temperature_schedule is not None
                assert interventions is not None
                temp = temperature_schedule[current_step]
                for k, v in interventions.items():
                    # pyvene's intervention dict values are dynamically typed
                    mask_intervention = v[0] if isinstance(v, tuple) else v
                    loss = (
                        loss
                        + regularization_coefficient
                        * mask_intervention.get_sparsity_loss()  # pyright: ignore[reportCallIssue]
                    )
                    masks_list.append(mask_intervention.mask)  # pyright: ignore[reportArgumentType]
                    mask_intervention.set_temperature(temp)  # pyright: ignore[reportCallIssue]
                if config["featurizer_kwargs"]["tie_masks"]:
                    masks = torch.cat(masks_list)
                    sparse_loss = torch.norm(
                        torch.sigmoid(
                            masks / temp,
                        ),
                        p=1,
                    )
                    loss = loss + regularization_coefficient * sparse_loss

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

    summary = f"Trained intervention for {str(interchange_target)[:200]}"
    summary += "\nFinal metrics: " + " ".join(
        [f"{k}: {v}" for k, v in postfix_dict.items()]
    )
    return summary
