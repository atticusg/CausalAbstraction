"""
I/O utilities for saving experiment results.

This module provides functions for saving experiment results, metadata,
and trained models to disk.
"""

import os
import json
from typing import Dict, Any, Tuple

import torch
from safetensors.torch import save_file


def _key_to_str(key: Tuple[Any, ...]) -> str:
    """Convert a tuple key to a string representation for file naming."""
    if len(key) == 2:
        return f"{key[0]}__{key[1]}"
    elif len(key) == 1:
        return str(key[0])
    return "__".join(str(k) for k in key)


def save_experiment_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
    filename: str = "metadata.json",
) -> str:
    """
    Save experiment metadata to JSON file.

    Args:
        metadata: Metadata dictionary to save
        output_dir: Output directory
        filename: Filename for metadata (default: "metadata.json")

    Returns:
        Path to saved metadata file
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def save_json_results(
    data: Dict[str, Any],
    output_dir: str,
    filename: str,
) -> str:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON-serializable)
        output_dir: Output directory
        filename: Filename (should end with .json)

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return file_path


def save_tensor_results(
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    filename: str,
) -> str:
    """
    Save tensors to safetensors file.

    Args:
        tensors: Dictionary mapping names to tensors
        output_dir: Output directory
        filename: Filename (should end with .safetensors)

    Returns:
        Path to saved safetensors file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    save_file(tensors, file_path)

    return file_path


def save_intervention_results(
    results_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    output_dir: str,
    prefix: str = "",
) -> Dict[str, str]:
    """
    Save intervention results (scores and raw outputs).

    Args:
        results_by_key: Dict mapping keys to result dicts with:
            - score: float
            - scores_by_variable: dict (unused, kept for compatibility)
            - raw_results: dict with "string" and "sequences"
        output_dir: Output directory
        prefix: Optional prefix for subdirectory (e.g., "train_eval", "test_eval")

    Returns:
        Dict of output paths
    """
    if prefix:
        save_dir = os.path.join(output_dir, prefix)
    else:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    paths = {}

    # Save scores.json: {key: score}
    scores = {}
    for key, result in results_by_key.items():
        key_str = _key_to_str(key)
        scores[key_str] = result["avg_score"]

    scores_path = save_json_results(scores, save_dir, "scores.json")
    paths["scores_path"] = scores_path

    # Save raw_results.json: aggregated string outputs
    raw_results_json = {
        "string": [
            result["raw_results"]["string"] for result in results_by_key.values()
        ]
    }
    raw_path = save_json_results(raw_results_json, save_dir, "raw_results.json")
    paths["raw_results_path"] = raw_path

    # Save raw_results.safetensors: aggregated tensor outputs (if present)
    sequences_list = []
    for result in results_by_key.values():
        if "sequences" in result["raw_results"] and result["raw_results"]["sequences"]:
            sequences_list.extend(result["raw_results"]["sequences"])

    if sequences_list:
        sequences_tensor = torch.cat(sequences_list, dim=0)
        tensor_path = save_tensor_results(
            {"sequences": sequences_tensor},
            save_dir,
            "raw_results.safetensors",
        )
        paths["raw_results_tensors_path"] = tensor_path

    return paths


def save_training_artifacts(
    results_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Save training-specific artifacts (feature indices, models).

    Args:
        results_by_key: Dict mapping keys to result dicts with:
            - feature_indices: dict
            - trained_target: InterchangeTarget (optional, for model saving)
        output_dir: Output directory

    Returns:
        Dict of output paths
    """
    paths = {}

    # Save feature_indices.json: {key: {unit_id: [indices]}}
    training_dir = os.path.join(output_dir, "training")
    os.makedirs(training_dir, exist_ok=True)

    feature_indices_serializable = {}
    for key, result in results_by_key.items():
        key_str = _key_to_str(key)
        feature_indices_serializable[key_str] = result["feature_indices"]

    feature_path = save_json_results(
        feature_indices_serializable,
        training_dir,
        "feature_indices.json",
    )
    paths["feature_indices_path"] = feature_path

    # Save models to models/ directory
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for key, result in results_by_key.items():
        if "trained_target" in result:
            key_str = _key_to_str(key)
            model_path = os.path.join(models_dir, key_str)
            result["trained_target"].save(model_path)

    paths["models_dir"] = models_dir

    return paths


def save_aggregate_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Save aggregate experiment metadata.

    Alias for save_experiment_metadata for consistency.
    """
    return save_experiment_metadata(metadata, output_dir, "metadata.json")
