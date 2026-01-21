"""
Configuration settings for causal abstraction experiments.

This module provides default configurations and preset configurations for various
experiments that extend intervention_experiment.py
"""

import copy
from typing import Literal, TypedDict


# =============================================================================
# Full configs (all fields required) - used internally after merge
# =============================================================================


class DASConfig(TypedDict):
    """Configuration for DAS (Distributed Alignment Search) method."""

    n_features: int


class MaskingConfig(TypedDict):
    """Configuration for DBM/masking method."""

    regularization_coefficient: float
    temperature_annealing_fraction: float
    temperature_schedule: tuple[float, float]


class FeaturizerKwargs(TypedDict):
    """Keyword arguments for featurizer constructors."""

    tie_masks: bool


class ExperimentConfig(TypedDict):
    """
    Full configuration for causal abstraction experiments.

    All fields are present after merging with DEFAULT_CONFIG.
    """

    # General Experiment Parameters
    intervention_type: Literal["mask", "interchange"]
    train_batch_size: int
    evaluation_batch_size: int
    method_name: str
    id: str
    output_scores: bool
    top_k_logits: int | None

    # Training Parameters
    training_epoch: int
    init_lr: float
    max_output_tokens: int
    log_dir: str

    # Optional Training Parameters
    patience: int | None
    scheduler_type: Literal["constant", "linear", "cosine"]
    memory_cleanup_freq: int
    shuffle: bool

    # Method-specific Parameters
    DAS: DASConfig
    masking: MaskingConfig
    featurizer_kwargs: FeaturizerKwargs


# Default configuration with all available parameters
DEFAULT_CONFIG: ExperimentConfig = {
    # ========== General Experiment Parameters ==========
    # Type of intervention: "mask" for DBM, "interchange" for DAS
    "intervention_type": "mask",
    # Batch size for training and general processing
    "train_batch_size": 32,
    # Batch size specifically for evaluation
    "evaluation_batch_size": 32,
    # Name of the method being used (for logging/identification)
    "method_name": "default",
    # Unique identifier for this experiment configuration
    "id": "default_experiment",
    # Whether to output scores during evaluation
    "output_scores": False,
    # Number of top-k logits to extract and keep in memory (applied immediately after generation)
    # Set to None or 0 to keep full vocabulary logits (uses ~256K floats per token, can cause OOM)
    # Recommended: 10-50 (reduces memory by ~5000-25000x)
    "top_k_logits": 10,
    # ========== General Training Parameters ==========
    # Number of training epochs
    "training_epoch": 3,
    # Initial learning rate for optimization
    "init_lr": 1e-2,
    # Maximum number of tokens to generate for output
    "max_output_tokens": 1,
    # Directory for TensorBoard logs
    "log_dir": "logs",
    # ========== Optional Training Parameters ==========
    # Early stopping patience (None to disable)
    "patience": None,
    # Learning rate scheduler type ("constant", "linear", "cosine")
    "scheduler_type": "constant",
    # Frequency of memory cleanup during training (in batches)
    "memory_cleanup_freq": 50,
    # Whether to shuffle training data
    "shuffle": True,
    # ========== DAS-specific Parameters ==========
    # Used in train_interventions() when method="DAS"
    "DAS": {
        # Number of features/dimensions for the learned subspace
        "n_features": 32,
    },
    # ========== DBM/Masking-specific Parameters ==========
    # Used in train_interventions() when method="DBM"
    "masking": {
        # L1 regularization coefficient for sparsity
        "regularization_coefficient": 0.1,
        "temperature_annealing_fraction": 0.5,  # Fraction of training steps to anneal temperature
        # Temperature annealing schedule for Gumbel-Softmax (start, end)
        "temperature_schedule": (1.0, 0.001),
    },
    # Keyword arguments to pass to Featurizer constructors
    # This allows flexible configuration of featurizer behavior
    "featurizer_kwargs": {
        # Whether to tie mask weights within each atomic model unit (DBM only)
        "tie_masks": False
    },
}


# =============================================================================
# Partial configs (all fields optional) - used at API boundaries
# =============================================================================


class PartialDASConfig(TypedDict, total=False):
    """Partial DAS config for API boundaries."""

    n_features: int


class PartialMaskingConfig(TypedDict, total=False):
    """Partial masking config for API boundaries."""

    regularization_coefficient: float
    temperature_annealing_fraction: float
    temperature_schedule: tuple[float, float]


class PartialFeaturizerKwargs(TypedDict, total=False):
    """Partial featurizer kwargs for API boundaries."""

    tie_masks: bool


class PartialExperimentConfig(TypedDict, total=False):
    """
    Partial configuration for causal abstraction experiments.

    All fields are optional - used at API boundaries.
    Will be merged with DEFAULT_CONFIG to produce a full ExperimentConfig.
    """

    # General Experiment Parameters
    intervention_type: Literal["mask", "interchange"]
    train_batch_size: int
    evaluation_batch_size: int
    method_name: str
    id: str
    output_scores: bool
    top_k_logits: int | None

    # Training Parameters
    training_epoch: int
    init_lr: float
    max_output_tokens: int
    log_dir: str

    # Optional Training Parameters
    patience: int | None
    scheduler_type: Literal["constant", "linear", "cosine"]
    memory_cleanup_freq: int
    shuffle: bool

    # Method-specific Parameters
    DAS: PartialDASConfig
    masking: PartialMaskingConfig
    featurizer_kwargs: PartialFeaturizerKwargs


# Type alias for API boundaries - accepts full, partial, or no config
ExperimentConfigInput = ExperimentConfig | PartialExperimentConfig | None


def merge_with_defaults(
    config: ExperimentConfigInput,
) -> ExperimentConfig:
    """
    Merge partial config with DEFAULT_CONFIG, handling nested dicts.

    Args:
        config: Full, partial, or None config. Will be merged with defaults.

    Returns:
        Complete ExperimentConfig with all fields populated.
    """
    if config is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    merged = copy.deepcopy(DEFAULT_CONFIG)
    # Note: type: ignore[literal-required] is needed because basedpyright widens
    # TypedDict key types to `str` when iterating with .items(), even though the
    # keys are known to be valid literal keys of the TypedDict at runtime.
    for key, value in config.items():
        existing = merged.get(key)
        # Deep merge for nested dicts (DAS, masking, featurizer_kwargs)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = {**existing, **value}  # type: ignore[literal-required]
        else:
            merged[key] = value  # type: ignore[literal-required]
    return merged
