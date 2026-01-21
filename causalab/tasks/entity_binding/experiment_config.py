"""
DEPRECATED: This task is outdated and may not reflect current best practices.
See causalab/tasks/MCQA/ for an up-to-date example.

Task Plugin Interface for Entity Binding

This module provides a standardized interface for the entity binding task
to work with the universal experiment scripts.
"""

from typing import Dict, Callable, Any, List
from causalab.tasks.entity_binding.config import (
    create_sample_love_config,
    create_sample_action_config,
    EntityBindingTaskConfig
)
from causalab.tasks.entity_binding.causal_models import (
    create_positional_causal_model,
    create_positional_entity_causal_model
)
from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.tasks.entity_binding.token_positions import get_entity_token_indices_structured
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index


def get_task_config(config_name: str) -> EntityBindingTaskConfig:
    """
    Get task configuration by name.

    Args:
        config_name: One of 'love', 'action', 'positional_entity'

    Returns:
        EntityBindingTaskConfig object

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name == 'love':
        config = create_sample_love_config()
        config.max_groups = 2
    elif config_name == 'action':
        config = create_sample_action_config()
        config.max_groups = 3
    elif config_name == 'positional_entity':
        config = create_sample_love_config()
        config.max_groups = 2
        config.fixed_query_indices = (0,)  # Fix query indices for positional experiments
    else:
        raise ValueError(f"Unknown config name: {config_name}. Use 'love', 'action', or 'positional_entity'")

    # Add instruction wrapper for better performance
    config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"

    return config


def get_causal_model(config: EntityBindingTaskConfig, model_type: str = 'positional'):
    """
    Get causal model for the given configuration.

    Args:
        config: Task configuration
        model_type: 'positional' or 'positional_entity'

    Returns:
        CausalModel object

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'positional':
        return create_positional_causal_model(config)
    elif model_type == 'positional_entity':
        return create_positional_entity_causal_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'positional' or 'positional_entity'")


def get_counterfactual_generator(cf_name: str, config: EntityBindingTaskConfig) -> Callable:
    """
    Get counterfactual generator function by name.

    Args:
        cf_name: Counterfactual generator name (currently only 'swap_query_group' supported)
        config: Task configuration

    Returns:
        Counterfactual generator function

    Raises:
        ValueError: If cf_name is not recognized
    """
    if cf_name == 'swap_query_group':
        return lambda: swap_query_group(config)
    else:
        raise ValueError(f"Unknown counterfactual generator: {cf_name}. Use 'swap_query_group'")


def get_token_positions(pipeline, config: EntityBindingTaskConfig, position_type: str = 'last_token') -> Dict[str, TokenPosition]:
    """
    Get token positions for the experiment.

    Args:
        pipeline: LMPipeline object
        config: Task configuration
        position_type: Type of token positions to create ('last_token' or 'entity_tokens')

    Returns:
        Dictionary mapping position names to TokenPosition objects

    Raises:
        ValueError: If position_type is not recognized
    """
    token_positions = {}

    if position_type == 'last_token':
        token_positions['last_token'] = TokenPosition(
            lambda x: get_last_token_index(x, pipeline),
            pipeline,
            id="last_token"
        )
    elif position_type == 'both_e1':
        # For positional entity experiments - both e1 entities
        def indexer_both_e1(input_dict, is_original=True):
            """Return last tokens of both e1 entities, with order based on is_original."""
            # Convert query_indices to tuple if it's a list
            if 'query_indices' in input_dict and isinstance(input_dict['query_indices'], list):
                input_dict = dict(input_dict)
                input_dict['query_indices'] = tuple(input_dict['query_indices'])

            tokens_list = []
            for group_idx in [0, 1]:
                try:
                    tokens = get_entity_token_indices_structured(
                        input_dict,
                        pipeline,
                        config,
                        group_idx=group_idx,
                        entity_idx=1,  # e1 is the second entity
                        region='statement'
                    )
                    if tokens:
                        tokens_list.append(tokens[-1])
                    else:
                        tokens_list.append(None)
                except (ValueError, KeyError):
                    tokens_list.append(None)

            # If is_original=False (counterfactual), reverse the order
            if not is_original:
                tokens_list = tokens_list[::-1]

            # Filter out None values and return
            return [t for t in tokens_list if t is not None]

        token_positions['both_e1'] = TokenPosition(
            indexer_both_e1,
            pipeline,
            is_original=True,
            id="both_e1_last_tokens"
        )
    else:
        raise ValueError(f"Unknown position type: {position_type}. Use 'last_token' or 'both_e1'")

    return token_positions


def get_checker() -> Callable:
    """
    Get checker function for verifying model outputs match causal model outputs.

    Returns:
        Checker function that takes (neural_output, causal_output) and returns bool
    """
    def checker(neural_output, causal_output):
        """Check if neural network output matches causal model output."""
        return causal_output in neural_output["string"] or neural_output["string"].strip() in causal_output

    return checker


def get_default_target_variables(config_name: str) -> List[List[str]]:
    """
    Get default target variables to track for a given configuration.

    Args:
        config_name: Configuration name ('love', 'action', 'positional_entity')

    Returns:
        List of target variable lists
    """
    if config_name in ['love', 'action']:
        return [
            ["positional_query_group"],
            ["query_entity"],
            ["raw_output"]
        ]
    elif config_name == 'positional_entity':
        return [
            [
                "positional_entity_g0_e1<-positional_entity_g1_e1",
                "positional_entity_g1_e1<-positional_entity_g0_e1",
                "positional_entity_g0_e0<-positional_entity_g1_e0",
                "positional_entity_g1_e0<-positional_entity_g0_e0"
            ]
        ]
    else:
        raise ValueError(f"Unknown config name: {config_name}")


def get_pipeline_config(config_name: str) -> Dict[str, Any]:
    """
    Get pipeline configuration (max_new_tokens, max_length) for a given config.

    Args:
        config_name: Configuration name

    Returns:
        Dictionary with pipeline config parameters
    """
    if config_name in ['love', 'action', 'positional_entity']:
        return {
            'max_new_tokens': 5,
            'max_length': 256
        }
    else:
        raise ValueError(f"Unknown config name: {config_name}")
