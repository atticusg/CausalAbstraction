"""Natural domains arithmetic: unified task for weekdays, months, and hours.

Factory task — use NaturalDomainConfig to select the domain variant.
"""

from .config import NaturalDomainConfig, DOMAIN_PRESETS
from .causal_models import create_causal_model, create_random_causal_model
from .counterfactuals import generate_dataset
from .token_positions import create_token_positions

__all__ = [
    "NaturalDomainConfig",
    "DOMAIN_PRESETS",
    "create_causal_model",
    "create_random_causal_model",
    "generate_dataset",
    "create_token_positions",
]
