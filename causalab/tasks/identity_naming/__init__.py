"""Identity naming: factory task for entity -> canonical name/number mappings.

Factory task — use IdentityNamingConfig to select the domain variant.
"""

from .config import IdentityNamingConfig, DOMAIN_PRESETS
from .causal_models import create_causal_model
from .counterfactuals import generate_dataset
from .token_positions import create_token_positions

__all__ = [
    "IdentityNamingConfig",
    "DOMAIN_PRESETS",
    "create_causal_model",
    "generate_dataset",
    "create_token_positions",
]
