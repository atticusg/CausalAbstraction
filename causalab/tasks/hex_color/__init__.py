"""``hex_color``: perceptual colour-classification task.

A hue-jittered hex code is shown to the model, which must name the colour it
best matches from six fixed choices (red/orange/yellow/green/blue/purple).
Singleton task; the answer variable ``color`` carries a periodic hue-centre
embedding (360° period).
"""

from .causal_models import CAUSAL_MODEL
from .counterfactuals import COUNTERFACTUAL_GENERATORS

__all__ = ["CAUSAL_MODEL", "COUNTERFACTUAL_GENERATORS"]
