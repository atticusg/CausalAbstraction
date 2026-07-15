"""Session-scoped tiny pipeline fixture for hierarchical_equality token-position tests.

The shipped HE token-position regexes (``causalab/tasks/hierarchical_equality/
token_positions.py``) require a real :class:`~causalab.neural.pipeline.LMPipeline`
because :func:`_get_var_token_indices` calls
``pipeline.load([trace], return_offsets_mapping=True)`` to map character
spans to token indices. The ``tests/_helpers/tiny.py::tiny_random_tokenizer`` /
``tiny_random_model`` factories give a real Llama stub but not an
``LMPipeline`` wrapper, so we wrap once per session here.

Why session scope? Building the pipeline triggers an HF model load (under
~1s warmed); paying it once per session keeps the token-positions class
sub-second across its parametrised invariants.
"""

from __future__ import annotations

import pytest

from causalab.neural.pipeline import LMPipeline
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME


@pytest.fixture(scope="session")
def he_tiny_pipeline() -> LMPipeline:
    """Tiny Llama stub wrapped in an LMPipeline (CPU, float32)."""
    return LMPipeline(
        TINY_RANDOM_MODEL_NAME,
        max_new_tokens=1,
        device="cpu",
        dtype="float32",
    )
