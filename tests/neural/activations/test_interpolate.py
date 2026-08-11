"""``run_interpolation_interventions`` — the public interpolation wrapper.

WU3 (#505) signature tests: the wrapper takes ``Sequence[Sequence[SiteSpec]]``
natively and requires an explicit ``fn``. The interpolation *math* is
oracle-pinned in ``tests/neural/activations/test_interpolation_hook_oracle.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.neural.activations.interpolate import run_interpolation_interventions
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.site import Site
from causalab.neural.specs import SiteSpec
from causalab.neural.token_positions import TokenPosition

from tests.neural.activations.hook_oracle import make_trace

pytestmark = pytest.mark.unit


def _groups(pipeline: LMPipeline) -> list[list[SiteSpec]]:
    tp = TokenPosition(lambda _x: [0], pipeline, id="first_token")
    spec = SiteSpec(
        fsite=FeaturizedSite(Site("block_output", 0)),
        positions=tp,
        key="resid_L0_first_token",
        width=pipeline.model.config.hidden_size,
    )
    return [[spec]]


def _dataset() -> list[dict[str, Any]]:
    return [
        {"input": make_trace(t), "counterfactual_inputs": [make_trace(c)]}
        for t, c in [("alpha beta", "gamma delta"), ("one two", "three four")]
    ]


class TestRunInterpolationInterventionsUnit:
    def test_fn_required_guard(self, tiny_pipeline: LMPipeline) -> None:
        """Omitting ``fn`` is refused before any execution."""
        with pytest.raises(TypeError, match="requires fn"):
            run_interpolation_interventions(
                tiny_pipeline, _dataset()[:1], _groups(tiny_pipeline)
            )
