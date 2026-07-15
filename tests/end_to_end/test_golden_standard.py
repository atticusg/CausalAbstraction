"""Solvability-floor gate for the runner goldens.

Enforces the "Runner-golden standard" (docs/TESTS.md): every *accuracy* golden
must pin ``accuracy.accuracy >= 0.9``. This is a pure inspection of the
pinned JSONs under ``tests/end_to_end/goldens/`` — it runs no model, so it
lives on the CPU ``test`` job (``unit`` tier) and catches a future
``update_goldens.py`` regen that would silently lower the bar (e.g. a model
or scoring change that drops a task below the floor).

Goldens with no ``accuracy.accuracy`` key — the analysis-only chains such as
``output_manifold`` that ship no baseline accuracy — carry no accuracy gate
and are excluded from this gate's parametrization entirely (not collected and
skipped). The numerical reproduction of their pinned values is the
``test_goldens.py`` GPU job's concern; this gate only guards the floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.end_to_end._helpers.golden import enumerate_goldens, load_golden

pytestmark = pytest.mark.unit

# The runner-golden accuracy floor (docs/TESTS.md "Runner-golden standard").
_ACCURACY_FLOOR = 0.9


def _accuracy_golden_paths() -> list[Path]:
    """Golden JSONs that pin an ``accuracy.accuracy`` — the only goldens this
    floor gate applies to. Analysis-only chains (e.g. ``output_manifold``,
    ``mcqa_attention_pattern``) pin no accuracy and are excluded from the
    parametrization entirely, rather than collected and skipped."""
    return [
        p
        for p in enumerate_goldens()
        if load_golden(p).values.get("accuracy.accuracy") is not None
    ]


_GOLDEN_PATHS = _accuracy_golden_paths()


@pytest.mark.parametrize(
    "golden_path",
    _GOLDEN_PATHS,
    ids=[p.stem for p in _GOLDEN_PATHS],
)
def test_accuracy_golden_clears_floor(golden_path: Path) -> None:
    """Every golden that pins an accuracy must pin it at or above the floor."""
    golden = load_golden(golden_path)
    acc = golden.values["accuracy.accuracy"]
    assert acc >= _ACCURACY_FLOOR, (
        f"[{golden.runner}] pinned accuracy.accuracy={acc!r} is below the "
        f"{_ACCURACY_FLOOR} runner-golden floor (docs/TESTS.md 'Runner-golden "
        f"standard'). A regen must not silently lower the bar — fix the "
        f"task / model / scoring (or drop the golden), don't pin a sub-floor "
        f"accuracy."
    )
