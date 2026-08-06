"""Do the goldens encode the *known result*, or merely a stable one?

``test_goldens.py`` re-runs each config and checks it reproduces the recorded
values. That is a regression test: it proves the pipeline still does what it did
last time. It cannot prove the pipeline does the right thing, because a wrong
answer that is stable passes just as well — and `update_goldens.py` will happily
record one.

These tests close the other half. They read the *recorded* values and assert they
carry the published finding. Together with the numeric gate the chain becomes:

    live run == recorded values      (test_goldens.py, on GPU)
    recorded values == known science (here, on CPU in milliseconds)
    => live run == known science

Which is the property a user actually cares about: that path patching run on IOI
finds the IOI circuit, not just the same numbers as yesterday.

No model is loaded — this reads JSON. That is deliberate: it means the validity
check runs in the default CPU tier on every commit, so a re-record that silently
enshrines a broken result fails here rather than being discovered by someone
reading a heatmap months later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.numerical_unit

_GOLDENS = Path(__file__).parent / "goldens"

# Wang et al. (2022), "Interpretability in the Wild", §3.1 — the IOI circuit in
# GPT-2 small. Name movers copy the indirect object into the residual stream and
# so raise the IO-vs-subject logit difference; negative name movers do the
# opposite. These are the paper's head identities, not ours.
NAME_MOVERS = {(9, 9), (9, 6), (10, 0)}
NEGATIVE_NAME_MOVERS = {(10, 7), (11, 10)}


def _effect_grid(golden: str) -> dict[tuple[int, int], float]:
    """The recorded (layer, head) -> direct-effect map for a path-patching golden."""
    values = json.loads((_GOLDENS / f"{golden}.json").read_text())["values"]
    grid: dict[tuple[int, int], float] = {}
    for key, value in values.items():
        match = re.search(r"effect_grid\.(\d+)\|(\d+)$", key)
        if match and isinstance(value, float):
            grid[(int(match.group(1)), int(match.group(2)))] = value
    return grid


class TestIOIPathPatchingFindsTheIOICircuit:
    """The headline claim path patching exists to support.

    The config picks GPT-2 small precisely so this is checkable: it is the model
    the circuit was characterised in. A refactor that inverted a sign, patched
    the wrong head, or measured the indirect effect instead of the direct one
    would still produce a full, plausible, stable heatmap — and would fail here.
    """

    def test_the_grid_covers_every_head(self) -> None:
        grid = _effect_grid("ioi_path_patching")
        assert len(grid) == 144, "GPT-2 small is 12 layers x 12 heads"

    def test_name_movers_are_the_strongest_positive_heads(self) -> None:
        """The three heads the paper names must be the top three, not merely
        present somewhere in the upper half."""
        grid = _effect_grid("ioi_path_patching")
        top3 = {head for head, _ in sorted(grid.items(), key=lambda kv: -kv[1])[:3]}
        assert top3 == NAME_MOVERS, f"top three were {sorted(top3)}"

    def test_negative_name_movers_are_the_strongest_negative_heads(self) -> None:
        grid = _effect_grid("ioi_path_patching")
        bottom2 = {head for head, _ in sorted(grid.items(), key=lambda kv: kv[1])[:2]}
        assert bottom2 == NEGATIVE_NAME_MOVERS, f"bottom two were {sorted(bottom2)}"

    def test_the_two_families_have_the_signs_the_paper_gives_them(self) -> None:
        """Sign, independently of rank: a global sign flip would keep every
        ranking intact while reversing what the method claims."""
        grid = _effect_grid("ioi_path_patching")
        for head in NAME_MOVERS:
            assert grid[head] > 0, f"name mover {head} scored {grid[head]:+.4f}"
        for head in NEGATIVE_NAME_MOVERS:
            assert grid[head] < 0, (
                f"negative name mover {head} scored {grid[head]:+.4f}"
            )

    def test_the_circuit_is_sparse(self) -> None:
        """Most heads do nothing on this task — that sparsity *is* the finding.

        A method that reported a large effect everywhere would rank the name
        movers correctly and still be useless, so rank alone is not enough.
        """
        grid = _effect_grid("ioi_path_patching")
        strong = {head for head, v in grid.items() if abs(v) > 0.2}
        assert len(strong) <= 12, f"{len(strong)} of 144 heads scored |effect| > 0.2"
        assert NAME_MOVERS <= strong
        assert NEGATIVE_NAME_MOVERS <= strong

    def test_name_movers_dominate_the_typical_head(self) -> None:
        """The separation is orders of magnitude, which is what makes the
        heatmap readable rather than a judgement call about thresholds."""
        grid = _effect_grid("ioi_path_patching")
        median = sorted(abs(v) for v in grid.values())[len(grid) // 2]
        best = max(grid[h] for h in NAME_MOVERS)
        assert best > 100 * median, (
            f"top name mover {best:.4f} vs median |{median:.5f}|"
        )


class TestCausalTracingExperimentIsNonVacuous:
    """Causal tracing measures how much of a destroyed behaviour a restoration
    brings back. That number only means something if the behaviour was there to
    begin with and the corruption actually destroyed it.

    Those two preconditions are what these assert. If corruption silently stopped
    working — a noise vector of the wrong scale, a span that resolved to nothing,
    an entry site that stopped being written — the floor would rise to meet the
    ceiling and *every* recovery value would become noise around zero. The
    numbers would still be stable, so the numeric gate would pass, and the
    heatmap would still render. It would simply no longer be measuring anything.
    """

    def _results(self) -> dict[str, float]:
        values = json.loads(
            (_GOLDENS / "ioi_causal_sufficiency_residual.json").read_text()
        )["values"]
        return {
            key.split("results.")[-1]: value
            for key, value in values.items()
            if isinstance(value, float)
        }

    def test_the_model_does_the_task_uncorrupted(self) -> None:
        """The ceiling is the clean logit difference: IO over the repeated
        subject. Near zero would mean GPT-2 is not solving IOI here, and nothing
        downstream could be interpreted."""
        assert self._results()["clean_ceiling"] > 1.0

    def test_corruption_actually_destroys_the_behaviour(self) -> None:
        """The floor must fall well below the ceiling. This is the assertion that
        catches a corruption which quietly became a no-op."""
        results = self._results()
        ceiling, floor = results["clean_ceiling"], results["corrupted_floor"]
        assert floor < 0.25 * ceiling, f"floor {floor:+.4f} vs ceiling {ceiling:+.4f}"

    def test_there_is_a_gap_worth_measuring(self) -> None:
        results = self._results()
        assert results["clean_ceiling"] - results["corrupted_floor"] > 1.0

    def test_no_restoration_beats_the_clean_run(self) -> None:
        """A cell recovering *more* than the uncorrupted model is not a strong
        finding, it is a bug — the restored run cannot know more than the run it
        was restored from."""
        results = self._results()
        ceiling = results["clean_ceiling"]
        cells = {k: v for k, v in results.items() if k.startswith("recovery_grid.")}
        assert cells, "no recovery grid recorded"
        for name, value in cells.items():
            assert value <= ceiling, (
                f"{name} recovered {value:+.4f} > ceiling {ceiling:+.4f}"
            )
