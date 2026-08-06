"""The position-indexing primitives, exercised directly.

``FlatIndex`` names every selected ``(example, position)`` pair, and
``gather_positions`` / ``scatter_positions`` / ``align_per_example`` all read
that naming. Everything the engine does to an activation goes through these
three, so a mistake here is a wrong number in every method at once — and a quiet
one, because the shapes still line up.

Twice now a bug has come from *inferring* what a tensor means from its
dimensions instead of from the index. A ``(batch, 1, width)`` steering vector and
a ``(total_positions, width)`` interchange source can have the same length: they
agree whenever every example selects exactly one token, which is the common case
and therefore the case that hides the disagreement. These tests walk count
patterns where the two readings come apart, so the coincidence cannot pass.

The engine-level oracles check that interventions land the right values in a real
forward. These check the arithmetic underneath, where a case can be constructed
that no model fixture happens to produce.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.activations.engine import (
    align_per_example,
    flat_index,
    gather_positions,
    scatter_positions,
)

pytestmark = pytest.mark.unit

WIDTH = 4

# (label, per-example positions). Chosen so the per-example and per-position
# readings of a source disagree: `uniform single` is the case where they agree,
# and is here to show the others are not simply exercising a broken path.
COUNT_PATTERNS = [
    ("uniform single", [[1], [2], [3]]),
    ("uniform multi", [[1, 2], [3, 4]]),
    ("ragged", [[1, 2], [3]]),
    ("ragged, widening", [[0], [1, 2], [3, 4, 5]]),
    ("single example", [[2, 4]]),
]


def _per_example_source(batch: int) -> torch.Tensor:
    """One distinguishable vector per example, in the ``(batch, 1, width)`` shape
    a steering vector arrives in."""
    return (
        torch.arange(batch, dtype=torch.float32)
        .view(batch, 1, 1)
        .expand(batch, 1, WIDTH)
        .contiguous()
    )


class TestFlatIndex:
    @pytest.mark.parametrize("label,positions", COUNT_PATTERNS, ids=lambda v: v)
    def test_pairs_match_the_positions_asked_for(self, label, positions) -> None:
        del label
        index = flat_index(positions, device="cpu")
        expected_rows = [i for i, ps in enumerate(positions) for _ in ps]
        expected_cols = [p for ps in positions for p in ps]
        assert index.rows.tolist() == expected_rows
        assert index.cols.tolist() == expected_cols
        assert index.counts.tolist() == [len(ps) for ps in positions]

    def test_an_empty_selection_contributes_no_pairs(self) -> None:
        """Representable at this level — the engine refuses it higher up, where
        it can say which unit and example are at fault."""
        index = flat_index([[1, 2], []], device="cpu")
        assert index.counts.tolist() == [2, 0]
        assert index.rows.tolist() == [0, 0]


class TestGatherScatterAgainstAPerExampleLoop:
    """The batched primitives must equal handling each example on its own."""

    @pytest.mark.parametrize("label,positions", COUNT_PATTERNS, ids=lambda v: v)
    def test_gather_matches_the_loop(self, label, positions) -> None:
        del label
        torch.manual_seed(0)
        activation = torch.randn(len(positions), 8, WIDTH)
        got = gather_positions(activation, flat_index(positions, device="cpu"))
        want = torch.cat(
            [activation[i, torch.tensor(ps)] for i, ps in enumerate(positions) if ps],
            dim=0,
        )
        torch.testing.assert_close(got, want)

    @pytest.mark.parametrize("label,positions", COUNT_PATTERNS, ids=lambda v: v)
    def test_scatter_matches_the_loop(self, label, positions) -> None:
        del label
        torch.manual_seed(0)
        activation = torch.randn(len(positions), 8, WIDTH)
        total = sum(len(ps) for ps in positions)
        values = torch.randn(total, WIDTH)

        batched = activation.clone()
        scatter_positions(batched, flat_index(positions, device="cpu"), values)

        looped = activation.clone()
        offset = 0
        for i, ps in enumerate(positions):
            if ps:
                looped[i, torch.tensor(ps)] = values[offset : offset + len(ps)]
            offset += len(ps)
        torch.testing.assert_close(batched, looped)

    def test_scatter_touches_only_the_named_positions(self) -> None:
        """A write must not disturb a token the unit did not select — the failure
        that would make an intervention look stronger than it is."""
        torch.manual_seed(0)
        activation = torch.randn(2, 6, WIDTH)
        before = activation.clone()
        index = flat_index([[1], [4]], device="cpu")
        scatter_positions(activation, index, torch.zeros(2, WIDTH))
        untouched = [(0, p) for p in (0, 2, 3, 4, 5)] + [
            (1, p) for p in (0, 1, 2, 3, 5)
        ]
        for example, position in untouched:
            torch.testing.assert_close(
                activation[example, position], before[example, position]
            )
        assert (activation[0, 1] == 0).all() and (activation[1, 4] == 0).all()


class TestAlignPerExample:
    """A per-example value must be repeated by that example's count.

    The regression that matters: reading it as one-row-per-position instead. Both
    readings produce a correctly-shaped tensor, so nothing downstream complains —
    the values are simply attributed to the wrong examples.
    """

    @pytest.mark.parametrize("label,positions", COUNT_PATTERNS, ids=lambda v: v)
    def test_per_example_vector_repeats_by_its_own_count(
        self, label, positions
    ) -> None:
        del label
        index = flat_index(positions, device="cpu")
        aligned = align_per_example(_per_example_source(len(positions)), index)
        # Row j must carry the vector of the example that contributed pair j.
        expected = torch.tensor(
            [float(i) for i, ps in enumerate(positions) for _ in ps]
        )
        assert aligned.shape == (index.rows.shape[0], WIDTH)
        torch.testing.assert_close(aligned[:, 0], expected)

    def test_zero_count_example_never_contributes_a_row(self) -> None:
        """The exact shape coincidence that broke this: with counts ``(2, 0)`` the
        batch size equals the total, so a length check alone cannot tell a
        per-example value from a per-position one — and the wrong reading puts
        example 1's vector on example 0's second token."""
        index = flat_index([[1, 2], []], device="cpu")
        aligned = align_per_example(_per_example_source(2), index)
        assert aligned.shape[0] == 2
        torch.testing.assert_close(aligned[:, 0], torch.tensor([0.0, 0.0]))

    def test_a_per_position_source_passes_through_untouched(self) -> None:
        """An interchange source is gathered through this same index, so it
        already has one row per pair and must not be repeated again."""
        index = flat_index([[1, 2], [3]], device="cpu")
        source = torch.randn(3, WIDTH)
        torch.testing.assert_close(align_per_example(source, index), source)

    def test_a_source_with_its_own_position_axis_flattens(self) -> None:
        """``(batch, n_positions, width)`` is already one row per pair, just not
        flattened yet."""
        index = flat_index([[1, 2], [3, 4]], device="cpu")
        source = torch.randn(2, 2, WIDTH)
        torch.testing.assert_close(
            align_per_example(source, index), source.reshape(4, WIDTH)
        )


class TestEmptySelectionIsRefused:
    """A unit that selects nothing for some example must not run.

    Widths may differ across a batch now, and zero is a width like any other as
    far as the indexing is concerned: that example contributes no pairs and is
    left untouched, then scored beside examples that *were* intervened on. A
    wrong number rather than a crash, so it is refused where the message can name
    the unit. Before spans could be ragged the uniform-width guard caught this
    incidentally; it no longer exists.
    """

    def _unit_and_positions(self, pipeline, positions):
        from causalab.neural.LM_units import ResidualStream
        from causalab.neural.token_positions import TokenPosition

        hidden = pipeline.model.config.hidden_size
        unit = ResidualStream(
            layer=0,
            token_indices=TokenPosition(lambda _x: [1], pipeline, id="p1"),
            target_output=True,
            shape=(hidden,),
        )
        return [unit], [positions]

    def test_an_example_selecting_nothing_raises(self, mock_tiny_lm) -> None:
        from causalab.neural.activations.engine import build_plans

        units, positions = self._unit_and_positions(mock_tiny_lm, [[1, 2], []])
        with pytest.raises(ValueError, match="selects no token positions"):
            build_plans(units, positions, "collect")

    def test_the_message_names_the_offending_examples(self, mock_tiny_lm) -> None:
        from causalab.neural.activations.engine import build_plans

        units, positions = self._unit_and_positions(mock_tiny_lm, [[1], [], [2], []])
        with pytest.raises(ValueError, match=r"\[1, 3\]"):
            build_plans(units, positions, "collect")

    def test_a_ragged_but_non_empty_batch_is_still_fine(self, mock_tiny_lm) -> None:
        """The guard must reject *empty*, not *ragged* — those were the same
        thing before and are not any more."""
        from causalab.neural.activations.engine import build_plans

        units, positions = self._unit_and_positions(mock_tiny_lm, [[1, 2], [3]])
        assert len(build_plans(units, positions, "collect")) == 1
