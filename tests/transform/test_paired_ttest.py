"""``paired_ttest@1`` against a hand-computed oracle.

Differences 0.5, 1.0, 0.5, 1.0 → mean 0.75, sample sd sqrt(0.25/3) = 0.288675,
so t = 0.75 / (0.288675 / 2) = 5.196152 on 3 degrees of freedom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from causalab.transform.ops.paired_ttest import paired_ttest
from causalab.transform.schema import TransformError

pytestmark = pytest.mark.numerical_unit

A = pd.DataFrame({"example": [0, 1, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]})
B = pd.DataFrame({"example": [0, 1, 2, 3], "value": [0.5, 1.0, 2.5, 3.0]})
DEFAULTS = {"value_column": "value", "pair_column": "example"}


def test_statistics_match_the_oracle() -> None:
    stats = paired_ttest(inputs={"a": A, "b": B}, params=DEFAULTS)["stats"]
    assert list(stats.columns) == [
        "n_pairs",
        "df",
        "mean_difference",
        "t_statistic",
        "p_value",
    ]
    row = stats.iloc[0]
    assert row["n_pairs"] == 4 and row["df"] == 3
    assert row["mean_difference"] == pytest.approx(0.75)
    assert row["t_statistic"] == pytest.approx(5.196152, abs=1e-6)
    assert row["p_value"] == pytest.approx(0.013847, abs=1e-6)


def test_the_test_is_two_sided() -> None:
    """Swapping the arms flips the statistic's sign and leaves p alone."""
    forward = paired_ttest(inputs={"a": A, "b": B}, params=DEFAULTS)["stats"].iloc[0]
    reverse = paired_ttest(inputs={"a": B, "b": A}, params=DEFAULTS)["stats"].iloc[0]
    assert reverse["t_statistic"] == pytest.approx(-forward["t_statistic"])
    assert reverse["p_value"] == pytest.approx(forward["p_value"])


def test_rows_are_averaged_within_a_pair_before_the_test() -> None:
    """A swept document writes one row per (example, point); a bare join would
    multiply rows and inflate n."""
    doubled = pd.concat([A, A], ignore_index=True)
    once = paired_ttest(inputs={"a": A, "b": B}, params=DEFAULTS)["stats"].iloc[0]
    twice = paired_ttest(inputs={"a": doubled, "b": B}, params=DEFAULTS)["stats"].iloc[
        0
    ]
    assert twice["n_pairs"] == once["n_pairs"] == 4
    assert twice["t_statistic"] == pytest.approx(once["t_statistic"])


def test_only_shared_pairs_are_compared() -> None:
    partial = pd.DataFrame({"example": [0, 1], "value": [0.5, 1.0]})
    stats = paired_ttest(inputs={"a": A, "b": partial}, params=DEFAULTS)["stats"]
    assert stats.iloc[0]["n_pairs"] == 2


def test_a_constant_difference_is_reported_rather_than_fudged() -> None:
    """Zero spread makes the statistic degenerate; inf/0.0 is honest, an
    epsilon in the denominator would not be."""
    shifted = A.assign(value=A["value"] + 1.0)
    stats = paired_ttest(inputs={"a": shifted, "b": A}, params=DEFAULTS)["stats"]
    assert stats.iloc[0]["t_statistic"] == float("inf")
    assert stats.iloc[0]["p_value"] == 0.0


def test_identical_arms_are_not_significant() -> None:
    stats = paired_ttest(inputs={"a": A, "b": A}, params=DEFAULTS)["stats"]
    assert stats.iloc[0]["t_statistic"] == 0.0
    assert stats.iloc[0]["p_value"] == 1.0


def test_too_few_pairs_is_refused() -> None:
    one = pd.DataFrame({"example": [0], "value": [1.0]})
    with pytest.raises(TransformError) as err:
        paired_ttest(inputs={"a": one, "b": one}, params=DEFAULTS)
    assert "at least 2" in str(err.value)


def test_a_missing_column_names_the_offending_side() -> None:
    with pytest.raises(TransformError) as err:
        paired_ttest(
            inputs={"a": A, "b": B.rename(columns={"value": "score"})}, params=DEFAULTS
        )
    assert "input 'b'" in str(err.value)
