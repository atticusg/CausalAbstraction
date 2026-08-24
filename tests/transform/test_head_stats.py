"""``head_stats@1`` against a hand-computed oracle."""

from __future__ import annotations

import pandas as pd
import pytest

from causalab.transform.ops.head_stats import head_stats
from causalab.transform.schema import TransformError

pytestmark = pytest.mark.numerical_unit

TABLE = pd.DataFrame(
    {
        "sites.target.layer": [0, 0, 1, 1, 1],
        "sites.target.head": [0, 0, 0, 1, 1],
        "value": [1.0, 3.0, 5.0, 2.0, 4.0],
    }
)
DEFAULTS = {
    "layer_column": "sites.target.layer",
    "head_column": "sites.target.head",
    "value_column": "value",
}


def test_cells_match_the_oracle() -> None:
    """(0,0): {1,3} -> mean 2, population std 1. (1,0): {5} -> 5, 0.
    (1,1): {2,4} -> mean 3, population std 1."""
    stats = head_stats(inputs={"table": TABLE}, params=DEFAULTS)["stats"]
    assert list(stats.columns) == ["layer", "head", "n", "mean", "std"]
    assert stats.to_dict("records") == [
        {"layer": 0, "head": 0, "n": 2, "mean": 2.0, "std": 1.0},
        {"layer": 1, "head": 0, "n": 1, "mean": 5.0, "std": 0.0},
        {"layer": 1, "head": 1, "n": 2, "mean": 3.0, "std": 1.0},
    ]


def test_a_single_row_cell_has_zero_spread_not_nan() -> None:
    """ddof=0 on purpose: NaN here would poison every downstream mean."""
    stats = head_stats(inputs={"table": TABLE}, params=DEFAULTS)["stats"]
    lonely = stats[(stats["layer"] == 1) & (stats["head"] == 0)]
    assert lonely["std"].iloc[0] == 0.0


def test_rows_are_sorted_so_the_output_is_deterministic() -> None:
    shuffled = TABLE.iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    first = head_stats(inputs={"table": TABLE}, params=DEFAULTS)["stats"]
    second = head_stats(inputs={"table": shuffled}, params=DEFAULTS)["stats"]
    assert first.equals(second)


def test_a_missing_column_names_what_the_table_has() -> None:
    params = {**DEFAULTS, "head_column": "sites.target.expert"}
    with pytest.raises(TransformError) as err:
        head_stats(inputs={"table": TABLE}, params=params)
    assert "sites.target.expert" in str(err.value)
    assert "sites.target.head" in str(err.value)  # what it does have
