"""``paired_ttest@1`` — a paired t-test between two metric tables.

Proves the remaining two corners of the IO contract: **multiple input slots**,
and a table output that is a single row of statistics rather than a grid.

Pairing is by an explicit key column (``example`` in the metric tables the
backend writes), and both sides are averaged within a key first — a swept
document writes one row per (example, point), so a bare join would otherwise
multiply rows and inflate the sample size.
"""

from __future__ import annotations

from typing import Any, Mapping

from causalab.transform.registry import register
from causalab.transform.schema import Str, Table, TransformError

__all__ = ["paired_ttest"]


@register(
    name="paired_ttest",
    version=1,
    inputs={
        "a": Table(description="the first arm"),
        "b": Table(description="the second arm, paired with the first"),
    },
    outputs={
        "stats": Table(
            columns={
                "n_pairs": "int64",
                "df": "int64",
                "mean_difference": "float64",
                "t_statistic": "float64",
                "p_value": "float64",
            },
            description="one row: the two-sided paired t-test of a - b",
        )
    },
    params={
        "value_column": Str(default="value", description="the compared column"),
        "pair_column": Str(default="example", description="the pairing key"),
    },
    description="Two-sided paired t-test of the difference between two tables.",
)
def paired_ttest(
    *, inputs: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    import pandas as pd
    from scipy import stats as scipy_stats

    value_column = str(params["value_column"])
    pair_column = str(params["pair_column"])
    sides = {}
    for slot in ("a", "b"):
        frame = inputs[slot]
        for column in (pair_column, value_column):
            if column not in frame.columns:
                raise TransformError(
                    f"paired_ttest@1: input {slot!r} has no column {column!r} "
                    f"(has {sorted(map(str, frame.columns))})"
                )
        sides[slot] = (
            frame.groupby(pair_column, sort=True)[value_column].mean().rename(slot)
        )
    joined = pd.concat([sides["a"], sides["b"]], axis=1, join="inner").sort_index()
    n = int(len(joined))
    if n < 2:
        raise TransformError(
            f"paired_ttest@1: {n} pair(s) shared between the two tables — a "
            "paired t-test needs at least 2"
        )
    difference = joined["a"] - joined["b"]
    mean = float(difference.mean())
    spread = float(difference.std(ddof=1))
    df = n - 1
    if spread == 0.0:
        # every pair moved by exactly the same amount: the test is degenerate,
        # and reporting inf/0.0 is more honest than a NaN or a fudged epsilon.
        t_statistic = 0.0 if mean == 0.0 else float("inf") * (1.0 if mean > 0 else -1.0)
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        t_statistic = mean / (spread / (n**0.5))
        p_value = float(2.0 * scipy_stats.t.sf(abs(t_statistic), df))
    stats = pd.DataFrame(
        {
            "n_pairs": [n],
            "df": [df],
            "mean_difference": [mean],
            "t_statistic": [float(t_statistic)],
            "p_value": [p_value],
        }
    )
    return {"stats": stats}
