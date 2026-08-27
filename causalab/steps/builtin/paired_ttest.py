"""``causalab:paired_ttest`` — two-sided paired t-test of two metric tables.

```json
"compare": {
  "type": "script", "script": "causalab:paired_ttest",
  "inputs": {"a": {"step": "arm_a", "file": "iia.json"},
             "b": {"step": "arm_b", "file": "iia.json"}},
  "outputs": {"stats": {"file": "ttest.json",
                        "columns": {"n_pairs": "int64", "df": "int64",
                                    "mean_difference": "float64",
                                    "t_statistic": "float64",
                                    "p_value": "float64"}}}
}
```

Formerly ``paired_ttest@1`` in the transform-op registry; numerics unchanged.
It is also the two-input case, which is why it survived the port as a shipped
script rather than an example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.steps.io import StepError, frame, write_table

__all__ = ["main"]


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import pandas as pd
    from scipy import stats as scipy_stats

    value_column = str(inputs.get("value_column", "value"))
    pair_column = str(inputs.get("pair_column", "example"))
    sides = {}
    for slot in ("a", "b"):
        table = frame(Path(inputs[slot]))
        for column in (pair_column, value_column):
            if column not in table.columns:
                raise StepError(
                    f"paired_ttest: input {slot!r} has no column {column!r} "
                    f"(has {sorted(map(str, table.columns))})"
                )
        sides[slot] = (
            table.groupby(pair_column, sort=True)[value_column].mean().rename(slot)
        )
    joined = pd.concat([sides["a"], sides["b"]], axis=1, join="inner").sort_index()
    n = int(len(joined))
    if n < 2:
        raise StepError(
            f"paired_ttest: {n} pair(s) shared between the two tables — a "
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
    write_table(
        Path(outputs["stats"]),
        [
            {
                "n_pairs": n,
                "df": df,
                "mean_difference": mean,
                "t_statistic": float(t_statistic),
                "p_value": p_value,
            }
        ],
    )
