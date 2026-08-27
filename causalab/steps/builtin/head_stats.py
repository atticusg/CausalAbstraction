"""``causalab:head_stats`` — mean and spread of a metric over each (layer, head).

```json
"per_head": {
  "type": "script", "script": "causalab:head_stats",
  "inputs": {"table": {"step": "scan", "file": "iia.json"}},
  "outputs": {"stats": {"file": "head_stats.json",
                        "columns": {"layer": "int64", "head": "int64",
                                    "n": "int64", "mean": "float64",
                                    "std": "float64"}}}
}
```

Formerly ``head_stats@1`` in the transform-op registry; numerics unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.steps.io import StepError, frame, write_frame

__all__ = ["main"]


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table = frame(Path(inputs["table"]))
    layer_column = str(inputs.get("layer_column", "sites.target.layer"))
    head_column = str(inputs.get("head_column", "sites.target.head"))
    value_column = str(inputs.get("value_column", "value"))
    for column in (layer_column, head_column, value_column):
        if column not in table.columns:
            raise StepError(
                f"head_stats: the input table has no column {column!r} "
                f"(has {sorted(map(str, table.columns))})"
            )
    grouped = table.groupby([layer_column, head_column], sort=True)[value_column]
    # ddof=0 on purpose: a cell with one row has no sample spread, and 0.0 is
    # the honest answer for "how far apart are these", where NaN would poison
    # every downstream mean.
    stats = grouped.agg(n="count", mean="mean", std=lambda s: s.std(ddof=0))
    stats = stats.reset_index().rename(
        columns={layer_column: "layer", head_column: "head"}
    )
    write_frame(stats[["layer", "head", "n", "mean", "std"]], Path(outputs["stats"]))
