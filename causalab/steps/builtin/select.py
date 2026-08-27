"""``causalab:select`` — reduce a metric table to named values.

The stage-1 → stage-2 seam: turn a swept metric table into the scalar(s) the
next protocol needs (the locate → DAS handoff), as data instead of a notebook.

```json
"best": {
  "type": "script", "script": "causalab:select",
  "inputs": {
    "table": {"step": "locate", "file": "iia.json"},
    "choose": "max",
    "emit": {"best_layer": "sites.target.layer", "best_pos": "positions.tap"}
  },
  "outputs": {"values": {"file": "values.json",
                         "keys": {"best_layer": 0, "best_pos": {"index": -1}}}}
}
```

Rows are grouped by the producing document's **sweep-coordinate columns** — read
from the step's ``_step.json`` (:mod:`._sidecar`), not authored — and aggregated
by mean over examples. ``choose`` then picks the best group and ``emit`` reads
that group's columns. The exact rule, including when a table is ranked *as
written*, is :func:`._sidecar.aggregate`.

Two behaviours v1 had as spec rules and this has as script behaviour, on
purpose: the axes come from published data rather than from the document model,
and the as-written case is decided by the data rather than by the producing
step's type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from causalab.steps.builtin._sidecar import aggregate
from causalab.steps.io import StepError, frame, write_values

__all__ = ["main"]

CHOICES = ("max", "min")


def _decode(value: Any) -> Any:
    """Coordinate cells round-trip through a metric table as JSON when they are
    structured (a swept position spec); decode them back so what lands in the
    values object is what a consuming document can parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "item"):
        return value.item()
    return value


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table_path = Path(inputs["table"])
    choose = str(inputs.get("choose", "max"))
    if choose not in CHOICES:
        raise StepError(f"'choose' is one of {list(CHOICES)}, got {choose!r}")
    value_column = str(inputs.get("value", "value"))
    emit = inputs.get("emit")
    if not isinstance(emit, Mapping) or not emit:
        raise StepError("'emit' maps output key to the column it reads")

    df = frame(table_path)
    if df.empty:
        raise StepError(f"{table_path.name} has no rows to select from")
    if value_column not in df.columns:
        raise StepError(
            f"{table_path.name} has no column {value_column!r} "
            f"(has {sorted(map(str, df.columns))})"
        )

    grouped, _ = aggregate(df, table_path, value_column)

    index = (
        grouped[value_column].idxmax()
        if choose == "max"
        else grouped[value_column].idxmin()
    )
    # single-row FRAME indexing: a row Series would upcast mixed dtypes and
    # emit integer coordinates as floats, which a consuming document's strict
    # parse then refuses
    row = grouped.loc[[index]]

    values: dict[str, Any] = {}
    for key, column in emit.items():
        if column not in row.columns:
            raise StepError(
                f"emit column {column!r} is not in the aggregated table "
                f"({sorted(map(str, row.columns))}) — the producing run "
                "carried no such axis"
            )
        values[str(key)] = _decode(row[column].iloc[0])
    # only the emitted keys land here: the values object is what a consuming
    # document reads by name, and a winning score nobody declared would be an
    # undeclared key in a file whose shape is part of the record
    write_values(outputs["values"], values)
