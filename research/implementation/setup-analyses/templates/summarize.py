"""Summarize one metric table for a workflow script step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_table, write_table


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    rows = read_table(Path(inputs["table"]))
    if not rows:
        raise StepError("summarize: the input table is empty")
    column = str(inputs.get("value_column", "value"))
    if any(column not in row for row in rows):
        raise StepError(f"summarize: not every row contains {column!r}")
    values = [float(row[column]) for row in rows]
    write_table(
        Path(outputs["summary"]),
        [{"count": len(values), "mean": sum(values) / len(values)}],
    )
