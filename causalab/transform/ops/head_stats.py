"""``head_stats@1`` — a per-(layer, head) summary of a metric table.

The table→table direction of the IO contract. Deterministic: a sorted
group-by over committed bytes, with a *population* standard deviation so a
one-row cell yields ``0.0`` rather than a NaN that would then travel silently
into a plot.
"""

from __future__ import annotations

from typing import Any, Mapping

from causalab.transform.registry import register
from causalab.transform.schema import Str, Table, TransformError

__all__ = ["head_stats"]


@register(
    name="head_stats",
    version=1,
    inputs={
        "table": Table(
            description="a metric table; its sweep-coordinate columns are "
            "whatever the producing document stamped, so they are not declared"
        )
    },
    outputs={
        "stats": Table(
            columns={
                "layer": "int64",
                "head": "int64",
                "n": "int64",
                "mean": "float64",
                "std": "float64",
            },
            description="one row per (layer, head) cell, layer-then-head sorted",
        )
    },
    params={
        "layer_column": Str(
            default="sites.target.layer", description="the swept layer axis"
        ),
        "head_column": Str(
            default="sites.target.head", description="the swept head axis"
        ),
        "value_column": Str(default="value", description="the column summarized"),
    },
    description="Mean and spread of a metric over each (layer, head) cell.",
)
def head_stats(
    *, inputs: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    frame = inputs["table"]
    layer_column = str(params["layer_column"])
    head_column = str(params["head_column"])
    value_column = str(params["value_column"])
    for column in (layer_column, head_column, value_column):
        if column not in frame.columns:
            raise TransformError(
                f"head_stats@1: the input table has no column {column!r} "
                f"(has {sorted(map(str, frame.columns))})"
            )
    grouped = frame.groupby([layer_column, head_column], sort=True)[value_column]
    # ddof=0 on purpose: a cell with one row has no sample spread, and 0.0 is
    # the honest answer for "how far apart are these", where NaN would poison
    # every downstream mean.
    stats = grouped.agg(n="count", mean="mean", std=lambda s: s.std(ddof=0))
    stats = stats.reset_index()
    stats = stats.rename(columns={layer_column: "layer", head_column: "head"})
    return {"stats": stats[["layer", "head", "n", "mean", "std"]]}
