"""``causalab:plot`` — heatmap and lines over a metric table.

```json
"scan_heatmap": {
  "type": "script", "script": "causalab:plot",
  "inputs": {"table": {"step": "locate", "file": "iia.json"},
             "plot": "heatmap", "x": "sites.target.layer", "y": "positions.tap",
             "figure": "scan_iia.png"},
  "outputs": {"plotted": {"file": "scan_iia.json"}}
}
```

Two kinds cover the pipeline: scan grids and metric-vs-axis curves. In v1 that
was a closed spec enum; here it is just what this script does, so a third kind
is a user script rather than a spec change — which is the point of the step
type.

Rows are aggregated exactly as ``causalab:select`` does — group by the producing
document's sweep axes (from ``_step.json``), mean over examples — so a figure
and the values chosen from the same table always agree.

**A figure must cover every axis of what it renders.** An uncovered axis would
silently collapse into duplicate cells, averaging over a dimension the reader
cannot see. v1 refused that at load; here it is refused at the step, which is
the honest place now that the axes are read from published data.

**What this step declares is the aggregated table, not the image.** A ``.png``
is neither JSON nor safetensors, and the two-format rule (§2.5) admits no third
— so the *declared* output is the exact rows that were plotted, which is better
provenance than the picture anyway: a reader can recompute the figure, and a
later step can reference the numbers. The image is written beside it and is
published by sitting in the step directory (§0, everything declared is
published — and a step directory is the publication). Its name comes from the
``figure`` input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.steps.builtin._sidecar import aggregate, axes_for
from causalab.steps.io import StepError, frame, write_frame

__all__ = ["main"]

KINDS = ("heatmap", "lines")


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table_path = Path(inputs["table"])
    kind = str(inputs.get("plot", "heatmap"))
    if kind not in KINDS:
        raise StepError(f"'plot' is one of {list(KINDS)}, got {kind!r}")
    value_column = str(inputs.get("value", "value"))
    x = inputs.get("x")
    if not isinstance(x, str):
        raise StepError("'x' names the horizontal-axis column")
    y = inputs.get("y")
    series = inputs.get("series")
    if kind == "heatmap" and not isinstance(y, str):
        raise StepError("a heatmap needs 'y'")

    df = frame(table_path)
    if value_column not in df.columns:
        raise StepError(
            f"{table_path.name} has no column {value_column!r} "
            f"(has {sorted(map(str, df.columns))})"
        )
    axes = [axis for axis in axes_for(table_path) if axis in df.columns]
    covered = {c for c in (x, y, series) if isinstance(c, str)}
    for column in covered:
        if column not in df.columns:
            raise StepError(
                f"column {column!r} is not in {table_path.name} "
                f"(has {sorted(map(str, df.columns))})"
            )
    uncovered = [axis for axis in axes if axis not in covered]
    if uncovered:
        raise StepError(
            f"the plot leaves the axes {sorted(uncovered)} uncovered — every "
            "sweep axis must be x, y or series, or the figure silently averages "
            "over a dimension the reader cannot see"
        )
    table, _ = aggregate(df, table_path, value_column)

    figure, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if kind == "heatmap":
        grid = table.pivot_table(
            index=str(y), columns=x, values=value_column, aggfunc="mean"
        )
        image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
        ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
        ax.set_xlabel(x)
        ax.set_ylabel(str(y))
        figure.colorbar(image, ax=ax, label=value_column)
    else:
        if isinstance(series, str):
            for value, group in table.groupby(series, sort=True):
                ordered = group.sort_values(x)
                ax.plot(
                    ordered[x],
                    ordered[value_column],
                    marker="o",
                    label=f"{series}={value}",
                )
            ax.legend()
        else:
            ordered = table.sort_values(x)
            ax.plot(ordered[x], ordered[value_column], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(value_column)
    ax.set_title(f"{table_path.name} — {value_column}")

    # the declared output is the data; the image lands beside it (docstring)
    plotted = Path(next(iter(outputs.values())))
    figure_name = str(inputs.get("figure", "figure.png"))
    if "/" in figure_name or figure_name.startswith("."):
        raise StepError(
            f"'figure' is a bare filename inside the step directory, got "
            f"{figure_name!r}"
        )
    plotted.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plotted.parent / figure_name)
    plt.close(figure)
    write_frame(table, plotted)
