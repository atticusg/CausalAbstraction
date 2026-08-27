"""The per-step ``_step.json`` record: its format, its reader, and the shared
reduction rule that reads it.

The runner writes one per step (workflow spec §4): the files it declared, and
for a protocol step the **sweep axes** its document expanded to. v1 derived
group-by columns from those axes inside the document model; v2 publishes them
as data and lets a script read them, which is what keeps
group-by-coordinates-then-mean working now that ``select`` and ``plot`` are
scripts rather than step types (§6).

A script is handed *files*, not step names, so the sidecar is found beside the
input it was given — which also means a script works identically against a run
tree and against a hand-made directory in a test.

**Why this lives in ``io/`` rather than ``workflow/``.** It is a file format, and
both readers of it are outside the workflow package: the shipped ``select``
script and the ``io.plots`` renderer. Putting it here keeps the dependency one
way — ``workflow`` → ``io`` — where the reverse would be a cycle, since the
runner already reads ``io.step_io``. The runner writes the record through
:func:`write_sidecar`; everything that consumes one reads it through here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "EXAMPLE_COLUMN",
    "SIDECAR",
    "aggregate",
    "axes_for",
    "read_sidecar",
    "write_sidecar",
]

SIDECAR = "_step.json"


def read_sidecar(table_path: Path) -> dict[str, Any]:
    """The record for the step that wrote ``table_path``, or ``{}``.

    Absent is not an error: a script may be pointed at a table nobody's runner
    produced (a pinned file under the repo root, a fixture), and the reduction
    still has to work — it just has no axes to group by."""
    candidate = Path(table_path).parent / SIDECAR
    if not candidate.is_file():
        return {}
    try:
        with candidate.open() as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def axes_for(table_path: Path) -> tuple[str, ...]:
    """The sweep-axis column ids of the step that wrote ``table_path``."""
    record = read_sidecar(table_path)
    axes = record.get("axes")
    if not isinstance(axes, list):
        return ()
    return tuple(str(axis) for axis in axes)


#: The column a protocol run stamps per example. Its presence is what makes
#: "mean over examples" a meaningful reduction.
EXAMPLE_COLUMN = "example"


def aggregate(
    df: Any, table_path: Path, value_column: str
) -> tuple[Any, tuple[str, ...]]:
    """The table a reduction should work on, plus the axes it grouped by.

    Three cases, and the discriminator is the **data** rather than the kind of
    step that produced it (v1 special-cased a ``transform`` producer by type):

    1. the producer published sweep axes → group by them, mean over the rest;
    2. no axes but an ``example`` column → the whole table is one group, so the
       mean over examples is the single row to rank;
    3. no axes and no ``example`` column → the rows **are** the unit. A script
       that wrote one row per principal component already decided what a row
       means, and re-aggregating would collapse exactly the rows a consumer
       wants to choose between.

    Shared by ``select`` and ``plot`` on purpose: a figure and the value chosen
    from the same table must never disagree about what a row is.
    """
    import pandas as pd

    axes = tuple(axis for axis in axes_for(table_path) if axis in df.columns)
    if axes:
        return (
            df.groupby(list(axes), sort=True)[value_column].mean().reset_index(),
            axes,
        )
    if EXAMPLE_COLUMN in df.columns:
        return pd.DataFrame([{value_column: df[value_column].mean()}]), ()
    return df, ()


def write_sidecar(step_dir: Path, entry: Any) -> None:
    """Publish one step's record beside its outputs (workflow spec §4).

    ``axes`` is the load-bearing field: it is how a downstream script groups a
    swept table by its coordinate columns without the document model having to
    derive it (§6)."""
    (Path(step_dir) / SIDECAR).write_text(json.dumps(dict(entry), indent=2) + "\n")
