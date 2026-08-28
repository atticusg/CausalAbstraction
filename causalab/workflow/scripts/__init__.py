"""Step scripts whose purpose *is* wiring a workflow together.

Only what belongs to the workflow layer itself lives here. Numerical analysis is
:mod:`causalab.analysis`; rendering is :mod:`causalab.io.plots`; the IO helpers a
script uses are :mod:`causalab.io.step_io`. A document addresses any of them the
same way — ``{"script": {"module": "…"}}`` — so this package has no privileged
status, only a narrow subject.

| module | what it does |
|---|---|
| ``select`` | reduce a metric table to named values, which a later document's ``set`` reads |

``select`` sits here rather than under ``analysis/`` because its output exists to
be consumed by the *next step* rather than by a reader: it is the stage-1 →
stage-2 seam expressed as data instead of a notebook.
"""

from __future__ import annotations

__all__: list[str] = []
