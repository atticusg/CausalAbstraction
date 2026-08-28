"""Metric tables on disk: native JSON, an array of row objects.

JSON and safetensors are the only two formats in the stack (IM spec §2.12,
workflow spec §2.5). A metric table is the structured, readable half of that
pair, so it is a plain JSON array — nothing wrapping it, no envelope, no
column header:

.. code-block:: json

    [
      {"example": 0, "sites.target.layer": 18, "value": 0.83},
      {"example": 1, "sites.target.layer": 18, "value": 0.91}
    ]

Labels repeat on every row. That is the deliberate trade — a file ``jq`` and a
human can both read, at the cost of size — and it is why the *schema* promise
lives in a step's ``outputs`` declaration rather than inside the file: an empty
table has no rows to infer from, and a declaration still covers that case.

This module lives in ``protocol/`` because it is **torch-free** and both sides
of the stack need it: the reference engine writes tables through it, and the
workflow layer's step scripts read them. It owns no pandas dependency either —
callers that want a DataFrame build one from the rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.errors import ProtocolError

__all__ = ["TABLE_SUFFIX", "read_table", "write_table"]

#: The one extension a metric table may carry.
TABLE_SUFFIX = ".json"


def write_table(target: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ``rows`` as a metric table.

    Non-finite floats become ``null``: ``json.dumps`` would otherwise emit the
    bare tokens ``NaN``/``Infinity``, which Python reads back but no other JSON
    parser accepts — and a metric that computed nothing is exactly the "no
    value" a ``null`` means (the same choice the per-step ``matched`` flag
    encodes for continuation reads)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [{key: _finite(value) for key, value in row.items()} for row in rows]
    target.write_text(json.dumps(payload, indent=2) + "\n")


def read_table(path: Path) -> list[dict[str, Any]]:
    """One metric table back as a list of row dicts."""
    if not path.is_file():
        raise ProtocolError("P2", f"table {str(path)!r} does not exist")
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProtocolError(
            "P2",
            f"{path.name} is not a metric table — expected a JSON array of row objects",
        )
    return rows


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
