"""How many token positions a unit selects for one example.

This used to also bucket a dataset into equal-position-count groups, because the
gather produced a rectangular ``(b, [h,] n_pos, d)`` tensor and every example in
a batch had to expose the same number of positions. Positions are now indexed
per selected token, so a batch may mix widths and the bucketing is gone.

The count itself is still needed: per-example *replace* collection reads one
activation row per example, so it has to reject a span that resolves to more
than one token.
"""

from __future__ import annotations

from typing import Any

from causalab.neural.units import AtomicModelUnit


def unit_position_count(unit: AtomicModelUnit, example_input: Any) -> int:
    """Number of token positions ``unit`` gathers for a single example.

    ``AttentionHead.index_component`` returns ``[head_axis, position_axis]``; the
    position axis (``idx[1]``, a one-row list-of-lists for a single example) is
    what carries the position count — reading ``len(idx)`` would return the
    axis count (2) and ``len(idx[0])`` the head count. ``pos`` units return the
    position list directly. ``attention_mask=None`` keeps indices in the
    example's own unpadded frame, which is all we need for counting.
    """
    # index_component's return shape is unit-type dependent (nested lists whose
    # depth varies), so it's typed broadly; treat as Any for the structural indexing.
    idx: Any = unit.index_component(example_input, batch=False, attention_mask=None)
    if unit.unit == "h.pos":
        return len(idx[1][0])
    return len(idx)
