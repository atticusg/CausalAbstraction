"""Resolve the ablation ``span`` config into a single ``TokenPosition``.

``span: all`` ablates every token position; a list of names
(``span: [last_token, ...]``) ablates the union of those named task positions.
Returning one ``TokenPosition`` (with a stable ``id``) keeps the rest of the
analysis span-agnostic — the builders and heatmap keys treat it like any other
position.
"""

from __future__ import annotations

from typing import Any

from causalab.neural.token_positions import TokenPosition, get_all_tokens


def resolve_span(span: Any, task: Any, pipeline: Any, dataset: list) -> TokenPosition:
    """Resolve ``span`` (the string ``"all"`` or a list of names) to one position.

    For named spans, the names are looked up among ``task.create_token_positions``
    by ``id``; multiple names are unioned into a single indexer (sorted, unique)
    so the component is ablated at every named position at once.
    """
    if isinstance(span, str):
        if span == "all":
            return get_all_tokens(dataset[0]["input"], pipeline)
        names = [span]
    else:
        names = list(span)

    # Tasks return either a {name: TokenPosition} dict or a list of positions.
    positions = task.create_token_positions(pipeline)
    available = (
        dict(positions)
        if isinstance(positions, dict)
        else {tp.id: tp for tp in positions}
    )
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"Unknown token-position name(s) {missing} for task '{task.name}'. "
            f"Available: {sorted(available)}. Use 'all' for every position."
        )

    selected = [available[name] for name in names]
    if len(selected) == 1:
        return selected[0]

    def union_indexer(inp: Any) -> list[int]:
        indices: set[int] = set()
        for tp in selected:
            indices.update(tp.index(inp))
        return sorted(indices)

    return TokenPosition(union_indexer, pipeline, id="+".join(names))
