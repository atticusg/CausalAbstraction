"""Sweep-axis discovery and deterministic expansion (spec §3).

A document with ``{"sweep": …}`` wrappers denotes a *set* of point
protocols: every axis is an explicit wrapper on a field (or whole entry) of
a named table, the axes cross-multiply, and expansion is a pure function of
the document — one document ⇒ the same ordered point list, always.

This module works on the **raw mapping** (the order-preserving tree
:func:`causalab.protocol.schema.load_raw` produces, after artifact
resolution), not on the parsed dataclasses: substituting one axis value is a
tree edit, and re-parsing each concrete point through
:func:`~causalab.protocol.schema.parse_document` afterwards guarantees a
point is exactly as valid as the same document written by hand.

Axis identity is name identity (§3): the axis id is the dotted path of the
wrapped field (``sites.target.layer``, ``positions.tap``,
``featurizers.rot.k``, ``train.seed``), axes are ordered by first appearance
in the document, and the cross product iterates the *last* axis fastest —
so point order is the lexicographic order of coordinate indices.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Mapping

from causalab.protocol.errors import ValidationError

__all__ = ["Axis", "Expansion", "Point", "expand", "find_axes"]


#: Refuse a cross product larger than this without an explicit override
#: (§5.14 — "may be capped without an explicit override flag").
DEFAULT_POINT_CAP = 4096


@dataclasses.dataclass(frozen=True)
class Axis:
    """One sweep axis: the dotted path of the wrapped value and its values.

    ``path`` addresses the wrapper's location in the raw tree
    (``("sites", "target", "layer")``); ``id`` is its dotted spelling — the
    coordinate key in results and derived names.
    """

    path: tuple[str, ...]
    values: tuple[Any, ...]

    @property
    def id(self) -> str:
        return ".".join(self.path)


@dataclasses.dataclass(frozen=True)
class Point:
    """One expanded point: coordinates plus the concrete raw document."""

    coords: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Expansion:
    """The result of expanding one document: its axes (document order) and
    the full cross product (last axis fastest)."""

    axes: tuple[Axis, ...]
    points: tuple[Point, ...]

    @property
    def is_swept(self) -> bool:
        return bool(self.axes)


def _is_sweep_node(node: Any) -> bool:
    return isinstance(node, dict) and "sweep" in node


def _sweep_values(node: Mapping[str, Any], path: tuple[str, ...]) -> tuple[Any, ...]:
    """The value list a wrapper denotes; a range object is sugar for the list
    it expands to (§3). Shape errors are §5.14 rejections — the parser
    already checks the shapes it can see, but expansion may encounter
    wrappers in positions the schema types as free-form."""
    spec = node["sweep"]
    if len(node) != 1:
        raise ValidationError(
            14, "a sweep wrapper holds nothing but the axis", path=".".join(path)
        )
    if isinstance(spec, Mapping):
        keys = set(spec)
        if keys != {"range"}:
            raise ValidationError(
                14,
                "a sweep object form takes exactly {'range': [...]}",
                path=".".join(path),
            )
        rng = spec["range"]
        if (
            not isinstance(rng, list)
            or not 2 <= len(rng) <= 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng)
        ):
            raise ValidationError(
                14,
                "sweep range must be [start, stop] or [start, stop, step] of integers",
                path=".".join(path),
            )
        step = rng[2] if len(rng) == 3 else 1
        if step == 0:
            raise ValidationError(
                14, "sweep range step must be non-zero", path=".".join(path)
            )
        values: tuple[Any, ...] = tuple(range(rng[0], rng[1], step))
    elif isinstance(spec, list):
        values = tuple(spec)
    else:
        raise ValidationError(
            14,
            f"a sweep wrapper takes a list or a range object, got {type(spec).__name__}",
            path=".".join(path),
        )
    if not values:
        raise ValidationError(
            14, "a sweep axis must have at least one value", path=".".join(path)
        )
    if _any_nested_wrapper(values):
        raise ValidationError(
            14,
            "sweep values may not contain nested sweep wrappers",
            path=".".join(path),
        )
    return values


def _any_nested_wrapper(node: Any) -> bool:
    if _is_sweep_node(node):
        return True
    if isinstance(node, Mapping):
        return any(_any_nested_wrapper(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_any_nested_wrapper(v) for v in node)
    return False


def find_axes(raw: Mapping[str, Any]) -> tuple[Axis, ...]:
    """Every sweep axis in the document, in order of first appearance.

    Wrappers are only discovered under string keys (table entries and their
    fields) — a wrapper *inside a list* (e.g. inside an authored ``dims``
    list) has no name identity and is rejected.
    """
    axes: list[Axis] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if _is_sweep_node(node):
            axes.append(Axis(path=path, values=_sweep_values(node, path)))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, path + (str(key),))
        elif isinstance(node, list):
            for item in node:
                if _any_nested_wrapper(item):
                    raise ValidationError(
                        14,
                        "a sweep wrapper inside a list has no name identity; "
                        "declare the axis on a named field",
                        path=".".join(path),
                    )

    walk(raw, ())
    return tuple(axes)


def _substitute(
    node: Any, assignment: Mapping[tuple[str, ...], Any], path: tuple[str, ...]
) -> Any:
    if _is_sweep_node(node):
        return assignment[path]
    if isinstance(node, Mapping):
        return {
            key: _substitute(value, assignment, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_substitute(item, assignment, path) for item in node]
    return node


def expand(
    raw: Mapping[str, Any], *, point_cap: int | None = DEFAULT_POINT_CAP
) -> Expansion:
    """Expand a raw document into its point protocols (§3).

    The cross product of all axes, last axis fastest; a document with no
    axes expands to exactly itself. ``point_cap`` refuses accidental
    combinatorial explosions (§5.14) — pass ``None`` for an explicit
    override.
    """
    axes = find_axes(raw)
    if not axes:
        return Expansion(axes=(), points=(Point(coords={}, raw=raw),))
    total = 1
    for axis in axes:
        total *= len(axis.values)
    if point_cap is not None and total > point_cap:
        raise ValidationError(
            14,
            f"sweep expands to {total} points, over the cap of {point_cap}; "
            "pass an explicit override to run a campaign this large",
        )
    points: list[Point] = []
    for combo in _cross(axes):
        assignment = {axis.path: value for axis, value in zip(axes, combo)}
        coords = {axis.id: value for axis, value in zip(axes, combo)}
        points.append(Point(coords=coords, raw=_substitute(raw, assignment, ())))
    return Expansion(axes=axes, points=tuple(points))


def _cross(axes: tuple[Axis, ...]) -> Iterator[tuple[Any, ...]]:
    if not axes:
        yield ()
        return
    head, *rest = axes
    for value in head.values:
        for tail in _cross(tuple(rest)):
            yield (value, *tail)


def coordinate_label(coords: Mapping[str, Any], *, entry: str | None = None) -> str:
    """The ``[k=8]`` / ``[target.layer=5]`` suffix for derived names (§3).

    Coordinates on the named entry itself drop the entry prefix
    (``rot[k=8]``, not ``rot[featurizers.rot.k=8]``); transitive coordinates
    keep ``<entry>.<field>``; the table name is always dropped. ``train``
    axes read best bare (``seed=0``)."""
    parts: list[str] = []
    for axis_id, value in coords.items():
        segments = axis_id.split(".")
        if len(segments) >= 2 and segments[0] in (
            "positions",
            "sites",
            "featurizers",
            "params",
            "reads",
            "edits",
            "intervened_models",
            "metrics",
            "train",
        ):
            segments = segments[1:]
        if entry is not None and len(segments) >= 2 and segments[0] == entry:
            segments = segments[1:]
        parts.append(f"{'.'.join(segments)}={_label_value(value)}")
    return f"[{','.join(parts)}]" if parts else ""


def _label_value(value: Any) -> str:
    if isinstance(value, Mapping):
        # a swept spec (e.g. a position): label by its single distinguishing pair
        pairs = ",".join(f"{k}:{v}" for k, v in value.items())
        return "{" + pairs + "}"
    return str(value)
