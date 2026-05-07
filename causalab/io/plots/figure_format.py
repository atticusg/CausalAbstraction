"""Shared figure output format for static matplotlib figures (PNG / PDF)."""

from __future__ import annotations

import os
from typing import Literal

FigureFormat = Literal["png", "pdf"]

ALLOWED_FIGURE_FORMATS: frozenset[str] = frozenset({"png", "pdf"})


def normalize_figure_format(value: str | None, *, default: str = "pdf") -> str:
    """Return ``png`` or ``pdf``; validate input."""
    raw = default if value is None else str(value)
    fmt = raw.lower().lstrip(".")
    if fmt not in ALLOWED_FIGURE_FORMATS:
        raise ValueError(
            f"figure_format must be one of {sorted(ALLOWED_FIGURE_FORMATS)}, got {value!r}"
        )
    return fmt


def path_with_figure_format(path: str, figure_format: str | None) -> str:
    """Set or replace the file extension on ``path`` using ``figure_format``."""
    fmt = normalize_figure_format(figure_format, default="pdf")
    root, _ext = os.path.splitext(path)
    return f"{root}.{fmt}"


def resolve_figure_format_from_analysis(analysis) -> str:
    """Read ``analysis.visualization.figure_format`` (Hydra / OmegaConf / dict)."""
    vis = None
    if hasattr(analysis, "get"):
        vis = analysis.get("visualization")
    if not vis:
        return normalize_figure_format(None, default="pdf")
    raw = (
        vis.get("figure_format", "pdf") if hasattr(vis, "get") else vis["figure_format"]
    )
    return normalize_figure_format(raw, default="pdf")
