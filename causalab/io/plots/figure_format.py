"""Shared figure output format for static matplotlib figures (PNG / PDF)."""

from __future__ import annotations

import os
from typing import Literal

FigureFormat = Literal["png", "pdf"]

ALLOWED_FIGURE_FORMATS: frozenset[str] = frozenset({"png", "pdf"})


def normalize_figure_format(value: str | None, *, default: str = "png") -> str:
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
    fmt = normalize_figure_format(figure_format, default="png")
    root, _ext = os.path.splitext(path)
    return f"{root}.{fmt}"
