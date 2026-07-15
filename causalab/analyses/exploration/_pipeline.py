"""Shared pipeline construction for the exploration CLIs.

Every exploration CLI builds an :class:`~causalab.neural.pipeline.LMPipeline`
the same way — optional ``--device`` / ``--dtype`` overrides, default auto
otherwise. Centralizing it here keeps the CLIs thin and gives the
device/dtype plumbing one tested home instead of a copy per script.
"""

from __future__ import annotations

from causalab.neural.pipeline import LMPipeline

# Maps the ``--dtype`` choice string to the matching ``torch`` attribute name.
DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def build_pipeline(
    model: str,
    max_new_tokens: int,
    device: str | None = None,
    dtype: str | None = None,
) -> LMPipeline:
    """Build an ``LMPipeline``, applying optional device/dtype overrides.

    ``device`` and ``dtype`` left as ``None`` defer to ``LMPipeline``'s own
    auto-resolution. ``dtype`` must be one of :data:`DTYPE_MAP`.
    """
    kwargs: dict = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        import torch

        kwargs["dtype"] = getattr(torch, DTYPE_MAP[dtype])
    return LMPipeline(model, max_new_tokens=max_new_tokens, **kwargs)
