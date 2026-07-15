"""Greedy-decoding helpers over an :class:`LMPipeline`.

Small, analysis-neutral primitives for reading a model's greedy continuation of
a prompt. Lives in ``methods/`` so callers (analyses, skill templates) share one
implementation instead of re-deriving the ``pipeline.generate`` str-or-list
normalization (docs/CODEBASE.md invariant 4).
"""

from __future__ import annotations

from causalab.neural.pipeline import LMPipeline


def greedy_output(pipeline: LMPipeline, prompt: str) -> str:
    """Return the model's greedy continuation for one prompt, stripped.

    ``pipeline.generate`` returns ``string`` as a bare ``str`` for a single
    input but a ``list[str]`` for a batch (see ``LMPipeline.dump``), so
    normalize the shape before stripping.
    """
    s = pipeline.generate([{"raw_input": prompt}])["string"]
    return (s if isinstance(s, str) else s[0]).strip()
