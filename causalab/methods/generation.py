"""Greedy-decoding helpers over an :class:`LMPipeline`.

Small, analysis-neutral primitives for reading a model's greedy continuation of
a prompt. Lives in ``methods/`` so callers (analyses, skill templates) share one
implementation (docs/CODEBASE.md invariant 4).
"""

from __future__ import annotations

from causalab.neural.pipeline import LMPipeline


def greedy_output(pipeline: LMPipeline, prompt: str) -> str:
    """Return the model's greedy continuation for one prompt, stripped.

    ``pipeline.generate`` returns a
    :class:`~causalab.neural.pipeline.GenerationResult` whose ``strings`` is
    always a list (EU5a, #486), one entry per input.
    """
    return pipeline.generate([{"raw_input": prompt}]).strings[0].strip()
