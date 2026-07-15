"""Domain-neutral runtime guards for prompts sent to an LLM.

Callers that want to enforce a "this prompt must not contain X" invariant
pass the forbidden substrings here. The guard is intentionally narrow — it
does not understand independence, leakage, or any analysis-specific
concept; it just refuses prompts that match.

Analyses build domain semantics on top: e.g. the explore-subspace
analysis treats finding any ``Significance`` field value in a derivation
prompt as a judge-independence violation, and surfaces this error to the
operator with that framing.
"""

from __future__ import annotations


class ForbiddenSubstringError(RuntimeError):
    """Raised when a guarded prompt contains a forbidden substring.

    Carries the offending substring (truncated) in the message so the log
    line is enough to diagnose the leak without re-rendering the prompt.
    """


def assert_no_forbidden_substrings(
    rendered_prompt: str,
    forbidden_substrings: list[str],
) -> None:
    """Raise :class:`ForbiddenSubstringError` if any forbidden substring matches.

    Empty / falsy entries in ``forbidden_substrings`` are skipped so callers
    can pass through unfiltered config without an extra cleanup step.
    """
    for needle in forbidden_substrings:
        if not needle:
            continue
        if needle in rendered_prompt:
            raise ForbiddenSubstringError(
                "Guarded prompt contains a forbidden substring. Substring "
                f"(truncated): {needle[:120]!r}"
            )
