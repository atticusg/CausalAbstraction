"""Domain-neutral LLM-judge primitives.

Three small surfaces that any analysis can compose:

- :func:`call_llm` — sync chat-completion against OpenRouter or OpenAI with
  retries on rate-limit and transient 5xx.
- :func:`resolve_credentials` — validate a provider's API key is present
  (raises an actionable ``RuntimeError`` if not) without making a call, so
  callers can fail fast before expensive work.
- :func:`extract_json_response` — robust JSON extraction from an LLM string.
- :func:`assert_no_forbidden_substrings` + :class:`ForbiddenSubstringError`
  — runtime guard for "this prompt must not contain X" invariants.

Analyses define their own prompt templates, evidence schemas, and call
orchestration on top. See ``causalab/analyses/characterize_subspace/`` for
the canonical caller.
"""

from causalab.methods.llm_judge.client import (
    Provider,
    call_llm,
    resolve_credentials,
)
from causalab.methods.llm_judge.guards import (
    ForbiddenSubstringError,
    assert_no_forbidden_substrings,
)
from causalab.methods.llm_judge.parsing import extract_json_response

__all__ = [
    "ForbiddenSubstringError",
    "Provider",
    "assert_no_forbidden_substrings",
    "call_llm",
    "extract_json_response",
    "resolve_credentials",
]
