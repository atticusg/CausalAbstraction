"""Thin sync wrapper around the OpenAI SDK pointed at OpenRouter or OpenAI.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint, so the
same SDK works for both — we just swap ``base_url`` and the auth env var.
This mirrors a standard client-factory pattern but keeps the
surface intentionally narrow: one function, sync, retries on rate limit.

Auth via environment:
- ``OPENROUTER_API_KEY`` when ``provider == "openrouter"`` (default).
- ``OPENAI_API_KEY`` when ``provider == "openai"``.

Raise ``RuntimeError`` if the required env var is unset rather than
deferring to the SDK's generic auth error — the message is more actionable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

logger = logging.getLogger(__name__)


Provider = Literal["openrouter", "openai"]


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MAX_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


def resolve_credentials(provider: Provider) -> tuple[str | None, str]:
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is unset. Export it (or pass "
                "judge.provider=openai and set OPENAI_API_KEY) to run the "
                "subspace-characterization judge."
            )
        return _OPENROUTER_BASE_URL, api_key
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is unset. Export it (or pass "
                "judge.provider=openrouter and set OPENROUTER_API_KEY) to "
                "run the subspace-characterization judge."
            )
        return None, api_key
    raise ValueError(
        f"Unknown provider: {provider!r}. Expected 'openrouter' or 'openai'."
    )


def call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    provider: Provider = "openrouter",
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str:
    """Send a chat-completion request and return the assistant's text.

    Synchronous. Retries up to ``max_retries`` times on
    ``openai.RateLimitError`` and on transient 5xx responses with exponential
    backoff. Other exceptions propagate.

    The caller owns prompt construction; this function is provider-agnostic
    by design so the same code path serves both derive and reconcile calls.
    """
    # Import lazily so the package's import-time cost stays small and unit
    # tests that monkey-patch this function don't pay for the SDK import.
    from openai import OpenAI, APIError, RateLimitError

    base_url, api_key = resolve_credentials(provider)
    client = OpenAI(api_key=api_key, base_url=base_url)

    backoff = _INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError(
                    f"LLM returned empty content for model={model!r}. "
                    "Check max_tokens and the model's availability."
                )
            return content
        except RateLimitError as e:
            last_err = e
            if attempt >= max_retries:
                break
            logger.warning(
                "Rate limited by %s; backing off %.1fs (attempt %d/%d)",
                provider,
                backoff,
                attempt + 1,
                max_retries,
            )
            time.sleep(backoff)
            backoff *= 2
        except APIError as e:
            status = getattr(e, "status_code", None)
            transient = isinstance(status, int) and 500 <= status < 600
            if not transient or attempt >= max_retries:
                raise
            last_err = e
            logger.warning(
                "Transient API error %s from %s; backing off %.1fs (attempt %d/%d)",
                status,
                provider,
                backoff,
                attempt + 1,
                max_retries,
            )
            time.sleep(backoff)
            backoff *= 2

    assert last_err is not None
    raise last_err
