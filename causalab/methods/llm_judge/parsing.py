"""Domain-neutral JSON extraction for LLM responses.

LLMs return JSON inconsistently: sometimes raw, sometimes inside a ```json
fenced block, sometimes embedded in prose. ``extract_json_response`` handles
the three common shapes and returns the first parsed JSON object. Callers
own the schema check on the returned dict.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json_response(raw: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``raw`` and return it as a dict.

    Tries fenced JSON first (most reliable signal), then falls back to the
    bare-braces pattern. Raises ``ValueError`` with the truncated raw input
    on failure so logs are actionable.
    """
    m = _JSON_FENCE_RE.search(raw)
    if m:
        candidate = m.group(1)
    else:
        m = _BARE_JSON_RE.search(raw)
        candidate = m.group(1) if m else raw
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM response. Raw response (truncated): "
            f"{raw[:400]!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM response parsed to {type(parsed).__name__}, expected dict. "
            f"Raw response (truncated): {raw[:400]!r}"
        )
    return parsed
