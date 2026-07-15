"""Tests for the domain-neutral primitives in ``causalab.methods.llm_judge``.

The analysis-specific judge tests live in
``tests/test_characterize_subspace_judge.py``; this file covers only what
``methods/llm_judge`` itself offers: JSON extraction, the substring guard,
and the env-var error paths in the client.
"""

from __future__ import annotations

import pytest

from causalab.methods.llm_judge import (
    ForbiddenSubstringError,
    assert_no_forbidden_substrings,
    extract_json_response,
)
from causalab.methods.llm_judge.client import resolve_credentials

pytestmark = pytest.mark.unit


class TestExtractJsonResponse:
    def test_parses_bare_json(self) -> None:
        out = extract_json_response('{"a": 1, "b": "two"}')
        assert out == {"a": 1, "b": "two"}

    def test_parses_fenced_json(self) -> None:
        raw = 'Sure! Here you go:\n\n```json\n{"a": 1}\n```\n\nLet me know if...'
        assert extract_json_response(raw) == {"a": 1}

    def test_parses_unlabeled_fenced_json(self) -> None:
        raw = 'preamble\n```\n{"x": [1, 2, 3]}\n```\ntrailing'
        assert extract_json_response(raw) == {"x": [1, 2, 3]}

    def test_parses_json_embedded_in_prose(self) -> None:
        raw = 'My verdict is {"verdict": "confirmed", "rationale": "ok"} — done.'
        assert extract_json_response(raw) == {"verdict": "confirmed", "rationale": "ok"}

    def test_raises_on_unparseable_input(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            extract_json_response("no json here at all")

    def test_raises_when_parsed_value_is_not_a_dict(self) -> None:
        with pytest.raises(ValueError, match="expected dict"):
            extract_json_response("[1, 2, 3]")


class TestAssertNoForbiddenSubstrings:
    def test_passes_with_no_match(self) -> None:
        assert_no_forbidden_substrings("This prompt is clean.", ["secret"])

    def test_raises_on_match(self) -> None:
        with pytest.raises(ForbiddenSubstringError, match="forbidden substring"):
            assert_no_forbidden_substrings(
                "The user said: secret password!", ["secret password"]
            )

    def test_skips_empty_needles(self) -> None:
        # Empty strings would trivially match every prompt; the guard must
        # ignore them so callers can pass through unfiltered config.
        assert_no_forbidden_substrings("anything goes here", ["", None])  # type: ignore[list-item]

    def test_includes_truncated_needle_in_error(self) -> None:
        long_needle = "x" * 500
        prompt = f"contains {long_needle} in it"
        with pytest.raises(ForbiddenSubstringError) as exc_info:
            assert_no_forbidden_substrings(prompt, [long_needle])
        # Error message must include a (truncated) preview, not the full needle.
        msg = str(exc_info.value)
        assert "xxxxx" in msg
        assert len(msg) < len(long_needle) + 200


class TestResolveCredentials:
    def test_openrouter_requires_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is unset"):
            resolve_credentials("openrouter")

    def test_openai_requires_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is unset"):
            resolve_credentials("openai")

    def test_openrouter_returns_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        base_url, key = resolve_credentials("openrouter")
        assert base_url == "https://openrouter.ai/api/v1"
        assert key == "sk-test"

    def test_openai_returns_no_custom_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        base_url, key = resolve_credentials("openai")
        assert base_url is None
        assert key == "sk-test"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_credentials("anthropic")  # type: ignore[arg-type]
