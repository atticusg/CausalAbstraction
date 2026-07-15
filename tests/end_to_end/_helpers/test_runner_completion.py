"""Unit tests for ``assert_runner_completed``.

Default tier (``unit`` per docs/TESTS.md) — declared explicitly via
``pytestmark`` at module scope so the warn-mode collection hook in
``tests/conftest.py`` does not flag these tests as untagged.
"""

from __future__ import annotations

import json

import pytest

from tests.end_to_end._helpers.runner_completion import assert_runner_completed

pytestmark = pytest.mark.unit


def test_passes_when_json_artifact_exists_and_parses(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"loss": 0.5}))
    # cfg is unused in Phase 2; pass None to confirm the helper accepts it.
    assert_runner_completed(None, tmp_path)


def test_fails_when_artifact_missing(tmp_path):
    with pytest.raises(AssertionError, match="expected runner artifact missing"):
        assert_runner_completed(None, tmp_path)


def test_fails_when_json_artifact_is_invalid(tmp_path):
    (tmp_path / "metrics.json").write_text("{not valid json")
    with pytest.raises(AssertionError, match="not valid JSON"):
        assert_runner_completed(None, tmp_path)


def test_passes_for_non_json_artifact_when_non_empty(tmp_path):
    (tmp_path / "results.txt").write_text("ok\n")
    assert_runner_completed(None, tmp_path, expected_artifacts=("results.txt",))


def test_fails_for_non_json_artifact_when_empty(tmp_path):
    (tmp_path / "results.txt").write_text("")
    with pytest.raises(AssertionError, match="exists but is empty"):
        assert_runner_completed(None, tmp_path, expected_artifacts=("results.txt",))


def test_supports_multiple_artifacts(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"a": 1}))
    (tmp_path / "summary.json").write_text(json.dumps({"b": 2}))
    (tmp_path / "log.txt").write_text("done\n")
    assert_runner_completed(
        None,
        tmp_path,
        expected_artifacts=("metrics.json", "summary.json", "log.txt"),
    )
