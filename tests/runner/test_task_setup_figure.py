"""Unit tests for the task-setup figure runner CLI.

Example sampling runs the *symbolic* causal model (``generate_datasets``), so
these are CPU-only — no neural-model load. Constrained tasks (e.g. MCQA, where
the gold color must appear among the choices) need their own counterfactual
generator to produce valid base inputs, so the data-path tests use the shipped
MCQA singleton; the spec-threading loop is tested with ``resolve_task`` /
``examples_for_task`` stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalab.runner import task_setup_figure as tsf

pytestmark = pytest.mark.unit

_MCQA = {"name": "MCQA", "target_variable": "answer_position"}


def test_examples_for_task_returns_prompt_answer_pairs() -> None:
    from causalab.runner.helpers import resolve_task

    task, _ = resolve_task("MCQA", {}, "answer_position", seed=0)
    rows = tsf.examples_for_task(task, n_examples=3, seed=0)
    assert 1 <= len(rows) <= 3
    for row in rows:
        assert set(row) == {"prompt", "answer"}
        assert row["prompt"] and row["answer"]


def test_record_for_task_carries_name_target_examples() -> None:
    from causalab.runner.helpers import resolve_task

    task, _ = resolve_task("MCQA", {}, "answer_position", seed=1)
    rec = tsf.record_for_task(task, n_examples=2, seed=1, description="MCQA task")
    assert rec["name"] == "MCQA"
    assert rec["target_variable"] == "answer_position"
    assert rec["description"] == "MCQA task"
    assert 1 <= len(rec["examples"]) <= 2


def test_build_task_records_threads_each_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub resolve_task + examples_for_task so the loop is exercised without task
    # machinery; assert it threads name/config/target and one record per spec.
    seen: list[tuple] = []

    def fake_resolve_task(name, config, target_variable, seed=None):
        seen.append((name, config, target_variable, seed))
        return SimpleNamespace(name=name, intervention_variable=target_variable), None

    monkeypatch.setattr(tsf, "resolve_task", fake_resolve_task)
    monkeypatch.setattr(
        tsf, "examples_for_task", lambda task, **kw: [{"prompt": "p", "answer": "a"}]
    )
    specs = [
        {"name": "alpha", "target_variable": "shown", "config": {"k": 1}},
        {"name": "beta", "target_variable": "n", "description": "d"},
    ]
    records = tsf.build_task_records(specs, n_examples=2, seed=7)

    assert [r["name"] for r in records] == ["alpha", "beta"]
    assert seen[0] == ("alpha", {"k": 1}, "shown", 7)
    assert seen[1] == ("beta", {}, "n", 7)  # config defaults to {}
    assert records[1]["description"] == "d"


def test_load_task_specs_inline_and_file(tmp_path: Path) -> None:
    payload = '[{"name": "t", "target_variable": "v"}]'
    assert tsf._load_task_specs(payload, None) == [
        {"name": "t", "target_variable": "v"}
    ]
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(payload, encoding="utf-8")
    assert tsf._load_task_specs(None, str(spec_file)) == [
        {"name": "t", "target_variable": "v"}
    ]


@pytest.mark.parametrize("bad", ["{}", "[1, 2]", "[]", "not json"])
def test_load_task_specs_rejects_bad_input(bad: str) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        tsf._load_task_specs(bad, None)


def test_load_task_specs_requires_a_source() -> None:
    with pytest.raises(ValueError, match="--tasks"):
        tsf._load_task_specs(None, None)


def test_cli_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "task_setup.html"
    tsf.main(
        [
            "--tasks",
            json.dumps([{"name": "MCQA", "target_variable": "answer_position"}]),
            "--n",
            "2",
            "--out",
            str(out),
        ]
    )
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "MCQA" in text


def test_build_task_records_real_mcqa_singleton() -> None:
    # End-to-end through resolve_task + generate_datasets on a shipped CPU task.
    records = tsf.build_task_records(
        [{"name": "MCQA", "target_variable": "answer_position"}],
        n_examples=3,
        seed=0,
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["name"] == "MCQA"
    assert rec["target_variable"] == "answer_position"
    assert rec["examples"], "MCQA should yield at least one worked example"
    assert all(r["prompt"] and r["answer"] for r in rec["examples"])
