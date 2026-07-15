"""Unit tests for the stdlib task-setup figure renderer.

Pure string/HTML work — no model, no GPU. The renderer turns plain task records
(``{name, target_variable?, description?, examples: [{prompt, answer}]}``) into a
self-contained ``task_setup.html`` page for the artifact
viewer's "Task setup" section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.io.plots.task_setup import (
    render_task_setup_html,
    write_task_setup_html,
)

pytestmark = pytest.mark.unit


def _task(name: str, examples: list[dict[str, str]], **kw) -> dict:
    return {"name": name, "examples": examples, **kw}


def test_renders_prompts_answers_and_target() -> None:
    html = render_task_setup_html(
        [
            _task(
                "comparative_degree",
                [{"prompt": "Rate A vs B.", "answer": "4"}],
                target_variable="degree",
                description="Rate A against B on a 1-5 scale.",
            )
        ]
    )
    # Self-contained document, with the task header, target, description, and the
    # worked example all present.
    assert html.startswith("<!doctype html>")
    assert "comparative_degree" in html
    assert "degree" in html  # target variable
    assert "Rate A against B on a 1-5 scale." in html  # description
    assert "Rate A vs B." in html  # prompt
    assert "4" in html  # answer


def test_multiple_tasks_render_multiple_cards() -> None:
    html = render_task_setup_html(
        [
            _task("task_one", [{"prompt": "p1", "answer": "a1"}]),
            _task("task_two", [{"prompt": "p2", "answer": "a2"}]),
        ]
    )
    assert html.count('class="card"') == 2
    assert "task_one" in html and "task_two" in html
    assert "p1" in html and "p2" in html


def test_html_is_escaped() -> None:
    # A prompt/answer carrying markup must not break out of the page.
    html = render_task_setup_html(
        [
            _task(
                "inj",
                [{"prompt": "<script>alert(1)</script> a & b", "answer": "<b>x</b>"}],
            )
        ]
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&amp;" in html
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_task_without_examples_renders_note_not_crash() -> None:
    html = render_task_setup_html([_task("empty_task", [])])
    assert "empty_task" in html
    assert "No examples available." in html


def test_no_tasks_renders_empty_state() -> None:
    html = render_task_setup_html([])
    assert html.startswith("<!doctype html>")
    assert "No tasks to show." in html


def test_title_is_used_and_escaped() -> None:
    html = render_task_setup_html([], title="Two & <comparison> tasks")
    assert "<title>Two &amp; &lt;comparison&gt; tasks</title>" in html


def test_missing_prompt_or_answer_keys_are_tolerated() -> None:
    # Renderer treats absent prompt/answer as empty rather than raising.
    html = render_task_setup_html([_task("t", [{"prompt": "only prompt"}])])
    assert "only prompt" in html


def test_write_task_setup_html_creates_file_and_parents(tmp_path: Path) -> None:
    out = tmp_path / "plan" / "figures" / "task_setup.html"
    written = write_task_setup_html(
        [_task("t", [{"prompt": "p", "answer": "a"}])], out, title="T"
    )
    assert written == out
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "p" in text and "a" in text
