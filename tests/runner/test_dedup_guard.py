"""Tests for the dataset-dedup incoherence guard in ``generate_datasets`` (#180).

``_deduplicate_by_input`` collapses examples whose base ``raw_input`` strings are
identical. When a prompt template ignores a variable that the dataset is
sweeping, distinct causal-model inputs render to the same prompt but carry
different gold labels; dedup would silently keep one label and discard the rest.

``_deduplicate_examples`` guards against this:

- **Pathological** collapse (same prompt -> differing gold) raises ``ValueError``
  unconditionally — the dataset is unsolvable by construction.
- **Benign** collapse (same prompt, identical gold) is information-preserving;
  a large drop only logs a warning once it exceeds ``dedup_warn_threshold``.

These guard both the dedup paths and the ``enumerate_all`` path of
``generate_datasets``.
"""

from __future__ import annotations

import logging

import pytest

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.runner.helpers import (
    _deduplicate_examples,  # pyright: ignore[reportPrivateUsage]
    generate_datasets,
)
from causalab.tasks.loader import Task

pytestmark = pytest.mark.unit

_HELPERS_LOGGER = "causalab.runner.helpers"


def _dedup_checker(neural_output: dict, causal_output: str) -> bool:
    """Trivial checker so the task satisfies the required-checker contract (#167).

    The dedup guard never scores outputs, so the body is irrelevant — the task
    just needs *a* checker to construct (``Task.checker`` has no default)."""
    return causal_output in str(neural_output.get("string", ""))


def _make_task(
    *,
    raw_input_parents: list[str],
    raw_input_compute,
    raw_output_parents: list[str],
    raw_output_compute,
    target: str | None = None,
    name: str = "dedup_test",
) -> Task:
    """Build a 2-input task (``shown`` in {a,b}, ``extra`` in 0..4).

    Callers wire ``raw_input``/``raw_output`` to depend on whichever inputs the
    scenario needs, so the same fixture covers conflicting, unique, and benign
    collapses.
    """
    values = {
        "shown": ["a", "b"],
        "extra": [0, 1, 2, 3, 4],
        "raw_input": None,
        "raw_output": None,
    }
    mechanisms = {
        "shown": input_var(["a", "b"]),
        "extra": input_var([0, 1, 2, 3, 4]),
        "raw_input": Mechanism(parents=raw_input_parents, compute=raw_input_compute),
        "raw_output": Mechanism(parents=raw_output_parents, compute=raw_output_compute),
    }
    model = CausalModel(mechanisms, values, id=name)
    return Task(
        name=name,
        causal_model=model,
        intervention_variable=target,
        checker=_dedup_checker,
    )


def _enumerate_examples(task: Task) -> list:
    """All 10 unique (shown, extra) inputs as ``generate_datasets``-shaped rows."""
    return [
        {"input": t, "counterfactual_inputs": []}
        for t in task.causal_model.enumerate_inputs()
    ]


# --- The prompt ignores a swept variable that drives the gold label ---------


def _conflicting_task() -> Task:
    """raw_input depends only on ``shown``; raw_output depends only on ``extra``.

    So ``"word: a"`` maps to five different ``"tier: N"`` golds -> incoherent.
    """
    return _make_task(
        raw_input_parents=["shown"],
        raw_input_compute=lambda t: f"word: {t['shown']}",
        raw_output_parents=["extra"],
        raw_output_compute=lambda t: f"tier: {t['extra']}",
    )


def test_conflicting_collapse_raises():
    task = _conflicting_task()
    examples = _enumerate_examples(task)
    # dedup would drop 8/10 (80%) and merge conflicting gold tiers.
    with pytest.raises(ValueError) as exc:
        _deduplicate_examples(examples, label="train", task=task, warn_threshold=0.2)
    msg = str(exc.value)
    assert "Incoherent train dataset" in msg
    assert "word: a" in msg  # the offending prompt
    assert "extra" in msg  # swept-but-absent-from-prompt variable
    assert "multiple distinct gold labels" in msg


def test_generate_datasets_resample_raises():
    """The guard fires through ``generate_datasets`` on the resample path."""
    task = _conflicting_task()
    with pytest.raises(ValueError, match="Incoherent"):
        generate_datasets(task, n_train=20, n_test=0, seed=0, resample_variable="extra")


def test_generate_datasets_enumerate_all_raises():
    """The guard fires on the ``enumerate_all`` path (which never dedups)."""
    task = _conflicting_task()
    # n_unique_inputs = 2 * 5 = 10 <= n_train -> enumerate branch.
    with pytest.raises(ValueError, match="Incoherent enumerated"):
        generate_datasets(task, n_train=100, n_test=0, seed=0, enumerate_all=True)


# --- Prompt distinguishes every row: no dedup, no warning -------------------


def test_unique_prompts_no_dedup_no_warning(caplog):
    task = _make_task(
        raw_input_parents=["shown", "extra"],
        raw_input_compute=lambda t: f"{t['shown']}-{t['extra']}",
        raw_output_parents=["shown"],
        raw_output_compute=lambda t: f"out: {t['shown']}",
    )
    examples = _enumerate_examples(task)
    with caplog.at_level(logging.INFO, logger=_HELPERS_LOGGER):
        result = _deduplicate_examples(
            examples, label="train", task=task, warn_threshold=0.2
        )
    assert len(result) == len(examples) == 10
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not [r for r in caplog.records if "Deduplicated" in r.getMessage()]


# --- Swept variable is inert: collapse is benign (warn only above threshold) -


def test_benign_collapse_warns_and_threshold_configurable(caplog):
    # ``extra`` varies but feeds neither the prompt nor the gold label, so the
    # 10 rows collapse to 2 unique prompts whose golds agree -> no error.
    task = _make_task(
        raw_input_parents=["shown"],
        raw_input_compute=lambda t: f"word: {t['shown']}",
        raw_output_parents=["shown"],
        raw_output_compute=lambda t: f"out: {t['shown']}",
    )
    examples = _enumerate_examples(task)

    # 80% drop exceeds the 20% threshold -> one warning naming the swept var.
    with caplog.at_level(logging.WARNING, logger=_HELPERS_LOGGER):
        result = _deduplicate_examples(
            examples, label="train", task=task, warn_threshold=0.2
        )
    assert len(result) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "extra" in warnings[0].getMessage()

    # Raising the threshold above the drop suppresses the warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_HELPERS_LOGGER):
        _deduplicate_examples(examples, label="train", task=task, warn_threshold=0.95)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_empty_examples_returns_empty():
    task = _conflicting_task()
    assert _deduplicate_examples([], label="train", task=task, warn_threshold=0.2) == []
