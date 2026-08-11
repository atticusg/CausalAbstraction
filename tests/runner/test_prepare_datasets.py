"""Unit tests for :func:`causalab.runner.helpers.prepare_datasets`.

``prepare_datasets`` is the neural-aware data-prep gate: it wraps the symbolic
``generate_datasets`` and adds a *required* correct-only filtering choice. These
tests pin the guardrail (no silent default, fail-loud on empty/underspecified)
without needing a real model — ``generate_datasets`` and ``filter_dataset`` are
stubbed so only ``prepare_datasets``'s own logic is exercised.
"""

from types import SimpleNamespace

import pytest

from causalab.runner import helpers

pytestmark = pytest.mark.unit


def _fake_task():
    return SimpleNamespace(causal_model=object(), checker=lambda out, exp: True)


def _patch_generate(monkeypatch, train, test):
    monkeypatch.setattr(
        helpers, "generate_datasets", lambda *a, **k: (list(train), list(test))
    )


def test_filter_correct_is_required_keyword_only():
    """No positional/defaulted filtering: callers must state the choice."""
    import inspect

    param = inspect.signature(helpers.prepare_datasets).parameters["filter_correct"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_filter_correct_false_bypasses_filter(monkeypatch):
    train = [{"input": {"raw_output": "A"}}, {"input": {"raw_output": "B"}}]
    test = [{"input": {"raw_output": "C"}}]
    _patch_generate(monkeypatch, train, test)

    import causalab.methods.filter as filt

    called = []
    monkeypatch.setattr(filt, "filter_dataset", lambda *a, **k: called.append(a) or [])

    out_train, out_test = helpers.prepare_datasets(
        _fake_task(), n_train=2, n_test=1, seed=0, filter_correct=False
    )
    assert out_train == train and out_test == test
    assert called == []  # filter must not run when filter_correct=False


def test_filter_correct_true_requires_pipeline_and_metric(monkeypatch):
    _patch_generate(monkeypatch, [{"input": {"raw_output": "A"}}], [])
    with pytest.raises(ValueError, match="requires"):
        helpers.prepare_datasets(
            _fake_task(), n_train=1, n_test=0, seed=0, filter_correct=True
        )
    with pytest.raises(ValueError, match="requires"):
        helpers.prepare_datasets(
            _fake_task(),
            n_train=1,
            n_test=0,
            seed=0,
            filter_correct=True,
            pipeline=object(),
            metric=None,
        )


def test_filter_correct_true_applies_filter_to_train_and_test(monkeypatch):
    train = [{"input": {"raw_output": "A"}}, {"input": {"raw_output": "B"}}]
    test = [{"input": {"raw_output": "C"}}, {"input": {"raw_output": "D"}}]
    _patch_generate(monkeypatch, train, test)

    import causalab.methods.filter as filt

    calls = []

    def spy(dataset, pipeline, causal_model, metric, batch_size=32):
        calls.append((len(dataset), batch_size))
        return dataset[:1]  # keep only the first pair

    monkeypatch.setattr(filt, "filter_dataset", spy)

    task = _fake_task()
    out_train, out_test = helpers.prepare_datasets(
        task,
        n_train=2,
        n_test=2,
        seed=0,
        filter_correct=True,
        pipeline=object(),
        metric=task.checker,
        filter_batch_size=8,
    )
    assert out_train == train[:1]
    assert out_test == test[:1]
    assert calls == [(2, 8), (2, 8)]  # train + test, both at the given batch size


def test_filter_correct_true_empty_train_raises(monkeypatch):
    _patch_generate(monkeypatch, [{"input": {"raw_output": "A"}}], [])
    import causalab.methods.filter as filt

    monkeypatch.setattr(filt, "filter_dataset", lambda *a, **k: [])
    task = _fake_task()
    with pytest.raises(ValueError, match="removed every training example"):
        helpers.prepare_datasets(
            task,
            n_train=1,
            n_test=0,
            seed=0,
            filter_correct=True,
            pipeline=object(),
            metric=task.checker,
        )
