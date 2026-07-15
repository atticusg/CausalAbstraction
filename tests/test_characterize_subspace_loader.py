"""Regression tests for the raw-HF loader in ``characterize_subspace``.

Guards issue #222: the shipped ``llama31_8b`` config defaults ``device: auto``,
and ``_load_hf_model`` must resolve that string to a concrete device *before*
constructing ``torch.device(...)`` — ``torch.device("auto")`` raises
``RuntimeError``. The fix routes the config device through ``resolve_device``
(``main.py:108``); these tests pin that contract at the *loader* site so a revert
to the raw ``torch.device(cfg.model.device)`` path fails CI.

The end-to-end smoke test stubs ``_load_hf_model`` wholesale, so the resolution
inside it is otherwise never exercised. Everything here runs on tiny stubs with
no model download and no real device — ``torch.device("cuda")`` constructs fine
without a GPU because the stubbed ``.to`` never touches hardware.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.analyses.characterize_subspace.main import _load_hf_model

pytestmark = pytest.mark.unit


class _RecordingModel:
    """Stand-in HF model whose ``.to`` records the device it was handed."""

    def __init__(self) -> None:
        self.to_calls: list[Any] = []

    def to(self, target: Any) -> "_RecordingModel":
        self.to_calls.append(target)
        return self

    def eval(self) -> "_RecordingModel":
        return self


class _StubTokenizer:
    # Exercises the pad_token back-fill branch in _load_hf_model.
    pad_token_id = None
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token: Any = None


@pytest.fixture
def stub_transformers(monkeypatch: pytest.MonkeyPatch) -> type:
    """Replace the function-local ``transformers`` loaders with tiny stubs.

    ``_load_hf_model`` does ``from transformers import AutoModelForCausalLM,
    AutoTokenizer`` at call time, so patching the module attributes is enough.
    Returns the model-stub class whose ``.last`` holds the loaded instance.
    """

    class _StubAutoModel:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _RecordingModel:
            return _RecordingModel()

    class _StubAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _StubTokenizer:
            return _StubTokenizer()

    monkeypatch.setattr("transformers.AutoModelForCausalLM", _StubAutoModel)
    monkeypatch.setattr("transformers.AutoTokenizer", _StubAutoTokenizer)
    return _StubAutoModel


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, False, "cuda"),  # default GPU box: criterion #1 — auto loads on GPU
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_load_hf_model_resolves_auto_to_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
    stub_transformers: type,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    """``device="auto"`` resolves to a concrete device and reaches ``.to`` as a
    valid ``torch.device`` — never the literal ``"auto"`` (which would raise)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    model, _tokenizer, resolved_device = _load_hf_model(
        "dummy-model", device="auto", dtype=None
    )

    assert resolved_device == expected
    assert resolved_device != "auto"

    # The model was moved to a real torch.device of the resolved type — i.e. the
    # loader resolved "auto" before constructing torch.device(...). A revert to
    # torch.device("auto") would have raised inside _load_hf_model above.
    assert len(model.to_calls) == 1
    moved_to = model.to_calls[0]
    assert isinstance(moved_to, torch.device)
    assert moved_to.type == expected


def test_load_hf_model_passes_explicit_device_through(
    monkeypatch: pytest.MonkeyPatch,
    stub_transformers: type,
) -> None:
    """An explicit ``device`` is honoured verbatim regardless of availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model, _tokenizer, resolved_device = _load_hf_model(
        "dummy-model", device="cpu", dtype=None
    )

    assert resolved_device == "cpu"
    assert model.to_calls == [torch.device("cpu")]
