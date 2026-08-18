"""Fixtures for the reference-backend tests.

The oracle side reuses ``tests/neural/activations/hook_oracle.py`` verbatim
— its helpers only touch ``pipeline.hf_model``, so a one-field shim carries
the backend's loaded model into the oracle unchanged (same assertions, same
tolerances, new stack under test)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from causalab.neural.pytorch_hooks.loading import ModelBundle, load_model

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"

BASE_TEXT = "the quick brown fox jumps"
SOURCE_TEXT = "a slow green turtle sleeps deeply"


@dataclasses.dataclass(frozen=True)
class OracleShim:
    """The one attribute the hook-oracle helpers read."""

    hf_model: Any


@pytest.fixture(scope="session", params=["llama", "gpt2"])
def bundle(request: pytest.FixtureRequest) -> ModelBundle:
    key = TINY_LLAMA if request.param == "llama" else TINY_GPT2
    return load_model(key)


@pytest.fixture(scope="session")
def llama_bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture()
def oracle(bundle: ModelBundle) -> OracleShim:
    return OracleShim(hf_model=bundle.model)
