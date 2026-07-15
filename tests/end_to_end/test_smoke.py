"""Smoke-tier end-to-end tests for every smoke runner config.

Drives each ``tests/end_to_end/configs/smoke/*.yaml`` through the real
runner pipeline (``causalab.runner.run_exp._run_steps``) against the tiny
HF Llama stub. Asserts via :func:`assert_runner_completed` that each
config's declared ``expected_artifacts`` land on disk.

This is the "didn't crash + artifacts exist" tier. Numerical assertions
live in ``test_goldens.py`` — the golden tier, which runs the coherent
model on GPU. Smoke is the cheap, broad CPU canary: it must run on CPU and
stay under the ~5min total wall budget; the tiny model keeps every runner
well under 30s.

Every smoke config is **self-describing**: it pins ``/model: tiny-random``,
bakes its own task scale (``n_train`` / ``batch_size``), and declares the
artifacts it must produce under ``expected_artifacts:`` (paths relative to
``experiment_root``). The test only fixes ``experiment_root`` so each
parametrized run is hermetic — there is no central expected-artifact map
and no runtime model/scale override to drift against the configs.

The smoke set is deliberately broad-but-shallow: every task's baseline
plus every analysis chain tiny-random can express, each on its own
config. Analyses that are mathematically undefined on random-init weights
(``output_manifold``, ``pullback`` — uniform logits collapse the per-class
belief centroids in Hellinger PCA space) cannot be smoke-tested; they are
golden-only on ``chat-coherent``. See docs/TESTS.md "Smoke vs golden split".

Why not mock the model? docs/TESTS.md is explicit: no mocks of internal
numerical code. The tiny model is a real ``LlamaForCausalLM`` with random
weights, so the runner pipeline exercises the real forward, generation,
and tokenization paths — drift between test and production gets caught the
day the contract changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causalab.runner.run_exp import _run_steps
from tests._helpers.tiny import tiny_random_model
from tests.end_to_end._helpers.enumeration import (
    enumerate_e2e_configs,
    load_e2e_runner_config,
)
from tests.end_to_end._helpers.runner_completion import assert_runner_completed

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="session")
def _prewarm_tiny_model():
    """Pre-fetches the HF tiny stub from the network before smoke tests run.

    What this actually does is populate the HuggingFace on-disk cache
    (``~/.cache/huggingface``) so the first parametrized test doesn't
    pay a cold network hit. The model object loaded here is **not**
    reused by the runner — ``LMPipeline`` calls
    ``AutoModelForCausalLM.from_pretrained(TINY_RANDOM_MODEL_NAME)``
    independently and gets its own (also-on-disk-cached) instance.

    Pre-warming here also surfaces any HF-cache / network problem as a
    fixture failure rather than a confusing first-test regression.
    """
    tiny_random_model()


@pytest.mark.parametrize("config_name", enumerate_e2e_configs("smoke"))
def test_runner_smoke(config_name: str, tmp_path: Path, _prewarm_tiny_model) -> None:
    """Compose a smoke runner, run it on the tiny model, assert it landed.

    Each ``tests/end_to_end/configs/smoke/<name>.yaml`` pins ``tiny-random``
    and its own task scale, and declares the artifacts it must produce via
    ``expected_artifacts:`` (paths relative to ``experiment_root``). The
    test fixes only ``experiment_root`` so each parametrized run is
    hermetic — no cross-config artifact pollution.
    """
    cfg = load_e2e_runner_config(config_name)
    OmegaConf.update(cfg, "experiment_root", str(tmp_path), force_add=True)

    # Run the real runner entry point. This calls into every analysis
    # step's ``main(cfg)`` — the same code path production uses.
    _run_steps(cfg)

    # ``expected_artifacts`` is declared in the config itself; a config that
    # forgets it fails loudly here rather than silently asserting nothing.
    expected_artifacts = tuple(cfg.expected_artifacts)
    assert_runner_completed(cfg, tmp_path, expected_artifacts=expected_artifacts)
