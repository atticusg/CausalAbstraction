"""Walking skeleton (WS1, #423): the first end-to-end IIA number on the new stack.

Residual-stream interchange on ONE preset task (weekdays), driven from task
config through every new-stack layer — pipeline load (F3), task token-position
resolution over the ST2 bridge, the identity featurizer path (ST3), a
cross-invoke interchange Edit (ED1/ED2 shape), the single-trace plan lowering
with its barrier (PL1) — and scored with the *existing* metric machinery off
Plan-saved logits: ``task.checker`` through ``make_causal_metric``, fed by the
MX1 scoring adapter (``outputs_from_logits`` → ``as_generation_result``, #408 — this
file's original throwaway glue, shipped). Fixed spans, a single CF group, no
heads, no ragged spans.

The gate artifacts are the captured pins in ``walking_skeleton_pins.json``:
a **tiny-random** section (CPU, fixed-seed value pins — the
``numerical_unit`` tier, since smoke is existence-only per docs/TESTS.md) and
a **golden** section (chat-coherent Qwen3-4B, GPU). Regenerate with
``CAUSALAB_UPDATE_WS_PINS=1 uv run pytest tests/neural/test_walking_skeleton.py``
(the run fails after writing so a regeneration is always reviewed).

This is deliberately NOT a public-wrapper reroute — that is PL3 (#405).
Gates Wave 8: PL3–PL5 open only on this file green.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.io.pipelines import load_pipeline
from causalab.methods.metric import (
    as_label_checker,
    as_generation_result,
    make_causal_metric,
    outputs_from_logits,
    score_intervention_outputs,
)
from causalab.neural.edit import Edit, ReadSource
from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.pipeline import LMPipeline
from causalab.neural.plan import EditOp, Plan, run_plan
from causalab.neural.positions import resolve_positions
from causalab.neural.site import Site
from causalab.tasks.loader import Task, load_task, load_task_counterfactuals

_PINS_PATH = Path(__file__).parent / "walking_skeleton_pins.json"
_UPDATE_ENV = "CAUSALAB_UPDATE_WS_PINS"

_SEED = 0
_TARGET_VARIABLE = "result"

# Verbatim from tests/end_to_end/configs/model/chat-coherent.yaml — the
# wording is load-bearing (a bare "final answer" reads as the number in the
# question; see that config's header).
_CHAT_COHERENT = "Qwen/Qwen3-4B-Instruct-2507"
_CHAT_DIRECTIVE = (
    "Reply with only the final answer word and nothing else. "
    "Do not restate the question."
)
_TINY_RANDOM = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _weekdays_task() -> Task:
    from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

    task = load_task(
        "natural_domains_arithmetic",
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
    )
    task.intervention_variable = _TARGET_VARIABLE
    return task


def _weekdays_dataset(task: Task, n: int, seed: int) -> list[dict[str, Any]]:
    cf_module = load_task_counterfactuals("natural_domains_arithmetic")
    return cf_module.generate_dataset(task.causal_model, n, seed)


def _rows(loaded: dict[str, Any], lo: int, hi: int) -> dict[str, torch.Tensor]:
    return {k: v[lo:hi] for k, v in loaded.items()}


def _run_skeleton(
    pipeline: LMPipeline, task: Task, layers: list[int], dataset: list[dict[str, Any]]
) -> dict[str, Any]:
    """Task config → Site/Edit/Plan interchange → IIA, one number per layer.

    Base and counterfactual traces are tokenized *together* so both plan
    inputs share one padded frame (the single-trace lowering refuses mixed
    frames). Positions come from the task's own ``last_token`` resolver
    through the ST2 bridge, per side of each pair.
    """
    token_position = task.create_token_positions(pipeline)["last_token"]
    base_traces = [ex["input"] for ex in dataset]
    source_traces = [ex["counterfactual_inputs"][0] for ex in dataset]
    n = len(base_traces)

    loaded = pipeline.load(base_traces + source_traces)
    base = _rows(loaded, 0, n)
    source = _rows(loaded, n, 2 * n)
    base_positions = resolve_positions(
        token_position, base_traces, base["attention_mask"], is_original=True
    )
    source_positions = resolve_positions(
        token_position, source_traces, source["attention_mask"], is_original=False
    )

    clean = run_plan(pipeline.model, Plan(inputs={"base": base}, save_logits=("base",)))
    clean_last = clean.logits["base"][:, -1, :].float()
    outputs_by_layer: dict[int, list[dict[str, Any]]] = {}
    delta_by_layer: dict[int, float] = {}
    for layer in layers:
        fsite = FeaturizedSite(Site("block_output", layer))
        plan = Plan(
            inputs={"source": source, "base": base},
            ops=(
                EditOp(
                    "base",
                    Edit(
                        fsite,
                        g=lambda f, f_src: f_src,
                        read_sources=(
                            ReadSource(
                                FeaturizedSite(Site("block_output", layer)),
                                positions=source_positions,
                                input="source",
                            ),
                        ),
                        positions=base_positions,
                    ),
                ),
            ),
            save_logits=("base",),
        )
        patched = run_plan(pipeline.model, plan).logits["base"]
        outputs_by_layer[layer] = outputs_from_logits(pipeline, patched)
        delta_by_layer[layer] = float(
            (patched[:, -1, :].float() - clean_last).abs().mean()
        )

    metric = make_causal_metric(as_label_checker(task.checker), (_TARGET_VARIABLE,))
    scores = score_intervention_outputs(
        results={
            (layer,): as_generation_result(outputs_by_layer[layer]) for layer in layers
        },
        dataset=dataset,
        metric=metric,
        causal_model=task.causal_model,
    )
    # The strings can coincide with clean (tiny-random argmaxes one token for
    # everything), so the patch's numerical footprint is part of the result:
    # a silently-inert interchange must fail the pin even when strings match.
    return {
        "iia": {str(layer): scores[(layer,)] for layer in layers},
        "intervened": {
            str(layer): [o["string"] for o in outputs_by_layer[layer]]
            for layer in layers
        },
        "clean": [
            o["string"] for o in outputs_from_logits(pipeline, clean.logits["base"])
        ],
        "logits_delta": {str(layer): delta_by_layer[layer] for layer in layers},
    }


def _assert_matches_pins(section: str, payload: dict[str, Any]) -> None:
    """Compare against the sidecar, or rewrite it under the update env var.

    A regeneration run always *fails* after writing — the diff must be
    reviewed and the suite re-run clean, mirroring the update-goldens gate.
    """
    payload = json.loads(json.dumps(payload))  # canonicalise like task_pins
    if os.environ.get(_UPDATE_ENV):
        pins = json.loads(_PINS_PATH.read_text()) if _PINS_PATH.is_file() else {}
        pins[section] = payload
        _PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
        pytest.fail(
            f"{_UPDATE_ENV} set: rewrote {section!r} pins at {_PINS_PATH}; "
            "review the diff and re-run without the flag."
        )
    if not _PINS_PATH.is_file():
        raise FileNotFoundError(
            f"No pins sidecar at {_PINS_PATH}. Generate via "
            f"{_UPDATE_ENV}=1 uv run pytest {Path(__file__).name}"
        )
    pinned = json.loads(_PINS_PATH.read_text()).get(section)
    assert pinned is not None, f"pins sidecar has no {section!r} section"
    assert payload["intervened"] == pinned["intervened"]
    assert payload["clean"] == pinned["clean"]
    for layer, value in pinned["iia"].items():
        assert payload["iia"][layer] == pytest.approx(value, abs=1e-9)
    for layer, value in pinned["logits_delta"].items():
        assert value > 0.0, f"pinned logits_delta for layer {layer} is inert"
        assert payload["logits_delta"][layer] == pytest.approx(value, rel=1e-3)


# --------------------------------------------------------------------------- #
#  numerical_unit — tiny-random, CPU: fixed-seed value pins on the protocol    #
# --------------------------------------------------------------------------- #
class TestWalkingSkeletonTinyRandom:
    pytestmark = pytest.mark.numerical_unit

    def test_weekdays_interchange_iia_matches_pins(self) -> None:
        task = _weekdays_task()
        pipeline = load_pipeline(
            _TINY_RANDOM, task, max_new_tokens=1, device="cpu", dtype="float32"
        )
        dataset = _weekdays_dataset(task, n=4, seed=_SEED)
        result = _run_skeleton(pipeline, task, layers=[0, 1], dataset=dataset)
        _assert_matches_pins("tiny-random", result)


# --------------------------------------------------------------------------- #
#  golden — chat-coherent, GPU: the first real IIA number on the new stack     #
# --------------------------------------------------------------------------- #
class TestWalkingSkeletonGolden:
    pytestmark = pytest.mark.golden

    def test_weekdays_interchange_iia_matches_pins(self) -> None:
        task = _weekdays_task()
        pipeline = load_pipeline(
            _CHAT_COHERENT,
            task,
            max_new_tokens=1,
            device="auto",
            dtype="bfloat16",
            use_chat_template=True,
            chat_answer_directive=_CHAT_DIRECTIVE,
        )
        dataset = _weekdays_dataset(task, n=8, seed=_SEED)
        result = _run_skeleton(pipeline, task, layers=[0, 18, 27], dataset=dataset)
        # Semantic floor before the byte pin: late-layer last-token interchange
        # must actually transplant the answer on a model that solves the task.
        best = max(result["iia"].values())
        assert best >= 0.5, f"no layer reaches IIA 0.5: {result['iia']}"
        _assert_matches_pins("golden", result)
