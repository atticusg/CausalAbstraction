"""The train loop on tiny-random: DAS and DBM fits (spec §2.11).

Random weights carry no task signal, so these are mechanism tests, not
quality tests: the fit runs, moves exactly the declared params, honors the
seed (same seed → byte-identical fit; different seed → different fit),
anneals the gate temperature, and reduces its own training objective on
the batch it optimizes (a sanity floor even a random model must clear —
the loss is optimized directly)."""

from __future__ import annotations

import pytest
import torch

from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.unit

BASES = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps deeply",
    "every shiny robot dances tonight",
    "some ancient rivers flow backwards",
]
COUNTERFACTUALS = [
    "cold silver mountains echo loudly",
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
]
ANSWERS = [" one", " two", " three", " four"]


def das_doc(*, seed: int = 0, epochs: int = 2) -> dict:
    return {
        "version": "1",
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "sites": {
            "tgt": {"component": "block_output", "layer": 0},
            "lm_head": {"component": "lm_head"},
        },
        "featurizers": {
            "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
        },
        "reads": {
            "v_cf": {
                "site": "tgt",
                "pos": {"index": -1},
                "model": "original",
                "input": "counterfactual",
                "featurizer": "rot",
            },
            "logits": {
                "site": "lm_head",
                "pos": {"index": -1},
                "model": "patched",
                "input": "base",
            },
        },
        "writes": {
            "patch": {
                "site": "tgt",
                "pos": {"index": -1},
                "featurizer": "rot",
                "do": {"swap": "v_cf"},
            }
        },
        "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
        "metrics": {"ce": {"kind": "cross_entropy", "of": "logits", "target": "label"}},
        "train": {
            "objective": [[1.0, "ce"]],
            "params": ["rot"],
            "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0},
            "steps": {"epochs": epochs},
            "batch": {"pairs": 2},
            "seed": seed,
        },
        "save": [
            {
                "value": "ce",
                "model": "patched",
                "input": "base",
                "file_path": "ce.json",
            },
            {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
        ],
    }


class _NoDatasets:
    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return ()


def _fit(doc_raw: dict) -> dict[str, torch.Tensor]:
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.protocol.resolve import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = ExecutionRequest(
        points=(),
        canonical=(),
        digests=(),
        coords=(),
        document_digest="0" * 64,
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    stages = run_training(executor.doc, executor, request)
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in stages.items()
        for slot, param in stage.slot_params().items()
    }


def test_das_fit_moves_only_the_rotation_and_is_seeded():
    first = _fit(das_doc(seed=0))
    again = _fit(das_doc(seed=0))
    other = _fit(das_doc(seed=1))
    assert set(first) == {"rot.weight"}
    torch.testing.assert_close(
        first["rot.weight"], again["rot.weight"], atol=0.0, rtol=0.0
    )
    assert not torch.allclose(first["rot.weight"], other["rot.weight"], atol=1e-6)


def test_das_weight_stays_orthonormal():
    weight = _fit(das_doc(seed=0))["rot.weight"]
    gram = weight.T @ weight
    torch.testing.assert_close(gram, torch.eye(weight.shape[1]), atol=1e-5, rtol=1e-4)


def test_das_fit_reduces_its_own_objective():
    """Optimizing CE on the training rows must reduce CE on those rows —
    compare the document's metric before and after the fit."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.metrics import compute_metric

    bundle = load_model(TINY_LLAMA)

    def mean_ce(fit: bool) -> float:
        doc_raw = das_doc(seed=0, epochs=4)
        executor = executor_for(
            doc_raw,
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
        )
        if fit:
            from causalab.neural.engines.pytorch_hooks.train import run_training
            from causalab.protocol.resolve import ResolutionEnv

            request = ExecutionRequest(
                points=(),
                canonical=(),
                digests=(),
                coords=(),
                document_digest="0" * 64,
                env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
                output_dir=None,  # type: ignore[arg-type]
            )
            run_training(executor.doc, executor, request)
            executor.reset_reads()
        values = compute_metric(
            executor.doc.metrics["ce"],
            executor.read_value("logits"),
            executor.rows_for_metrics(),
            bundle.tokenizer,
        )
        return sum(values) / len(values)

    assert mean_ce(fit=True) < mean_ce(fit=False)


def dbm_doc() -> dict:
    doc = das_doc(seed=0, epochs=3)
    doc["featurizers"] = {"gate": {"kind": "gate"}}
    doc["reads"]["v_cf"]["featurizer"] = "gate"
    doc["writes"]["patch"]["featurizer"] = "gate"
    doc["train"]["params"] = ["gate"]
    doc["train"]["objective"] = [[1.0, "ce"], [0.01, {"l1": "gate"}]]
    doc["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    doc["save"] = [
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"},
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    return doc


def test_dbm_fit_trains_theta_and_anneals_temperature():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.protocol.resolve import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = ExecutionRequest(
        points=(),
        canonical=(),
        digests=(),
        coords=(),
        document_digest="0" * 64,
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    stages = run_training(executor.doc, executor, request)
    gate = stages["gate"]
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    assert gate.temperature < 1.0  # the anneal ran
    assert not gate.training  # left in (hard) eval mode


def test_validation_accepts_the_train_docs():
    for raw in (das_doc(), dbm_doc()):
        from tests.protocol._docs import in_order

        validate_document(parse_document(in_order(raw)), engine_is_local=True)
