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
    outcome = run_training(executor.doc, executor, request)
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
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
    from causalab.neural.shared.metrics import compute_metric

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
    gate = run_training(executor.doc, executor, request).stages["gate"]
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    assert gate.temperature < 1.0  # the anneal ran
    assert not gate.training  # left in (hard) eval mode


def test_early_stop_returns_the_best_fit_not_the_last(monkeypatch):
    """``early_stop`` selects a fit by its eval score, so the fit it selected
    is the one that must come back.

    Nothing snapshotted the parameters: ``best`` was tracked, the loop broke
    after ``patience`` non-improving evals, and the stages returned were the
    **last** ones — the worst of the tail. The question-mark run saw held-out
    1.000 at every seed, so last and best coincided *at ceiling*; that is luck,
    not a property, and off ceiling nothing in the saved bundle says which
    weights you have.

    The eval score is scripted here rather than engineered out of a random
    model: what is under test is the selection, and a deterministic peak is
    the only way to assert "the peak, not the end" without a flaky fit.
    """
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.protocol.resolve import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=5)
    doc_raw["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/test",
        "metrics": ["ce"],
    }
    doc_raw["train"]["early_stop"] = {"metric": "ce", "patience": 10, "mode": "max"}

    scores = [0.1, 0.9, 0.5, 0.4, 0.3]  # peak at the second eval
    seen: list[dict[str, torch.Tensor]] = []

    def fake_eval(doc, executor, request, split):
        stage = executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(train_module, "_run_eval", fake_eval)

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
    outcome = train_module.run_training(executor.doc, executor, request)

    assert len(seen) == len(scores)  # patience never fires: five evals ran
    peak, final = seen[1], seen[-1]
    key = "parametrizations.weight.original"  # what the optimizer actually steps
    assert not torch.allclose(peak[key], final[key])  # the fit really moved on

    returned = outcome.stages["rot"].state_dict()
    torch.testing.assert_close(returned[key], peak[key], atol=0.0, rtol=0.0)

    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "early_stop.best"
    # the reported score describes the weights that came back, not the last pass
    assert outcome.eval_score.metrics["ce"] == 0.9


def test_without_early_stop_the_last_fit_is_the_one_returned(monkeypatch):
    """Nothing is selecting, so nothing is restored — and the record says so
    rather than leaving a reader to guess."""
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.protocol.resolve import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=3)
    doc_raw["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/test",
        "metrics": ["ce"],
    }

    scores = [0.1, 0.9, 0.2]
    seen: list[dict[str, torch.Tensor]] = []

    def fake_eval(doc, executor, request, split):
        stage = executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(train_module, "_run_eval", fake_eval)

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
    outcome = train_module.run_training(executor.doc, executor, request)

    key = "parametrizations.weight.original"
    torch.testing.assert_close(
        outcome.stages["rot"].state_dict()[key], seen[-1][key], atol=0.0, rtol=0.0
    )
    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "last"
    assert outcome.eval_score.metrics["ce"] == 0.2
    assert outcome.eval_score.passes == 3


def test_an_update_counted_eval_is_refused_rather_than_never_run():
    """This loop only reaches an eval on an epoch boundary, so an ``updates``
    counter would run no eval at all and still save the fit."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.protocol.errors import ProtocolError
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.protocol.resolve import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=1)
    doc_raw["train"]["eval"] = {
        "every": {"updates": 1},
        "split": "weekdays/test",
        "metrics": ["ce"],
    }
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
    with pytest.raises(ProtocolError, match="must count epochs"):
        run_training(executor.doc, executor, request)


def test_validation_accepts_the_train_docs():
    for raw in (das_doc(), dbm_doc()):
        from tests.protocol._docs import in_order

        validate_document(parse_document(in_order(raw)), engine_is_local=True)


def test_a_gate_fit_reports_whether_its_mask_is_a_mask():
    """The DBM finding's non-GPU half.

    As shipped, `configs/protocols/dbm.json` produced **0 of 2048** dimensions
    outside [0.1, 0.9] and still scored **1.000** at layer 38 — because
    `Gate._mask` returns a *hard* `θ > 0` mask in eval mode, so an unseparated
    θ makes the mask a coin flip on gradient noise. Roughly half the dimensions
    swap, which at the readout layer scores 1.000. A meaningless mask and a
    perfect number, with nothing in the saved outputs to tell them apart.

    Asserted on the fit's own report rather than on a value: what is under test
    is that the fact is *recorded*, not that a random model separates θ. The
    retune that makes θ separate needs a GPU run and is not asserted here.
    """
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
    outcome = run_training(executor.doc, executor, request)

    report = outcome.diagnostics["gate"]
    width = int(report["width"])
    assert width == outcome.stages["gate"].theta.numel()
    assert 0.0 <= report["decisive_fraction"] <= 1.0
    assert 0 <= report["hard_mask_size"] <= width
    # the hard mask is what the eval-mode score was computed through, so it has
    # to be reported as a count of *this* gate, not a fraction of some other
    assert report["hard_mask_size"] == float((outcome.stages["gate"].theta > 0).sum())


def test_a_subspace_fit_reports_no_mask_diagnostic():
    """Only a gate has a mask to be indecisive about — the report is per kind,
    not a fixed schema every fit has to fill with zeros."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.protocol.resolve import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        das_doc(seed=0, epochs=1),
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
    assert run_training(executor.doc, executor, request).diagnostics == {}
