"""Tests for the nnterp model-load validation gate (:mod:`causalab.neural.validate`).

Tiers (see docs/TESTS.md; ``neural`` owes ``unit`` + ``property`` direct):

* ``unit`` — report/CLI/wrapper logic and the failure path, with nnterp's
  constructor / subprocess mocked at the boundary (forced-error + external-command
  boundaries, per the mocking policy).
* ``property`` — the real load-gate contract on ``tiny-random`` (CPU): the
  smallest real Llama standardizes and the report carries its structural facts.
* ``golden`` — the gate on the coherent ``chat-coherent`` backbone (GPU), which
  pins the decoupled-``head_dim`` fact F1 surfaced on Qwen3-4B.
"""

from __future__ import annotations

import os
import types

import pytest

from nnterp.rename_utils import RenamingError

from causalab.neural.validate import (
    ModelValidationError,
    ModelValidationReport,
    main,
    run_nnterp_tests,
    validate_model_load,
)
from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME


@pytest.fixture(autouse=True)
def _hermetic_preflight(monkeypatch):
    """Stub the architecture preflight for this module's tests.

    ``validate_model_load`` now runs :func:`assert_architecture_supported`
    first, which reads the checkpoint's config dict from the hub — but these
    unit tests use fake repo ids, and HTTP is a system boundary CI must not
    hit (docs/TESTS.md mocking policy). The preflight class below overrides
    this fixture to mock at the hub boundary instead, so the real logic runs.
    """
    monkeypatch.setattr(
        "causalab.neural.validate.assert_architecture_supported",
        lambda model_name, token=None: None,
    )


class TestReportAndErrorPathUnit:
    pytestmark = pytest.mark.unit

    def test_renaming_error_reported_not_raised(self, monkeypatch):
        def _raise(*_a, **_k):
            raise RenamingError("module tree did not standardize")

        monkeypatch.setattr("causalab.neural.validate.StandardizedTransformer", _raise)
        report = validate_model_load("some/model")
        assert report.ok is False
        assert report.model_name == "some/model"
        assert "RenamingError" in report.error
        assert report.num_layers is None

    def test_value_error_reported_not_raised(self, monkeypatch):
        def _raise(*_a, **_k):
            raise ValueError("No hidden size config key found")

        monkeypatch.setattr("causalab.neural.validate.StandardizedTransformer", _raise)
        report = validate_model_load("some/model")
        assert report.ok is False
        assert "ValueError" in report.error

    def test_unexpected_error_propagates(self, monkeypatch):
        # A non-validation error is a real bug, not a "model failed the gate"
        # verdict — it must propagate rather than be reported as ok=False.
        def _raise(*_a, **_k):
            raise RuntimeError("cuda blew up")

        monkeypatch.setattr("causalab.neural.validate.StandardizedTransformer", _raise)
        with pytest.raises(RuntimeError, match="cuda blew up"):
            validate_model_load("some/model")

    def test_raise_if_failed(self):
        ok = ModelValidationReport(model_name="m", ok=True, num_layers=2)
        ok.raise_if_failed()  # no-op
        bad = ModelValidationReport(model_name="m", ok=False, error="RenamingError: x")
        with pytest.raises(ModelValidationError, match="failed nnterp load-time"):
            bad.raise_if_failed()

    def test_report_reads_gqa_fields_from_text_config(self, monkeypatch):
        """#449 finding 1: `head_dim` / `num_key_value_heads` must come from
        the nested ``text_config`` on models that nest them (Gemma3), not from
        the (absent) top-level keys."""
        from transformers import Gemma3Config

        cfg = Gemma3Config()
        text = cfg.text_config
        fake = types.SimpleNamespace(
            config=cfg,
            num_heads=text.num_attention_heads,
            hidden_size=text.hidden_size,
            num_layers=text.num_hidden_layers,
            vocab_size=text.vocab_size,
        )
        monkeypatch.setattr(
            "causalab.neural.validate.StandardizedTransformer",
            lambda *_a, **_k: fake,
        )
        report = validate_model_load("google/gemma-3-fake")
        assert report.ok is True
        assert report.num_kv_heads == text.num_key_value_heads == 4
        assert report.head_dim == text.head_dim == 256
        # 256 != 2304 // 8 — correctly flagged as decoupled.
        assert report.decoupled_head_dim is True

    def test_summary_row(self):
        ok = ModelValidationReport(
            model_name="m",
            ok=True,
            num_layers=36,
            num_heads=32,
            num_kv_heads=8,
            hidden_size=2560,
            head_dim=128,
            vocab_size=100,
            decoupled_head_dim=True,
        )
        row = ok.summary_row()
        assert row.startswith("[ OK ] m:")
        assert "layers=36" in row and "head_dim=128" in row
        assert "decoupled-head_dim" in row
        assert "scan=" not in row  # unknown scan support stays silent

        bad = ModelValidationReport(model_name="m", ok=False, error="RenamingError: x")
        assert bad.summary_row().startswith("[FAIL] m: RenamingError")

    def test_summary_row_scan_column(self):
        # The CAP5 preflight column: scan-clean vs scan-unsupported must be
        # visible per model (nnterp's allow_dispatch fallback would otherwise
        # hide the difference behind a passing load gate).
        ok = ModelValidationReport(model_name="m", ok=True, scan_supported=True)
        assert "scan=ok" in ok.summary_row()
        no_scan = ModelValidationReport(
            model_name="m", ok=True, scan_supported=False, scan_error="RuntimeError: x"
        )
        assert "scan=UNSUPPORTED" in no_scan.summary_row()


class TestArchitecturePreflightGateUnit:
    """The gate must run the same architecture preflight as ``LMPipeline`` and
    report a too-new ``model_type`` as a failed verdict carrying the
    transformers-version cause — without constructing the model.
    """

    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def _hermetic_preflight(self, monkeypatch):
        # Override the module stub: mock at the hub boundary instead, so the
        # real preflight logic runs against a gemma-4-shaped config dict.
        monkeypatch.setattr(
            "transformers.PretrainedConfig.get_config_dict",
            classmethod(
                lambda cls, name, **kw: (
                    {"model_type": "gemma4", "transformers_version": "5.5.0.dev0"},
                    {},
                )
            ),
        )

    def test_unsupported_architecture_is_a_failed_gate_verdict(self, monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError(
                "StandardizedTransformer must not be constructed when the "
                "preflight rejects the architecture"
            )

        monkeypatch.setattr("causalab.neural.validate.StandardizedTransformer", _boom)
        report = validate_model_load("google/gemma-4-E2B-it")
        assert report.ok is False
        assert "UnsupportedArchitectureError" in report.error
        assert "gemma4" in report.error
        assert "5.5.0.dev0" in report.error


class TestRunNnterpTestsUnit:
    pytestmark = pytest.mark.unit

    def test_builds_command_and_runs_in_neutral_cwd(self, monkeypatch):
        captured = {}

        def fake_run(cmd, cwd, check):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["check"] = check
            return types.SimpleNamespace(returncode=7)

        monkeypatch.setattr("causalab.neural.validate.subprocess.run", fake_run)
        rc = run_nnterp_tests(
            ["gpt2", "tiny"],
            class_names=["LlamaForCausalLM"],
            extra_pytest_args=["-q"],
        )
        assert rc == 7
        cmd = captured["cmd"]
        assert cmd[1:4] == ["-m", "nnterp", "run_tests"]
        assert "--model-names" in cmd and "gpt2" in cmd and "tiny" in cmd
        assert "--class-names" in cmd and "LlamaForCausalLM" in cmd
        assert cmd[-1] == "-q"
        assert captured["check"] is False
        # Neutral scratch cwd — NOT the causalab repo (whose pyproject/conftest
        # would otherwise impose causalab's tier markers on nnterp's own tests).
        assert os.path.basename(captured["cwd"]).startswith("nnterp-run-tests-")
        assert captured["cwd"] != os.getcwd()


class TestCliUnit:
    pytestmark = pytest.mark.unit

    def _ok(self, name):
        return ModelValidationReport(model_name=name, ok=True, num_layers=2)

    def _fail(self, name):
        return ModelValidationReport(
            model_name=name, ok=False, error="RenamingError: x"
        )

    def test_all_ok_exits_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **_k: self._ok(name),
        )
        assert main(["--model-names", "a", "b"]) == 0
        assert "[ OK ] a" in capsys.readouterr().out

    def test_any_failure_exits_one(self, monkeypatch):
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **_k: self._ok(name) if name == "a" else self._fail(name),
        )
        assert main(["--model-names", "a", "b"]) == 1

    def test_run_tests_invoked_only_when_gate_passes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "causalab.neural.validate.run_nnterp_tests",
            lambda names: calls.append(list(names)) or 0,
        )
        # gate passes -> run_tests invoked
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **_k: self._ok(name),
        )
        assert main(["--model-names", "a", "--run-tests"]) == 0
        assert calls == [["a"]]

        # gate fails -> run_tests skipped, exit 1
        calls.clear()
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **_k: self._fail(name),
        )
        assert main(["--model-names", "a", "--run-tests"]) == 1
        assert calls == []

    def test_run_tests_return_code_surfaces(self, monkeypatch):
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **_k: self._ok(name),
        )
        monkeypatch.setattr(
            "causalab.neural.validate.run_nnterp_tests", lambda names: 3
        )
        assert main(["--model-names", "a", "--run-tests"]) == 3

    def test_no_dispatch_flag_forwarded(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "causalab.neural.validate.validate_model_load",
            lambda name, **k: seen.update(k) or self._ok(name),
        )
        main(["--model-names", "a", "--no-dispatch"])
        assert seen == {"allow_dispatch": False}

    def test_model_names_required(self):
        with pytest.raises(SystemExit):
            main([])


class TestTinyRandomProperty:
    """The real load gate on the smallest real Llama (CPU) — the smoke-equivalent."""

    pytestmark = pytest.mark.property

    def test_tiny_random_standardizes(self):
        report = validate_model_load(TINY_RANDOM_MODEL_NAME)
        report.raise_if_failed()
        assert report.ok is True
        assert report.num_layers == 2
        assert report.num_heads == 4
        assert report.hidden_size == 16
        assert report.vocab_size == 32000
        # tiny-random is a plain Llama: head_dim == hidden / n_head (16/4), so
        # not decoupled — the negative case for the golden's decoupled assertion.
        assert report.head_dim == 4
        assert report.num_kv_heads == 4
        assert report.decoupled_head_dim is False
        # The CAP5 scan-preflight column: a plain Llama forward fake-runs, so
        # plans against this checkpoint are preflightable.
        assert report.scan_supported is True
        assert report.scan_error is None


class TestChatCoherentGolden:
    """The load gate on the coherent GPU backbone (Qwen3-4B), the sole GPU tier."""

    pytestmark = pytest.mark.golden

    def test_chat_coherent_standardizes_and_is_decoupled(self):
        report = validate_model_load("Qwen/Qwen3-4B-Instruct-2507")
        report.raise_if_failed()
        assert report.ok is True
        assert report.num_layers == 36
        assert report.num_heads == 32
        assert report.num_kv_heads == 8
        # The F1 finding: Qwen3-4B decouples head_dim (128) from hidden/n_head
        # (2560/32 = 80). nnterp standardizes it cleanly at the model level.
        assert report.head_dim == 128
        assert report.decoupled_head_dim is True
