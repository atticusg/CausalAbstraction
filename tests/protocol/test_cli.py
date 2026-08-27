"""The CLI verbs (spec §9) — validate / explain / digest, plus --set."""

from __future__ import annotations

import json

import pytest

from causalab.cli import main
from causalab.protocol.loader import check_data_columns, load

from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.unit


def _argv(verb: str, name: str, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *extra,
    ]


def test_validate_ok(capsys, artifacts_root):
    assert main(_argv("validate", "02_interchange_im.json", artifacts_root)) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_data_checks_columns(capsys, artifacts_root):
    code = main(_argv("validate", "02_interchange_im.json", artifacts_root, "--data"))
    assert code == 0


def test_validate_data_catches_missing_column(env):
    loaded = load(CORPUS_DIR / "02_interchange_im.json", env)
    raw = json.loads(json.dumps(dict(loaded.raw)))
    raw["metrics"]["logit_diff"]["a"] = "not_a_column"
    reloaded = load(raw, env)
    with pytest.raises(Exception) as err:
        check_data_columns(reloaded, env)
    assert "not_a_column" in str(err.value)


def test_digest_prints_the_document_digest(capsys, env, artifacts_root):
    assert main(_argv("digest", "04_das_im.json", artifacts_root)) == 0
    printed = capsys.readouterr().out.strip()
    assert printed == load(CORPUS_DIR / "04_das_im.json", env).document_digest


def test_explain_reports_plan(capsys, artifacts_root):
    assert main(_argv("explain", "03_path_patching_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "forwards  4 per point" in out
    assert "paired_forward" in out


def test_explain_sweep_reports_point_count(capsys, artifacts_root):
    assert (
        main(_argv("explain", "07_weekdays_locate_scan_im.json", artifacts_root)) == 0
    )
    out = capsys.readouterr().out
    assert "points    64" in out


def test_set_override_changes_digest(capsys, env, artifacts_root):
    assert (
        main(
            _argv(
                "digest",
                "02_interchange_im.json",
                artifacts_root,
                "--set",
                "sites.target.layer=5",
            )
        )
        == 0
    )
    overridden = capsys.readouterr().out.strip()
    assert (
        overridden != load(CORPUS_DIR / "02_interchange_im.json", env).document_digest
    )


def test_refusal_exits_nonzero(capsys, artifacts_root):
    code = main(
        _argv(
            "validate",
            "02_interchange_im.json",
            artifacts_root,
            "--set",
            "sites.target.layer=99",
        )
    )
    assert code == 1
    assert "refused" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# run-verb execution flags: --device / --dtype / --points
# --------------------------------------------------------------------------- #


class _CapturingBackend:
    """Stands in for the reference backend: records construction kwargs and
    the ExecutionRequest, executes nothing."""

    last: "_CapturingBackend | None" = None

    name = "capture"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    is_local = True

    def __init__(self, *, device: str = "cpu", dtype: str = "fp32") -> None:
        self.device = device
        self.dtype = dtype
        self.request = None
        type(self).last = self

    def execute(self, request):
        from causalab.protocol.backend import RunResult

        self.request = request
        return RunResult(files={})


@pytest.fixture
def capturing_backend(monkeypatch):
    """Swap the lazily-imported reference backend module for the stub."""
    import sys as _sys
    import types

    stub = types.ModuleType("causalab.neural.pytorch_hooks")
    stub.PytorchHooksBackend = _CapturingBackend
    monkeypatch.setitem(_sys.modules, "causalab.neural.pytorch_hooks", stub)
    _CapturingBackend.last = None
    return _CapturingBackend


def _run_argv(name: str, artifacts_root, out, *extra: str) -> list[str]:
    return _argv("run", name, artifacts_root, "--out", str(out), *extra)


def test_run_threads_device_and_dtype_into_the_backend(
    capturing_backend, artifacts_root, tmp_path
):
    code = main(
        _run_argv(
            "02_interchange_im.json",
            artifacts_root,
            tmp_path,
            "--device",
            "cuda:1",
            "--dtype",
            "bf16",
        )
    )
    assert code == 0
    assert capturing_backend.last.device == "cuda:1"
    assert capturing_backend.last.dtype == "bf16"


def test_run_defaults_stay_cpu_fp32(capturing_backend, artifacts_root, tmp_path):
    assert main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path)) == 0
    assert capturing_backend.last.device == "cpu"
    assert capturing_backend.last.dtype == "fp32"


def test_points_selects_a_shard_without_moving_the_campaign_digest(
    capturing_backend, env, artifacts_root, tmp_path
):
    loaded = load(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env)
    code = main(
        _run_argv(
            "07_weekdays_locate_scan_im.json",
            artifacts_root,
            tmp_path,
            "--points",
            "3:7",
        )
    )
    assert code == 0
    request = capturing_backend.last.request
    assert len(request.points) == 4
    assert request.digests == tuple(loaded.point_digests[3:7])
    assert request.coords == tuple(p.coords for p in loaded.expansion.points[3:7])
    assert request.document_digest == loaded.document_digest


@pytest.mark.parametrize("spec", ["7", "3:3", "60:70", "-1:4", "a:b"])
def test_points_refuses_malformed_and_out_of_range(
    capturing_backend, artifacts_root, tmp_path, capsys, spec
):
    # the = form keeps argparse from reading a leading "-" as a flag
    code = main(
        _run_argv(
            "07_weekdays_locate_scan_im.json",
            artifacts_root,
            tmp_path,
            f"--points={spec}",
        )
    )
    assert code == 1
    assert "refused" in capsys.readouterr().err


def test_points_refused_on_workflow_documents(
    capturing_backend, artifacts_root, tmp_path, capsys
):
    doc = tmp_path / "wf.json"
    doc.write_text(json.dumps({"version": "1", "steps": {}}))
    code = main(
        [
            "run",
            str(doc),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts_root),
            "--out",
            str(tmp_path / "out"),
            "--points",
            "0:1",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "refused" in err and "workflow" in err


# --------------------------------------------------------------------------- #
#  column positions and match modes are checked like any other reference      #
# --------------------------------------------------------------------------- #


def test_validate_data_flags_a_missing_position_column(env):
    """A ``{"column": …}`` position is an explicit reference, so
    ``validate --data`` catches a typo at load instead of the backend hitting
    it mid-run (§2.3)."""
    loaded = load(CORPUS_DIR / "10_task_table_iia_im.json", env)
    raw = json.loads(json.dumps(dict(loaded.raw)))
    raw["positions"]["subject"] = {"column": "not_a_column"}
    with pytest.raises(Exception) as err:
        check_data_columns(load(raw, env), env)
    assert "not_a_column" in str(err.value)


def test_validate_data_accepts_the_generated_tables_columns(env):
    """The positive half: every reference in the task-table document resolves
    against the built table, including the answer-form group column."""
    loaded = load(CORPUS_DIR / "10_task_table_iia_im.json", env)
    refs = check_data_columns(loaded, env)
    assert "label_forms" in refs  # the metric's expected group
    assert "entity" in refs  # the column position


def test_validate_data_flags_a_missing_relative_to_column(env):
    loaded = load(CORPUS_DIR / "10_task_table_iia_im.json", env)
    raw = json.loads(json.dumps(dict(loaded.raw)))
    raw["positions"]["subject"] = {
        "index": 1,
        "relative_to": {"column": "not_a_column"},
    }
    with pytest.raises(Exception) as err:
        check_data_columns(load(raw, env), env)
    assert "not_a_column" in str(err.value)


def test_explain_reports_the_decode_and_what_it_obliges(capsys, artifacts_root):
    """A generate document's cost is legible before it runs: how far it
    decodes, and which reads oblige a vocabulary tensor."""
    assert main(_argv("explain", "11_probe_generate_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "generate" in out
    assert "decode 8 tokens (greedy)" in out
    assert "tail at lm_head: distribution per addressed position" in out
