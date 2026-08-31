"""The CLI verbs (spec §9) — validate / explain / digest, plus --set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.protocol.engine import Engine
from causalab.protocol.schema import COMPONENTS
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


class _CapturingEngine(Engine):
    """Stands in for the reference engine: records construction kwargs and
    the ExecutionRequest, executes nothing."""

    last: "_CapturingEngine | None" = None

    name = "capture"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self.request = None
        type(self).last = self

    def execute(self, request):
        from causalab.protocol.engine import RunResult

        self.request = request
        return RunResult(files={})


@pytest.fixture
def capturing_engine(monkeypatch):
    """Swap the lazily-imported reference engine module for the stub."""
    import sys as _sys
    import types

    stub = types.ModuleType("causalab.neural.engines.pytorch_hooks")
    stub.PytorchHooksEngine = _CapturingEngine
    monkeypatch.setitem(_sys.modules, "causalab.neural.engines.pytorch_hooks", stub)
    _CapturingEngine.last = None
    return _CapturingEngine


def _run_argv(name: str, artifacts_root, out, *extra: str) -> list[str]:
    return _argv("run", name, artifacts_root, "--out", str(out), *extra)


def test_device_goes_to_the_engine_and_dtype_goes_to_the_document(
    capturing_engine, artifacts_root, tmp_path
):
    """§8: placement is the engine's, precision is the document's. ``--dtype``
    is shorthand for ``--set model.dtype``, so an overridden run's digest is
    the overridden document's — the record cannot disagree with the numbers."""
    unoverridden = main(_argv("digest", "02_interchange_im.json", artifacts_root))
    assert unoverridden == 0
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
    assert capturing_engine.last.device == "cuda:1"
    assert not hasattr(capturing_engine.last, "dtype")
    request = capturing_engine.last.request
    assert request.canonical[0]["model"]["dtype"] == "bf16"


def test_engine_auto_tolerates_a_missing_optional_engine(
    capturing_engine, artifacts_root, tmp_path
):
    """--engine auto is every *installed* engine: the nnsight extra being
    absent must not break runs that never needed it."""
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "auto")
    )
    assert code == 0
    assert capturing_engine.last is not None


class _AbsentModule:
    """A meta-path finder that makes one module unimportable, so the
    not-installed path stays testable in an env that has the extra."""

    def __init__(self, name: str) -> None:
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r} (simulated)")
        return None


def test_engine_nnsight_refuses_by_name_when_not_installed(
    capturing_engine, artifacts_root, tmp_path, capsys, monkeypatch
):
    """Naming an engine that is not installed is an error that says which
    extra provides it — unlike auto, which quietly narrows to what exists."""
    import sys as _sys

    target = "causalab.neural.engines.nnsight_tracing"
    for mod in [m for m in list(_sys.modules) if m.startswith(target)]:
        monkeypatch.delitem(_sys.modules, mod)
    monkeypatch.setattr(_sys, "meta_path", [_AbsentModule(target), *_sys.meta_path])
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "nnsight")
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "nnsight" in err and "extra" in err


def test_engine_nnsight_selects_the_nnsight_engine(
    capturing_engine, artifacts_root, tmp_path, monkeypatch
):
    """--engine nnsight builds the nnsight engine (stubbed here — the real
    one's answers are pinned by its parity suite)."""
    import sys as _sys
    import types

    class _CapturingNnsight(_CapturingEngine):
        name = "nnsight"

    stub = types.ModuleType("causalab.neural.engines.nnsight_tracing")
    stub.NnsightEngine = _CapturingNnsight
    monkeypatch.setitem(_sys.modules, "causalab.neural.engines.nnsight_tracing", stub)
    _CapturingNnsight.last = None
    code = main(
        _run_argv("01_harvest_im.json", artifacts_root, tmp_path, "--engine", "nnsight")
    )
    assert code == 0
    assert _CapturingNnsight.last is not None
    assert _CapturingNnsight.last.name == "nnsight"


def test_explain_engine_prints_the_engine_that_would_run(
    capturing_engine, artifacts_root, capsys
):
    """`explain` printed `requires` and stopped, so routing could not be
    pre-flighted — which is exactly what is not obvious on a model where one
    family of components is hooks-only and another is nnsight-only."""
    code = main(
        _argv("explain", "02_interchange_im.json", artifacts_root, "--engine", "auto")
    )
    assert code == 0
    assert "engine    capture" in capsys.readouterr().out


def test_explain_engine_prints_the_refusal_rather_than_raising(
    capturing_engine, artifacts_root, capsys, monkeypatch
):
    """A document nothing serves is the *more* useful answer of the two, so it
    is printed beside the plan instead of aborting the explanation."""
    monkeypatch.setattr(
        capturing_engine,
        "capabilities",
        frozenset(),  # serves nothing
    )
    # `pytorch_hooks`, not `auto`: auto also builds the real nnsight engine,
    # which would serve this document and route right past the stub
    code = main(
        _argv(
            "explain",
            "02_interchange_im.json",
            artifacts_root,
            "--engine",
            "pytorch_hooks",
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "engine    refused" in out
    assert "paired_forward" in out  # names what is missing


def test_explain_without_the_flag_loads_no_engine(artifacts_root, capsys):
    """Opt-in: engines are heavy, and the pure verbs stay torch-free. No
    `capturing_engine` fixture here — a load would import the real one."""
    assert main(_argv("explain", "02_interchange_im.json", artifacts_root)) == 0
    assert "engine" not in capsys.readouterr().out


def test_run_defaults_stay_cpu_fp32(capturing_engine, artifacts_root, tmp_path):
    assert main(_run_argv("02_interchange_im.json", artifacts_root, tmp_path)) == 0
    assert capturing_engine.last.device == "cpu"
    assert capturing_engine.last.request.canonical[0]["model"]["dtype"] == "fp32"


def test_dtype_and_set_may_not_contradict(artifacts_root, tmp_path):
    with pytest.raises(SystemExit) as err:
        main(
            _run_argv(
                "02_interchange_im.json",
                artifacts_root,
                tmp_path,
                "--dtype",
                "bf16",
                "--set",
                "model.dtype=fp16",
            )
        )
    assert "contradicts" in str(err.value)


def test_points_selects_a_shard_without_moving_the_campaign_digest(
    capturing_engine, env, artifacts_root, tmp_path
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
    request = capturing_engine.last.request
    assert len(request.points) == 4
    assert request.digests == tuple(loaded.point_digests[3:7])
    assert request.coords == tuple(p.coords for p in loaded.expansion.points[3:7])
    assert request.document_digest == loaded.document_digest


@pytest.mark.parametrize("spec", ["7", "3:3", "60:70", "-1:4", "a:b"])
def test_points_refuses_malformed_and_out_of_range(
    capturing_engine, artifacts_root, tmp_path, capsys, spec
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
    capturing_engine, artifacts_root, tmp_path, capsys
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
    ``validate --data`` catches a typo at load instead of the engine hitting
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


# --------------------------------------------------------------------------- #
# methods, applications, and the run record (§1.1, §9)
# --------------------------------------------------------------------------- #


REPO = Path(__file__).resolve().parents[2]
SHIPPED_METHOD = REPO / "causalab/configs/methods/interchange.json"
SHIPPED_RUN = REPO / "causalab/configs/runs/weekdays_8b_interchange.json"


def _file_argv(verb: str, path, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(path),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *extra,
    ]


# --------------------------------------------------------------------------- #
#  --register-from-hf: pre-flighting a document on an unregistered model        #
# --------------------------------------------------------------------------- #


class _StubConfig:
    """Just the attributes :func:`model_info_from_hf_config` reads."""

    num_attention_heads = 8
    hidden_size = 64
    num_hidden_layers = 40
    num_key_value_heads = 8
    head_dim = 8
    intermediate_size = 128
    vocab_size = 512
    dtype = "bfloat16"


def _unregistered_key(request) -> str:
    """A key unique to the calling test.

    ``register_model`` writes to a process-global registry, so a shared key
    would leak: whichever test registered it first would make the others'
    "still refuses" assertion vacuous.
    """
    return f"some-org/never-registered-40L-{abs(hash(request.node.name)):x}"


@pytest.fixture
def unregistered_document(tmp_path, request):
    """Corpus 02 retargeted at a key the registry has never heard of."""
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    raw["model"]["key"] = _unregistered_key(request)
    path = tmp_path / "unregistered_im.json"
    path.write_text(json.dumps(raw, indent=2))
    return path


def _verb(verb: str, path: Path, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(path),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *extra,
    ]


@pytest.mark.parametrize("verb", ["validate", "explain", "digest"])
def test_a_pure_verb_refuses_an_unregistered_model_without_the_flag(
    verb, unregistered_document, artifacts_root, capsys
):
    """The invariant: no flag, no network — so the refusal stands."""
    code = main(_verb(verb, unregistered_document, artifacts_root))
    assert code == 1
    assert "[V4]" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["validate", "explain", "digest"])
def test_register_from_hf_lets_a_pure_verb_pre_flight_an_unregistered_model(
    verb, unregistered_document, artifacts_root, capsys, monkeypatch
):
    """The gap all three A3B protocol runs hand-rolled a wrapper around.

    The documented workaround — validate against a *similar* registered model —
    produces a **false** refusal: `[V4] layer 36 out of range for the 36-layer
    model 'Qwen/Qwen3-4B-Instruct-2507'` on a perfectly valid 40-layer
    document. Pre-flighting has to be possible on the model the document names.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, key, **kw: _StubConfig()),
    )
    code = main(
        _verb(verb, unregistered_document, artifacts_root, "--register-from-hf")
    )
    assert code == 0, capsys.readouterr().err


def test_register_from_hf_pre_registers_every_inner_model_of_a_workflow(
    tmp_path, artifacts_root, monkeypatch, capsys, request
):
    """A workflow names several documents, so registering only the outer one
    would pre-flight nothing — which is why the runs' wrappers were
    workflow-aware."""
    import transformers

    seen: list[str] = []

    def fake(cls, key, **kw):
        seen.append(key)
        return _StubConfig()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(fake))
    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    key = _unregistered_key(request)
    raw["model"]["key"] = key
    inner = tmp_path / "inner_im.json"
    inner.write_text(json.dumps(raw, indent=2))
    workflow = tmp_path / "wf.json"
    workflow.write_text(
        json.dumps(
            {
                "version": "1",
                "output_dir": "out",
                "steps": {
                    "only": {
                        "type": "intervention_protocol",
                        "document": str(inner),
                    }
                },
            },
            indent=2,
        )
    )
    code = main(_verb("validate", workflow, artifacts_root, "--register-from-hf"))
    assert code == 0, capsys.readouterr().err
    assert key in seen


def test_validate_and_digest_work_on_a_method(capsys, artifacts_root):
    assert main(_file_argv("validate", SHIPPED_METHOD, artifacts_root)) == 0
    assert "method" in capsys.readouterr().out
    assert main(_file_argv("digest", SHIPPED_METHOD, artifacts_root)) == 0
    assert len(capsys.readouterr().out.strip()) == 64


def test_explain_on_a_method_prints_what_must_be_bound(capsys, artifacts_root):
    assert main(_file_argv("explain", SHIPPED_METHOD, artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "binds" in out
    assert "sites.target: layer" in out
    assert "model: key, revision, dtype" in out


def test_a_method_cannot_be_run(capsys, artifacts_root, tmp_path):
    code = main(
        _file_argv("run", SHIPPED_METHOD, artifacts_root, "--out", str(tmp_path))
    )
    assert code == 1
    assert "method file" in capsys.readouterr().err


def test_explain_on_a_split_document_names_its_method(capsys, artifacts_root):
    assert main(_file_argv("explain", SHIPPED_RUN, artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "method    " in out
    assert "(inline)" in out  # one file is one run
    assert "bf16" in out


def test_run_writes_the_protocol_record(capturing_engine, artifacts_root, tmp_path):
    """The record a reproducer reads first: what ran, at what precision, from
    which method, with the provenance digest of every point."""
    assert (
        main(_file_argv("run", SHIPPED_RUN, artifacts_root, "--out", str(tmp_path)))
        == 0
    )
    record = json.loads((tmp_path / "protocol.json").read_text())
    assert record["canonical"]["model"] == {
        "key": "meta-llama/Llama-3.1-8B",
        "revision": "main",
        "dtype": "bf16",
    }
    assert record["method"]["ref"] is None  # inlined in the run document
    assert len(record["method"]["digest"]) == 64
    assert [point["index"] for point in record["points"]] == [0]
    assert record["points"][0]["digest"] == record["document_digest"]
