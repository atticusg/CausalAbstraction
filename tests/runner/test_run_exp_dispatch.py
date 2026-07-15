"""Regression tests for ``scripts/run_exp.sh`` SLURM / session-local dispatch (#166).

Four adjacent bugs in the wrapper made ``--slurm`` dispatch fail for session-local
runner configs; they recurred across ≥7 sessions and all 9 manifold worktrees of
the 2026-06 sweep (umbrella #217). Each is exercised here so they can't silently
return:

  * **"runners/-prefix double-resolution"** (bug A): the submit-side discovery
    rewrites a bare runner name to ``runners/<name>`` and forwards *that* into the
    sbatch re-exec, where discovery prepends ``runners/`` a second time and looks
    for ``runners/runners/<name>.yaml``. Fixed by stripping a leading ``runners/``
    at the top of the script. Guarded by ``test_slurm_runner_not_double_prefixed``.
  * **"forward-array order"** (bug B): the slurm ``forward`` array put ``$runner``
    *before* ``--config-dir``/``--experiment-root``; the wrapper's arg-parser
    ``break``s at the first positional, so on re-exec those flags fell through to
    Hydra. Fixed by emitting flags first. Guarded by
    ``test_slurm_forward_flags_precede_runner``.
  * **sub-bug C**: ``--config-dir "$session_code/configs"`` was only added when the
    runner was discovered session-local. A runner copied in-tree is "qualified", so
    only ``++hydra.searchpath`` was added — which can't resolve nested ``defaults:``
    — and Hydra failed with ``Could not load 'analysis/<name>'``. Fixed by adding
    ``--config-dir`` whenever session configs exist. Guarded by
    ``test_inline_adds_config_dir_for_qualified_runner``.
  * **sub-bug D**: ``CAUSALAB_SESSION_CODE`` did not survive into the sbatch job
    env, so in-job session re-detection failed. Fixed by forwarding it via
    ``--export``. Guarded by ``test_slurm_exports_session_code``.

The real wrapper is driven with ``sbatch`` and ``uv`` stubbed onto ``PATH``, so
nothing is submitted and no model/GPU work runs — we assert on the captured argv.
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXP = REPO_ROOT / "scripts" / "run_exp.sh"

# A shipped runner, group-qualified so discovery is deterministic (no find/ambiguity).
SHIPPED_RUNNER = "hierarchical_equality/he_locate"


@pytest.fixture
def session_tree(tmp_path):
    """Build a fake research session matching the ``*agent_logs/*/artifacts*`` pattern.

    Returns ``(session_dir, artifacts_dir, session_code_dir)`` as resolved Paths.
    A minimal session-local runner lives at
    ``code/configs/runners/mfld/manifold_test.yaml``.
    """
    base = tmp_path.resolve()
    session_dir = base / "agent_logs" / "2026-06-11--dispatch-test--fixture"
    artifacts = session_dir / "artifacts"
    code = session_dir / "code"
    runners = code / "configs" / "runners" / "mfld"
    artifacts.mkdir(parents=True)
    runners.mkdir(parents=True)
    # Content is irrelevant — the slurm path stubs slurm_args, and discovery only
    # checks for the file's existence — but keep it plausible.
    (runners / "manifold_test.yaml").write_text(
        "# @package _global_\ndefaults:\n  - /base\n"
    )
    return session_dir, artifacts, code


@pytest.fixture
def stub_bin(tmp_path):
    """Put stub ``sbatch`` and ``uv`` on PATH; return (bin_dir, capture_dir).

    ``sbatch`` records its argv to ``<capture>/sbatch.args`` and exits 0 (so the
    wrapper's ``exec sbatch`` ends with rc 0). ``uv`` mimics
    ``causalab.runner.slurm_args``'s one-line ``<gpus> <time> <job_name>`` contract,
    and records ``run_exp`` argv to ``<capture>/run_exp.args`` for the inline path.
    """
    bin_dir = tmp_path / "stub_bin"
    capture = tmp_path / "capture"
    bin_dir.mkdir()
    capture.mkdir()

    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{capture}/sbatch.args"\nexit 0\n'
    )
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *slurm_args*) echo "1 01:00:00 causalab_test"; exit 0 ;;\n'
        f'  *run_exp*) printf "%s\\n" "$@" > "{capture}/run_exp.args"; exit 0 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    sbatch.chmod(0o755)
    uv.chmod(0o755)
    return bin_dir, capture


def _run(args, stub_bin, extra_env=None):
    bin_dir, _ = stub_bin
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    # A stray CAUSALAB_SESSION_CODE in the test runner's env would override the
    # experiment-root-derived session detection; clear it for a clean slate.
    env.pop("CAUSALAB_SESSION_CODE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(RUN_EXP), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _read_args(capture_dir, name):
    path = Path(capture_dir) / name
    assert path.exists(), f"{name} was never written — wrapper did not reach it"
    return path.read_text().splitlines()


# ---------------------------------------------------------------------------
# SLURM submit path (session-local runner)
# ---------------------------------------------------------------------------


def test_slurm_dispatch_session_local_runner_exits_zero(session_tree, stub_bin):
    """Acceptance #2: ``--slurm`` with a session-local runner reaches sbatch, rc 0."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    assert any("manifold_test" in a for a in sbatch_args)


def test_slurm_forward_flags_precede_runner(session_tree, stub_bin):
    """Bug B (forward-array order): --experiment-root must precede the runner positional."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    runner_idx = next(
        i for i, a in enumerate(sbatch_args) if a.endswith("manifold_test")
    )
    er_idx = sbatch_args.index("--experiment-root")
    assert er_idx < runner_idx, sbatch_args


def test_slurm_runner_not_double_prefixed(session_tree, stub_bin):
    """Bug A (runners/-prefix double-resolution): forwarded runner is singly prefixed."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    assert not any("runners/runners/" in a for a in sbatch_args), sbatch_args
    runner_args = [a for a in sbatch_args if a.endswith("manifold_test")]
    assert runner_args == ["runners/mfld/manifold_test"], sbatch_args


def test_slurm_exports_session_code(session_tree, stub_bin):
    """Sub-bug D: CAUSALAB_SESSION_CODE is forwarded into the job via --export."""
    session_dir, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    assert f"--export=ALL,CAUSALAB_SESSION_CODE={session_dir}" in sbatch_args, (
        sbatch_args
    )


# ---------------------------------------------------------------------------
# Inline path (sub-bug C)
# ---------------------------------------------------------------------------


def test_inline_adds_config_dir_for_qualified_runner(session_tree, stub_bin):
    """Sub-bug C: a qualified (non-session-local) runner still gets --config-dir
    pointing at the session configs whenever session code is present.

    Discriminating case: a *shipped* runner discovered with the session
    ``--experiment-root`` set. Under the old ``runner_session_local`` gate this
    added only ``++hydra.searchpath`` (which cannot resolve nested ``defaults:``);
    now ``--config-dir`` is always added when session configs exist.
    """
    _, artifacts, code = session_tree
    proc = _run(
        ["--experiment-root", str(artifacts), SHIPPED_RUNNER],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    run_exp_args = _read_args(stub_bin[1], "run_exp.args")
    assert "--config-dir" in run_exp_args, run_exp_args
    cd_idx = run_exp_args.index("--config-dir")
    assert run_exp_args[cd_idx + 1] == f"{code}/configs", run_exp_args


# ---------------------------------------------------------------------------
# Hydra introspection flag ordering (#269 I4)
# ---------------------------------------------------------------------------


def test_cfg_flag_before_runner_is_forwarded_not_runner(stub_bin):
    """`run_exp.sh --cfg job <runner>` forwards `--cfg job` to Hydra and still
    resolves the runner — instead of mistaking `--cfg` for the runner name."""
    proc = _run(["--cfg", "job", SHIPPED_RUNNER], stub_bin)
    assert proc.returncode == 0, proc.stderr
    run_exp_args = _read_args(stub_bin[1], "run_exp.args")
    # The runner reaches Hydra as the --config-name value, not "--cfg".
    cn_idx = run_exp_args.index("--config-name")
    assert run_exp_args[cn_idx + 1] == f"runners/{SHIPPED_RUNNER}", run_exp_args
    # And the introspection flag is forwarded.
    assert "--cfg" in run_exp_args and "job" in run_exp_args, run_exp_args


def test_cfg_flag_equals_form_forwarded(stub_bin):
    """The `--cfg=job` form is forwarded as `--cfg job`."""
    proc = _run(["--cfg=job", SHIPPED_RUNNER], stub_bin)
    assert proc.returncode == 0, proc.stderr
    run_exp_args = _read_args(stub_bin[1], "run_exp.args")
    cfg_idx = run_exp_args.index("--cfg")
    assert run_exp_args[cfg_idx + 1] == "job", run_exp_args


# ---------------------------------------------------------------------------
# Resolved-config snapshot at submit (#269 I5)
# ---------------------------------------------------------------------------


def test_resolved_config_snapshotted_at_submit(session_tree, stub_bin):
    """A session run writes <session>/run/<runner>_resolved.yaml before dispatch."""
    session_dir, artifacts, _ = session_tree
    proc = _run(
        ["--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    snapshot = session_dir / "run" / "manifold_test_resolved.yaml"
    assert snapshot.exists(), f"{snapshot} was not written pre-run"


def test_no_snapshot_without_session(stub_bin, tmp_path):
    """A non-session run (no agent_logs experiment-root) writes no snapshot."""
    proc = _run(
        ["--experiment-root", str(tmp_path / "plain_out"), SHIPPED_RUNNER],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    # No run/ snapshot dir is created outside a session.
    assert not (tmp_path / "plain_out" / "run").exists()


# ---------------------------------------------------------------------------
# SLURM --wait + documented log path (#269 I2, I3)
# ---------------------------------------------------------------------------


def test_slurm_wait_propagates_to_sbatch(session_tree, stub_bin):
    """`--slurm --wait` adds sbatch's own --wait so the call blocks until done."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--wait", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    assert "--wait" in sbatch_args, sbatch_args


def test_slurm_no_wait_by_default(session_tree, stub_bin):
    """Without --wait, sbatch is not passed --wait (preserves exec handoff)."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    sbatch_args = _read_args(stub_bin[1], "sbatch.args")
    assert "--wait" not in sbatch_args, sbatch_args


def test_slurm_echoes_log_path(session_tree, stub_bin):
    """The repo-root-relative slurm_logs path is surfaced at submit (I3)."""
    _, artifacts, _ = session_tree
    proc = _run(
        ["--slurm", "--experiment-root", str(artifacts), "manifold_test"],
        stub_bin,
    )
    assert proc.returncode == 0, proc.stderr
    # Job name comes from the stubbed slurm_args ("causalab_test").
    assert "slurm_logs/causalab_test_<jobid>" in proc.stderr, proc.stderr
