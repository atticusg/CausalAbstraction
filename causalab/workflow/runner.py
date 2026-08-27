"""The workflow runner (docs/workflow_protocol.md §8).

Executes a loaded workflow: steps in topological order, each step's outputs
under ``<out-root>/<output_dir>/<step>/``, protocol steps through the standard backend
routing against the run-tree/external artifact overlay, and script steps by
resolving their inputs, calling ``main(inputs, outputs)``, then verifying and
stamping what they wrote.

There is no backend choice at the workflow level — backends are chosen per
protocol step from the list the caller supplies (v2 ships one).

**The run tree is the publication.** There is no `save` section and no copy
step: a step's declared outputs land in its own directory and stay there
(§0). What the runner adds beside them is a record — ``_step.json`` per step,
``workflow.json`` for the run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.backend import Backend, ExecutionRequest, choose_backend
from causalab.protocol.errors import ProtocolError
from causalab.protocol.loader import apply_overrides, load, load_text
from causalab.protocol.resolve import ArtifactStore, ResolutionEnv
from causalab.protocol.sweep import DEFAULT_POINT_CAP
from causalab.io.step_record import SIDECAR, write_sidecar
from causalab.protocol.tables import TABLE_SUFFIX, read_table
from causalab.workflow.document import (
    LoadedWorkflow,
    OutputDecl,
    ProtocolStep,
    Reference,
    ScriptStep,
    resolve_script,
)

__all__ = [
    "OverlayArtifacts",
    "SIDECAR",
    "WorkflowRunResult",
    "run_workflow",
]

#: What identity a script-written tensor is stamped as coming from.
SCRIPT_BACKEND = "script"


@dataclasses.dataclass(frozen=True)
class OverlayArtifacts:
    """The §3 overlay: step outputs in the run tree shadow the external
    artifacts root. Every check is real here — this is the run-time store the
    load-time :class:`~causalab.workflow.document.DeferredArtifacts` defers
    to."""

    run_root: Path
    outer: ArtifactStore
    step_names: frozenset[str]

    def _local(self) -> Any:
        from causalab.protocol.resolve import FileArtifacts

        return FileArtifacts(root=self.run_root)

    def _in_run_tree(self, ref: str) -> bool:
        # STEP NAMES shadow the external root (§3) — exactly the names, never
        # mere directory existence: a rerun into a used output dir, or an
        # external ref matching a stray directory, must not change resolution
        return ref.split("/", 1)[0] in self.step_names

    def read_value(self, artifact: str, key: str) -> Any:
        if self._in_run_tree(artifact):
            return self._local().read_value(artifact, key)
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        if self._in_run_tree(file_path):
            return self._local().file_digest(file_path)
        return self.outer.file_digest(file_path)

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        if self._in_run_tree(file_path):
            return self._local().read_identity(file_path)
        return self.outer.read_identity(file_path)

    def resolve_path(self, file_path: str) -> Path:
        if self._in_run_tree(file_path):
            return self.run_root / file_path
        outer_resolve = getattr(self.outer, "resolve_path", None)
        if outer_resolve is None:
            raise ProtocolError(
                "P2", f"outer artifact store cannot resolve {file_path!r} to a file"
            )
        return outer_resolve(file_path)


@dataclasses.dataclass(frozen=True)
class WorkflowRunResult:
    """What one workflow run produced: the manifest (also on disk as
    ``workflow.json``) and the run tree it landed in."""

    manifest: Mapping[str, Any]
    run_root: Path


def run_workflow(
    loaded: LoadedWorkflow,
    env: ResolutionEnv,
    out_root: Path,
    backends: Sequence[Backend],
    *,
    resume: bool = False,
    reuse_nondeterministic: bool = False,
) -> WorkflowRunResult:
    """Execute one loaded workflow into ``<out_root>/<output_dir>/``."""
    run_root = out_root / loaded.document.output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    overlay = OverlayArtifacts(
        run_root=run_root,
        outer=env.artifacts,
        step_names=frozenset(loaded.document.steps),
    )
    run_env = ResolutionEnv(
        datasets=env.datasets, artifacts=overlay, model_info=env.model_info
    )
    step_manifest: dict[str, Any] = {}

    for name in loaded.order:
        step = loaded.document.steps[name]
        step_dir = run_root / name
        step_dir.mkdir(parents=True, exist_ok=True)
        skipped = _reusable(
            loaded, name, step, step_dir, resume, reuse_nondeterministic
        )
        if skipped is not None:
            step_manifest[name] = skipped
            continue
        if isinstance(step, ProtocolStep):
            entry = _run_protocol_step(name, step, loaded, run_env, step_dir, backends)
        else:
            entry = _run_script_step(name, step, loaded, run_root, step_dir)
        step_manifest[name] = entry
        write_sidecar(step_dir, entry)

    manifest = {
        "workflow_digest": loaded.digest,
        "output_dir": loaded.document.output_dir,
        "steps": step_manifest,
    }
    if loaded.nondeterministic:
        manifest["nondeterministic"] = list(loaded.nondeterministic)
    (run_root / "workflow.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return WorkflowRunResult(manifest=manifest, run_root=run_root)


def _reusable(
    loaded: LoadedWorkflow,
    name: str,
    step: Any,
    step_dir: Path,
    resume: bool,
    reuse_nondeterministic: bool,
) -> dict[str, Any] | None:
    """The prior run's record for ``name`` if ``--resume`` may reuse it (§8).

    The digest comparison is what makes this correct: a script step's digest
    carries its script's content hash, so editing the script busts the reuse.
    A step that declared itself non-deterministic is never reused silently —
    replaying it is exactly what it said it cannot guarantee."""
    if not resume:
        return None
    record_path = step_dir / SIDECAR
    if not record_path.is_file():
        return None
    try:
        with record_path.open() as handle:
            record = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    want = _step_identity(loaded, name, step)
    if record.get("identity") != want:
        return None
    if isinstance(step, ScriptStep) and not step.is_deterministic:
        if not reuse_nondeterministic:
            return None
    for rel in record.get("files", []):
        if not (step_dir / str(rel)).is_file():
            return None
    return {**record, "status": "reused"}


def _step_identity(loaded: LoadedWorkflow, name: str, step: Any) -> str:
    """What `--resume` compares: the step's own digest."""
    if isinstance(step, ScriptStep):
        return loaded.step_digests[name]
    return loaded.inner_digests[name]


# --------------------------------------------------------------------------- #
# protocol steps
# --------------------------------------------------------------------------- #


def _run_protocol_step(
    name: str,
    step: ProtocolStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    backends: Sequence[Backend],
) -> dict[str, Any]:
    doc_path = (loaded.workflow_dir / step.document).resolve()
    overridden = apply_overrides(dict(load_text(doc_path)), step.set)
    # real resolution now: earlier steps' outputs exist in the run tree
    inner = load(
        overridden,
        run_env,
        point_cap=step.max_points if step.max_points is not None else DEFAULT_POINT_CAP,
    )
    backend = choose_backend(list(inner.point_documents), backends)
    request = ExecutionRequest(
        points=tuple(p.raw for p in inner.expansion.points),
        canonical=inner.canonical_points,
        digests=inner.point_digests,
        coords=tuple(p.coords for p in inner.expansion.points),
        document_digest=inner.document_digest,
        env=run_env,
        output_dir=step_dir,
    )
    result = backend.execute(request)
    return {
        "type": "intervention_protocol",
        "status": "completed",
        "identity": inner.document_digest,
        "document": step.document,
        "backend": backend.name,
        "document_digest": inner.document_digest,  # fully resolved (§7)
        "points": len(inner.expansion.points),
        "point_digests": list(inner.point_digests),  # the provenance units (§7)
        # the sweep axes a downstream script groups by (§6)
        "axes": [axis.id for axis in inner.expansion.axes],
        "files": sorted(result.files),
    }


# --------------------------------------------------------------------------- #
# script steps
# --------------------------------------------------------------------------- #


def _run_script_step(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    step_dir: Path,
) -> dict[str, Any]:
    """Resolve inputs, run the script, verify and stamp its outputs (§4)."""
    from causalab.io import step_io

    resolved, tensor_identities = _resolve_inputs(name, step, run_root)
    outputs = {slot: step_dir / decl.file for slot, decl in step.outputs.items()}

    if step.runtime and step.runtime.get("isolate"):
        _run_isolated(name, step, loaded, resolved, outputs)
    else:
        _run_in_process(name, step, loaded, resolved, outputs)

    identity = step_io.inherited_identity(tensor_identities)
    identity["produced_by"] = loaded.step_digests[name]
    identity["backend"] = SCRIPT_BACKEND

    for slot, decl in step.outputs.items():
        target = outputs[slot]
        what = f"step {name!r}: output {slot!r} ({decl.file})"
        if not target.is_file():
            raise ProtocolError(
                "P2",
                f"{what} was not written — a script step must create every "
                "output it declares",
            )
        if decl.suffix == TABLE_SUFFIX:
            _verify_json_output(target, decl, what)
        elif decl.suffix == ".safetensors":
            step_io.stamp_tensor(target, identity, what=what)
        # a visualization output (.png/.pdf/.html) carries no record, so there
        # is nothing to stamp and no shape to check — existence is the contract

    return {
        "type": "script",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "script": step.script,
        "script_sha256": step.script_sha256,
        "digest": loaded.step_digests[name],
        "is_deterministic": step.is_deterministic,
        "inputs": {
            key: (value.target if isinstance(value, Reference) else value)
            for key, value in step.inputs.items()
        },
        "axes": [],  # a script step carries no sweep coordinates of its own
        "files": sorted(decl.file for decl in step.outputs.values()),
        **({"runtime": dict(step.runtime)} if step.runtime else {}),
    }


def _resolve_inputs(
    name: str, step: ScriptStep, run_root: Path
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """The §3 grammar, resolved: a locator becomes a path, a selector reads
    through it. Also returns the identity of every tensor input, which is what
    a safetensors output inherits (§4)."""
    from causalab.workflow.document import _repo_root
    from causalab.io import step_io

    resolved: dict[str, Any] = {}
    identities: list[Mapping[str, Any]] = []
    for slot, value in step.inputs.items():
        if not isinstance(value, Reference):
            resolved[slot] = value
            continue
        what = f"step {name!r}: input {slot!r} ({value.target})"
        if value.step is not None:
            target = run_root / value.step / str(value.file)
        else:
            candidate = Path(str(value.path))
            target = candidate if candidate.is_absolute() else _repo_root() / candidate
        if not target.is_file():
            raise ProtocolError("P2", f"{what} does not exist at {str(target)!r}")
        if value.key is not None:
            values = step_io.read_values(target)
            if value.key not in values:
                raise ProtocolError(
                    "P2",
                    f"{what}: no key {value.key!r} in {target.name} "
                    f"(has {sorted(values)})",
                )
            resolved[slot] = values[value.key]
            continue
        if value.entry is not None or value.slot is not None:
            tensor, identity = step_io.read_tensor_with_identity(
                target, slot=value.slot, entry=value.entry, what=what
            )
            resolved[slot] = tensor
            identities.append(identity)
            continue
        resolved[slot] = target
        if target.suffix == ".safetensors":
            # an unselected bundle is handed over as a path, but its identity
            # still flows: a fit over one harvest is bound to that harvest
            try:
                _, identity = step_io.read_tensor_with_identity(target, what=what)
                identities.append(identity)
            except ProtocolError:
                pass  # multi-slot bundle: nothing unambiguous to inherit
    return resolved, identities


def _verify_json_output(target: Path, decl: OutputDecl, what: str) -> None:
    """A declared shape, checked against what actually landed (§2.3).

    This is where v1's load-time column check moved to. It is later than a load
    error, but it is against the real file rather than a declaration believed on
    faith — and the declaration is what a consuming step was validated
    against."""
    from causalab.io import step_io

    if decl.keys is not None:
        written = step_io.read_values(target)
        missing = sorted(set(decl.keys) - set(written))
        if missing:
            raise ProtocolError(
                "P2",
                f"{what}: declares keys {sorted(decl.keys)} but wrote "
                f"{sorted(written)} — missing {missing}",
            )
        return
    if decl.columns is None:
        return
    rows = read_table(target)
    if not rows:
        return  # an empty table satisfies any column declaration
    present = set(rows[0])
    missing = sorted(set(decl.columns) - present)
    if missing:
        raise ProtocolError(
            "P2",
            f"{what}: declares columns {sorted(decl.columns)} but wrote "
            f"{sorted(present)} — missing {missing}",
        )


def _run_in_process(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Import the script and call ``main`` (§4).

    The import happens *here*, not at load: ``validate``/``digest`` must not
    pull a script's dependencies in (§4.2). By the time we are running, the
    process is already committed to executing."""
    import importlib.util

    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    spec = importlib.util.spec_from_file_location(f"_causalab_step_{name}", target)
    if spec is None or spec.loader is None:
        raise ProtocolError("P2", f"step {name!r}: cannot import {str(target)!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ProtocolError(
            "P2", f"step {name!r}: {step.script!r} has no callable 'main'"
        )
    main(inputs, dict(outputs))


def _run_isolated(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Run the script in a subprocess with its own dependency set (§4.1).

    Tensor-valued inputs cannot cross the process boundary, so an isolated step
    takes its tensors as *paths* — which means it must not use an ``entry``
    selector. Refused here rather than silently pickling a tensor into JSON."""
    runtime = dict(step.runtime or {})
    payload: dict[str, Any] = {}
    for slot, value in inputs.items():
        if isinstance(value, Path):
            payload[slot] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            payload[slot] = value
        else:
            raise ProtocolError(
                "P2",
                f"step {name!r}: input {slot!r} is a {type(value).__name__}, "
                "which cannot cross a process boundary — an isolated step takes "
                "tensors as paths, so drop the 'entry'/'slot' selector and read "
                "the bundle inside the script",
            )
    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    request = {
        "script": str(target),
        "inputs": payload,
        "outputs": {slot: str(path) for slot, path in outputs.items()},
    }
    command = ["uv", "run"]
    for dep in runtime.get("deps", ()):
        command += ["--with", str(dep)]
    command += ["python", "-m", "causalab.workflow.isolate"]

    environ = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")
        if key in os.environ
    }
    for passthrough in runtime.get("env", ()):
        if str(passthrough) in os.environ:
            environ[str(passthrough)] = os.environ[str(passthrough)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=environ,
            cwd=str(_shim_cwd()),
        )
    except FileNotFoundError as err:
        raise ProtocolError(
            "P2",
            f"step {name!r}: isolation needs 'uv' on PATH ({err})",
        ) from err
    if completed.returncode != 0:
        raise ProtocolError(
            "P2",
            f"step {name!r}: isolated script failed (exit "
            f"{completed.returncode})\n{completed.stderr.strip()}",
        )


def _shim_cwd() -> Path:
    """Where the subprocess runs, so ``uv run`` finds this project."""
    import causalab

    return Path(causalab.__file__).resolve().parent.parent


def _python() -> str:
    return sys.executable
