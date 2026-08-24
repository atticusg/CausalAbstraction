"""The workflow runner (docs/workflow_protocol.md §8).

Executes a loaded workflow: steps in topological order, each step's
outputs under ``<run>/<step>/``, protocol steps through the standard
backend routing against the run-tree/external artifact overlay, ``select``
as group-by-coordinates → mean-over-examples → argmax/argmin → a values
table, ``plot`` as the closed two-kind figure vocabulary, then the save
manifest published to the output root and ``workflow.json`` stamped.

There is no backend choice at the workflow level — backends are chosen
per protocol step from the list the caller supplies (v1 ships one).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.backend import Backend, ExecutionRequest, choose_backend
from causalab.protocol.errors import ProtocolError
from causalab.protocol.loader import apply_overrides, load, load_text
from causalab.protocol.resolve import ArtifactStore, ResolutionEnv
from causalab.protocol.sweep import DEFAULT_POINT_CAP
from causalab.protocol.workflow import (
    LoadedWorkflow,
    PlotStep,
    ProtocolStep,
    SelectStep,
    TransformStep,
)

__all__ = ["OverlayArtifacts", "WorkflowRunResult", "run_workflow"]


@dataclasses.dataclass(frozen=True)
class OverlayArtifacts:
    """The §3 overlay: step outputs in the run tree shadow the external
    artifacts root. Every check is real here — this is the run-time store
    the load-time :class:`~causalab.protocol.workflow.DeferredArtifacts`
    defers to."""

    run_root: Path
    outer: ArtifactStore
    step_names: frozenset[str]

    def _local(self) -> Any:
        from causalab.protocol.resolve import FileArtifacts

        return FileArtifacts(root=self.run_root)

    def _in_run_tree(self, ref: str) -> bool:
        # STEP NAMES shadow the external root (§3) — exactly the names,
        # never mere directory existence: a rerun into a used output dir or
        # an external ref matching a stray directory must not change resolution
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
    ``workflow.json``) and the published files."""

    manifest: Mapping[str, Any]
    published: Mapping[str, Path]


def run_workflow(
    loaded: LoadedWorkflow,
    env: ResolutionEnv,
    output_dir: Path,
    backends: Sequence[Backend],
) -> WorkflowRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = OverlayArtifacts(
        run_root=output_dir,
        outer=env.artifacts,
        step_names=frozenset(loaded.document.steps),
    )
    run_env = ResolutionEnv(
        datasets=env.datasets, artifacts=overlay, model_info=env.model_info
    )
    step_manifest: dict[str, Any] = {}
    run_axes: dict[str, tuple[Any, ...]] = {}
    # a transform step's table is consumed as written: its op already decided
    # what a row is, and it carries no sweep coordinates to group by (§2.4)
    as_written = frozenset(
        name
        for name, step in loaded.document.steps.items()
        if isinstance(step, TransformStep)
    )

    for name in loaded.order:
        step = loaded.document.steps[name]
        step_dir = output_dir / name
        step_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(step, ProtocolStep):
            entry = _run_protocol_step(
                name, step, loaded, run_env, step_dir, backends, run_axes
            )
        elif isinstance(step, TransformStep):
            entry = _run_transform_step(name, step, loaded, output_dir, step_dir)
        elif isinstance(step, SelectStep):
            entry = _run_select_step(
                name, step, output_dir, step_dir, run_axes, as_written
            )
        else:
            assert isinstance(step, PlotStep)
            entry = _run_plot_step(
                name, step, output_dir, step_dir, run_axes, as_written
            )
        step_manifest[name] = entry

    published: dict[str, Path] = {}
    for entry in loaded.document.save:
        source = output_dir / entry.step / entry.value
        if not source.is_file():
            raise ProtocolError(
                "P2",
                f"save entry {entry.step}/{entry.value} was not produced — the "
                "run tree is missing a promised output",
            )
        target = output_dir / entry.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        published[entry.file_path] = target

    manifest = {
        "workflow_digest": loaded.digest,
        "steps": step_manifest,
        "published": sorted(published),
    }
    (output_dir / "workflow.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return WorkflowRunResult(manifest=manifest, published=published)


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
    run_axes: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    doc_path = (loaded.workflow_dir / step.document).resolve()
    overridden = apply_overrides(dict(load_text(doc_path)), step.set)
    # real resolution now: earlier steps' outputs exist in the run tree
    inner = load(
        overridden,
        run_env,
        point_cap=step.max_points if step.max_points is not None else DEFAULT_POINT_CAP,
    )
    run_axes[name] = inner.expansion.axes
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
        "type": "protocol",
        "status": "completed",
        "document": step.document,
        "backend": backend.name,
        "document_digest": inner.document_digest,  # fully resolved (§7)
        "points": len(inner.expansion.points),
        "point_digests": list(inner.point_digests),  # the provenance units (§7)
        "files": sorted(result.files),
    }


# --------------------------------------------------------------------------- #
# transform steps
# --------------------------------------------------------------------------- #


def _run_transform_step(
    name: str,
    step: TransformStep,
    loaded: LoadedWorkflow,
    output_dir: Path,
    step_dir: Path,
) -> dict[str, Any]:
    """Run one registered op over earlier steps' outputs (§2.4).

    The op body and its numerics are imported here, not at module scope: this
    package drives backends but links against none, and a workflow with no
    transform step must not pay for torch.

    Inputs are read straight out of the run tree rather than through the
    artifact overlay: unlike a protocol step's document, which may name either
    an in-run step or the external artifacts root, a transform input *always*
    names a step — rule 4 checked it at load."""
    from causalab.transform import io as transform_io
    from causalab.transform.registry import lookup
    from causalab.transform.schema import Table, TransformError

    op = lookup(step.op)
    inputs: dict[str, Any] = {}
    inherited: list[Mapping[str, Any]] = []
    for slot, ref in step.inputs.items():
        source = output_dir / ref.step / ref.value
        what = f"step {name!r}: input {slot!r} ({ref.step}/{ref.value})"
        if isinstance(op.inputs[slot], Table):
            inputs[slot] = transform_io.read_table(source)
            continue
        tensor, identity = transform_io.read_tensor(
            source, slot=ref.slot, entry=ref.entry, what=what
        )
        inputs[slot] = tensor
        inherited.append(identity)

    # typed as `object`: the record says an op returns {slot: value}, but an op
    # body is ordinary Python, so the shape is checked rather than assumed
    returned: object = op.fn(inputs=inputs, params=dict(step.params))
    if not isinstance(returned, Mapping):
        raise TransformError(
            f"step {name!r}: op {op.id} returned {type(returned).__name__}, not "
            "a {slot: value} mapping"
        )
    produced: Mapping[str, Any] = returned
    undeclared = sorted(set(produced) - set(op.outputs))
    absent = sorted(set(op.outputs) - set(produced))
    if undeclared or absent:
        raise TransformError(
            f"step {name!r}: op {op.id} must return exactly its declared slots "
            f"{sorted(op.outputs)}"
            + (f"; it added {undeclared}" if undeclared else "")
            + (f"; it omitted {absent}" if absent else "")
        )

    identity = transform_io.inherited_identity(inherited)
    identity.update(
        {field: step.params[param] for field, param in op.identity_from_params.items()}
    )
    identity["produced_by"] = loaded.transform_digests[name]
    identity["backend"] = "transform"

    for slot, file_path in step.outputs.items():
        target = step_dir / file_path
        what = f"step {name!r}: output {slot!r} ({file_path})"
        decl = op.outputs[slot]
        if isinstance(decl, Table):
            transform_io.write_table(produced[slot], target, decl, what=what)
        else:
            transform_io.write_tensor(
                produced[slot], target, slot=slot, identity=identity, what=what
            )
    return {
        "type": "transform",
        "status": "completed",
        "op": op.id,
        "digest": loaded.transform_digests[name],  # the provenance unit (§2.4)
        "inputs": {
            slot: f"{ref.step}/{ref.value}" for slot, ref in step.inputs.items()
        },
        "files": sorted(step.outputs.values()),
    }


# --------------------------------------------------------------------------- #
# select steps
# --------------------------------------------------------------------------- #


def _aggregated(
    output_dir: Path,
    from_step: str,
    table: str,
    value_column: str,
    run_axes: Mapping[str, tuple[Any, ...]],
    as_written: frozenset[str] = frozenset(),
) -> "Any":
    """The mean-over-examples table, one row per sweep-coordinate group
    (§2.3): columns = the producing document's axis ids + the value.

    A **transform** producer is exempt: its op already decided what a row
    means, and its table carries no sweep coordinates, so re-aggregating here
    would silently collapse the rows the document was validated against."""
    import pandas as pd

    frame = pd.read_parquet(output_dir / from_step / table)
    if value_column not in frame.columns:
        raise ProtocolError("P2", f"{from_step}/{table} has no column {value_column!r}")
    if from_step in as_written:
        return frame
    coord_columns = [
        axis.id for axis in run_axes.get(from_step, ()) if axis.id in frame.columns
    ]
    if not coord_columns:
        return frame[[value_column]].mean().to_frame().T
    return frame.groupby(coord_columns, sort=True)[value_column].mean().reset_index()


def _decode_cell(value: Any) -> Any:
    """Coordinate cells round-trip through the metric tables as JSON when
    they are structured (a swept position spec); decode them back."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _run_select_step(
    name: str,
    step: SelectStep,
    output_dir: Path,
    step_dir: Path,
    run_axes: Mapping[str, tuple[Any, ...]],
    as_written: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    table = _aggregated(
        output_dir, step.from_, step.table, step.value, run_axes, as_written
    )
    chosen_index = (
        table[step.value].idxmax()
        if step.choose == "max"
        else table[step.value].idxmin()
    )
    # single-row FRAME indexing: a row Series would upcast mixed dtypes and
    # emit integer coordinates as floats, which the consuming document's
    # strict parse then refuses
    row = table.loc[[chosen_index]]
    values: dict[str, Any] = {}
    for key, column in step.emit.items():
        if column not in row.columns:
            raise ProtocolError(
                "P2",
                f"emit column {column!r} is not in the aggregated table "
                f"({list(row.columns)}) — the producing run carried no such axis",
            )
        values[key] = _decode_cell(row[column].iloc[0])
    (step_dir / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    return {
        "type": "select",
        "status": "completed",
        "from": step.from_,
        "chosen": values,
        "score": float(row[step.value].iloc[0]),
    }


# --------------------------------------------------------------------------- #
# plot steps
# --------------------------------------------------------------------------- #


def _run_plot_step(
    name: str,
    step: PlotStep,
    output_dir: Path,
    step_dir: Path,
    run_axes: Mapping[str, tuple[Any, ...]],
    as_written: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = _aggregated(
        output_dir, step.from_, step.table, step.value, run_axes, as_written
    )
    figure, axes = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if step.plot == "heatmap":
        assert step.y is not None  # parse guarantees it
        grid = table.pivot_table(
            index=step.y, columns=step.x, values=step.value, aggfunc="mean"
        )
        image = axes.imshow(grid.to_numpy(), aspect="auto", origin="lower")
        axes.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
        axes.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
        axes.set_xlabel(step.x)
        axes.set_ylabel(step.y)
        figure.colorbar(image, ax=axes, label=step.value)
    else:  # lines
        if step.series is not None:
            for series_value, group in table.groupby(step.series, sort=True):
                group = group.sort_values(step.x)
                axes.plot(
                    group[step.x],
                    group[step.value],
                    marker="o",
                    label=f"{step.series}={series_value}",
                )
            axes.legend()
        else:
            ordered = table.sort_values(step.x)
            axes.plot(ordered[step.x], ordered[step.value], marker="o")
        axes.set_xlabel(step.x)
        axes.set_ylabel(step.value)
    axes.set_title(f"{step.from_}/{step.table} — {step.value}")
    target = step_dir / step.file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return {
        "type": "plot",
        "status": "completed",
        "plot": step.plot,
        "from": step.from_,
        "file": step.file_path,
    }
