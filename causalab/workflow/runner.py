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

    def _local(self) -> Any:
        from causalab.protocol.resolve import FileArtifacts

        return FileArtifacts(root=self.run_root)

    def _in_run_tree(self, ref: str) -> bool:
        head = ref.split("/", 1)[0]
        return (self.run_root / head).exists()

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
    overlay = OverlayArtifacts(run_root=output_dir, outer=env.artifacts)
    run_env = ResolutionEnv(
        datasets=env.datasets, artifacts=overlay, model_info=env.model_info
    )
    step_manifest: dict[str, Any] = {}
    run_axes: dict[str, tuple[Any, ...]] = {}

    for name in loaded.order:
        step = loaded.document.steps[name]
        step_dir = output_dir / name
        step_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(step, ProtocolStep):
            entry = _run_protocol_step(
                name, step, loaded, run_env, step_dir, backends, run_axes
            )
        elif isinstance(step, SelectStep):
            entry = _run_select_step(name, step, output_dir, step_dir, run_axes)
        else:
            assert isinstance(step, PlotStep)
            entry = _run_plot_step(name, step, output_dir, step_dir, run_axes)
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
        "document": step.document,
        "backend": backend.name,
        "document_digest": inner.document_digest,  # fully resolved (§7)
        "points": len(inner.expansion.points),
        "files": sorted(result.files),
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
) -> "Any":
    """The mean-over-examples table, one row per sweep-coordinate group
    (§2.3): columns = the producing document's axis ids + the value."""
    import pandas as pd

    frame = pd.read_parquet(output_dir / from_step / table)
    if value_column not in frame.columns:
        raise ProtocolError("P2", f"{from_step}/{table} has no column {value_column!r}")
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
) -> dict[str, Any]:
    table = _aggregated(output_dir, step.from_, step.table, step.value, run_axes)
    row = (
        table.loc[table[step.value].idxmax()]
        if step.choose == "max"
        else table.loc[table[step.value].idxmin()]
    )
    values: dict[str, Any] = {}
    for key, column in step.emit.items():
        if column not in row.index:
            raise ProtocolError(
                "P2",
                f"emit column {column!r} is not in the aggregated table "
                f"({list(row.index)}) — the producing run carried no such axis",
            )
        values[key] = _decode_cell(row[column])
    (step_dir / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    return {
        "type": "select",
        "from": step.from_,
        "chosen": values,
        "score": float(row[step.value]),
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
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = _aggregated(output_dir, step.from_, step.table, step.value, run_axes)
    figure, axes = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if step.plot == "heatmap":
        assert step.y is not None  # parse guarantees it
        grid = table.pivot(index=step.y, columns=step.x, values=step.value)
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
        "plot": step.plot,
        "from": step.from_,
        "file": step.file_path,
    }
