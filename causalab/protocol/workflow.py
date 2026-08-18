"""The workflow-protocol document model (docs/workflow_protocol.md).

Backend-free, like the rest of this package: parsing, the workflow
load-error checklist, the derived dependency graph and schedule, and the
canonical form + digest. Executing a loaded workflow is
:mod:`causalab.workflow.runner`'s job.

Two load-time subtleties the spec commits to:

* **Step-dependent inner documents validate against representative
  values.** A protocol step whose document references another step's
  outputs (`{"artifact": "best", …}`, or a `file_path` under a step's run
  tree) cannot resolve those values before the run. At workflow load the
  refs substitute a *representative* from the emitting select step's
  domain — the producing document's axis values are known at load, so any
  one of them type-checks the consumer honestly — and run-tree
  ``file_path`` loads defer their existence/identity checks to run time
  (the deferring store advertises :meth:`DeferredArtifacts.defers`).
* **Digests split by dependency** (spec §7): a step with no in-run
  references stamps its document's full campaign digest; a step-dependent
  document stamps the digest of its overridden authored form, and the
  fully resolved digests land in the run manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.canonical import canonical_bytes
from causalab.protocol.errors import ParseError, ProtocolError, suggest
from causalab.protocol.loader import LoadedProtocol, apply_overrides, load, load_text
from causalab.protocol.resolve import ArtifactStore, ResolutionEnv
from causalab.protocol.sweep import DEFAULT_POINT_CAP

__all__ = [
    "DeferredArtifacts",
    "LoadedWorkflow",
    "PLOT_KINDS",
    "STEP_TYPES",
    "WorkflowError",
    "WorkflowDocument",
    "is_workflow",
    "load_workflow",
    "parse_workflow",
]

STEP_TYPES: tuple[str, ...] = ("protocol", "select", "plot")
PLOT_KINDS: tuple[str, ...] = ("heatmap", "lines")
CHOOSE_KINDS: tuple[str, ...] = ("max", "min")

_STEP_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

#: Top-level sections in mandatory order (§1).
SECTION_ORDER: tuple[str, ...] = ("version", "description", "steps", "save")


class WorkflowError(ProtocolError):
    """A workflow document violates checklist rule ``rule``
    (docs/workflow_protocol.md §5); code ``W<rule>``."""

    def __init__(self, rule: int, message: str, *, path: str | None = None) -> None:
        if not 1 <= rule <= 10:
            raise AssertionError(f"workflow checklist rule out of range: {rule}")
        self.rule = rule
        super().__init__(f"W{rule}", message, path=path)


# --------------------------------------------------------------------------- #
# object model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ProtocolStep:
    type: str
    document: str
    set: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    max_points: int | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class SelectStep:
    type: str
    from_: str
    table: str
    choose: str
    emit: Mapping[str, str]
    value: str = "value"
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class PlotStep:
    type: str
    plot: str
    from_: str
    table: str
    x: str
    file_path: str
    y: str | None = None
    series: str | None = None
    value: str = "value"
    after: tuple[str, ...] = ()
    description: str | None = None


Step = ProtocolStep | SelectStep | PlotStep


@dataclasses.dataclass(frozen=True)
class WorkflowSaveEntry:
    step: str
    value: str
    file_path: str


@dataclasses.dataclass(frozen=True)
class WorkflowDocument:
    version: str
    steps: Mapping[str, Step]
    save: tuple[WorkflowSaveEntry, ...]
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class LoadedWorkflow:
    """One loaded workflow: the parsed document, the derived schedule, the
    inner protocol loads (or authored-form info for step-dependent ones),
    the canonical form, and the digest."""

    document: WorkflowDocument
    workflow_dir: Path
    order: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    dependencies: Mapping[str, tuple[str, ...]]
    inner: Mapping[str, LoadedProtocol]
    inner_digest_kind: Mapping[str, str]  # "campaign" | "authored"
    inner_digests: Mapping[str, str]
    canonical: Mapping[str, Any]
    digest: str


def is_workflow(raw: Mapping[str, Any]) -> bool:
    """A workflow document is distinguished by its ``steps`` section (§1)."""
    return "steps" in raw


# --------------------------------------------------------------------------- #
# parsing (rules 1–3, shapes)
# --------------------------------------------------------------------------- #


def _check_keys(obj: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise WorkflowError(
                1, f"unknown key {key!r}{suggest(key, allowed)}", path=path
            )


def _need(obj: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    for field in fields:
        if field not in obj:
            raise WorkflowError(1, f"missing required field {field!r}", path=path)


def _str_field(obj: Mapping[str, Any], field: str, path: str) -> str:
    value = obj[field]
    if not isinstance(value, str):
        raise WorkflowError(1, f"{field!r} must be a string", path=path)
    return value


def _after(obj: Mapping[str, Any], path: str) -> tuple[str, ...]:
    value = obj.get("after", [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise WorkflowError(1, "'after' is a list of step names", path=path)
    return tuple(value)


def _parse_step(name: str, raw: Any, path: str) -> Step:
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "a step is an object", path=path)
    step_type = raw.get("type")
    if step_type not in STEP_TYPES:
        raise WorkflowError(
            1,
            f"unknown step type {step_type!r}{suggest(str(step_type), STEP_TYPES)}",
            path=f"{path}.type",
        )
    common = ("type", "after", "description")
    if step_type == "protocol":
        _check_keys(raw, (*common, "document", "set", "max_points"), path)
        _need(raw, ("document",), path)
        overrides = raw.get("set", {})
        if not isinstance(overrides, Mapping) or not all(
            isinstance(k, str) for k in overrides
        ):
            raise WorkflowError(1, "'set' maps dotted paths to values", path=path)
        max_points = raw.get("max_points")
        if max_points is not None and (
            not isinstance(max_points, int) or isinstance(max_points, bool)
        ):
            raise WorkflowError(1, "'max_points' is an integer", path=path)
        return ProtocolStep(
            type="protocol",
            document=_str_field(raw, "document", path),
            set=dict(overrides),
            max_points=max_points,
            after=_after(raw, path),
            description=raw.get("description"),
        )
    if step_type == "select":
        _check_keys(raw, (*common, "from", "table", "choose", "value", "emit"), path)
        _need(raw, ("from", "table", "choose", "emit"), path)
        choose = raw["choose"]
        if choose not in CHOOSE_KINDS:
            raise WorkflowError(
                1,
                f"unknown choose {choose!r}{suggest(str(choose), CHOOSE_KINDS)}",
                path=f"{path}.choose",
            )
        emit = raw["emit"]
        if (
            not isinstance(emit, Mapping)
            or not emit
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in emit.items()
            )
        ):
            raise WorkflowError(
                1, "'emit' is a non-empty {key: column} mapping", path=f"{path}.emit"
            )
        return SelectStep(
            type="select",
            from_=_str_field(raw, "from", path),
            table=_str_field(raw, "table", path),
            choose=str(choose),
            emit=dict(emit),
            value=str(raw.get("value", "value")),
            after=_after(raw, path),
            description=raw.get("description"),
        )
    _check_keys(
        raw,
        (*common, "plot", "from", "table", "x", "y", "series", "value", "file_path"),
        path,
    )
    _need(raw, ("plot", "from", "table", "x", "file_path"), path)
    plot = raw["plot"]
    if plot not in PLOT_KINDS:
        raise WorkflowError(
            1,
            f"unknown plot {plot!r}{suggest(str(plot), PLOT_KINDS)}",
            path=f"{path}.plot",
        )
    if plot == "heatmap" and "y" not in raw:
        raise WorkflowError(1, "a heatmap needs 'y'", path=path)
    return PlotStep(
        type="plot",
        plot=str(plot),
        from_=_str_field(raw, "from", path),
        table=_str_field(raw, "table", path),
        x=_str_field(raw, "x", path),
        y=str(raw["y"]) if "y" in raw else None,
        series=str(raw["series"]) if "series" in raw else None,
        value=str(raw.get("value", "value")),
        file_path=_str_field(raw, "file_path", path),
        after=_after(raw, path),
        description=raw.get("description"),
    )


def parse_workflow(raw: Mapping[str, Any]) -> WorkflowDocument:
    """Strict-parse one workflow document (rules 1–3)."""
    for key in raw:
        if key not in SECTION_ORDER:
            raise WorkflowError(
                1,
                f"unknown section {key!r}{suggest(key, SECTION_ORDER)}",
                path=str(key),
            )
    ranks = {name: i for i, name in enumerate(SECTION_ORDER)}
    order = [ranks[k] for k in raw]
    if order != sorted(order) or (list(raw) and list(raw)[-1] != "save"):
        raise WorkflowError(2, f"sections out of order: {list(raw)} (save last)")
    for section in ("version", "steps", "save"):
        if section not in raw:
            raise WorkflowError(1, f"missing required section {section!r}")
    if raw["version"] != "1":
        raise WorkflowError(1, f"unsupported version {raw['version']!r}")
    steps_raw = raw["steps"]
    if not isinstance(steps_raw, Mapping) or not steps_raw:
        raise WorkflowError(1, "'steps' is a non-empty step table")
    steps: dict[str, Step] = {}
    for name, step_raw in steps_raw.items():
        if not isinstance(name, str) or not _STEP_NAME.match(name):
            raise WorkflowError(
                3,
                f"step name {name!r} is not filesystem-safe ([A-Za-z0-9_-]+) — "
                "step names become run-tree directories (§1)",
            )
        steps[name] = _parse_step(name, step_raw, f"steps.{name}")
    save_raw = raw["save"]
    if not isinstance(save_raw, list) or not save_raw:
        raise WorkflowError(2, "'save' is a non-empty list (and the last section)")
    save: list[WorkflowSaveEntry] = []
    seen_paths: set[str] = set()
    for i, entry_raw in enumerate(save_raw):
        path = f"save[{i}]"
        if not isinstance(entry_raw, Mapping):
            raise WorkflowError(1, "a save entry is an object", path=path)
        _check_keys(entry_raw, ("step", "value", "file_path"), path)
        _need(entry_raw, ("step", "value", "file_path"), path)
        entry = WorkflowSaveEntry(
            step=_str_field(entry_raw, "step", path),
            value=_str_field(entry_raw, "value", path),
            file_path=_str_field(entry_raw, "file_path", path),
        )
        if entry.file_path in seen_paths:
            raise WorkflowError(
                3, f"two save entries write {entry.file_path!r}", path=path
            )
        seen_paths.add(entry.file_path)
        save.append(entry)
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise WorkflowError(1, "description is free text", path="description")
    return WorkflowDocument(
        version="1", steps=steps, save=tuple(save), description=description
    )


# --------------------------------------------------------------------------- #
# the deferring artifact store (§3 + the module docstring)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DeferredArtifacts:
    """Wraps an outer store for workflow-load-time inner validation:
    step-refs answer with representative values, run-tree paths defer
    their file checks to run time."""

    outer: ArtifactStore
    step_names: frozenset[str]
    representatives: Mapping[tuple[str, str], Any]

    def _head(self, ref: str) -> str:
        return ref.split("/", 1)[0]

    def defers(self, file_path: str) -> bool:
        return self._head(file_path) in self.step_names

    def read_value(self, artifact: str, key: str) -> Any:
        if self._head(artifact) in self.step_names:
            try:
                return self.representatives[(artifact, key)]
            except KeyError as err:
                raise KeyError(
                    f"step {artifact!r} emits no key {key!r} — the emit table is "
                    "the contract (workflow §2.3)"
                ) from err
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        if self.defers(file_path):
            return "0" * 64  # placeholder; the run stamps the real digest
        return self.outer.file_digest(file_path)

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        if self.defers(file_path):
            return None  # loader skips the check for deferring stores
        return self.outer.read_identity(file_path)


# --------------------------------------------------------------------------- #
# loading: dependencies, schedule, inner documents, checklist, digest
# --------------------------------------------------------------------------- #


def _walk_step_refs(node: Any, step_names: frozenset[str]) -> set[tuple[str, str]]:
    """Every ``{"artifact": <step…>, "key": …}`` reference into a step."""
    refs: set[tuple[str, str]] = set()
    if isinstance(node, Mapping):
        artifact = node.get("artifact")
        if isinstance(artifact, str) and artifact.split("/", 1)[0] in step_names:
            key = node.get("key")
            if isinstance(key, str):
                refs.add((artifact, key))
        for value in node.values():
            refs |= _walk_step_refs(value, step_names)
    elif isinstance(node, list):
        for item in node:
            refs |= _walk_step_refs(item, step_names)
    return refs


def _walk_run_tree_paths(node: Any, step_names: frozenset[str]) -> set[str]:
    """Every ``file_path`` string under a step's run tree."""
    paths: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            if (
                key == "file_path"
                and isinstance(value, str)
                and value.split("/", 1)[0] in step_names
            ):
                paths.add(value)
            paths |= _walk_run_tree_paths(value, step_names)
    elif isinstance(node, list):
        for item in node:
            paths |= _walk_run_tree_paths(item, step_names)
    return paths


def _authored_digest(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(raw)).hexdigest()


def load_workflow(
    source: Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    workflow_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LoadedWorkflow:
    """Load one workflow document through the full pipeline (§5)."""
    if isinstance(source, Path):
        raw: dict[str, Any] = dict(load_text(source))
        workflow_dir = source.parent if workflow_dir is None else workflow_dir
    else:
        raw = dict(source)
        workflow_dir = Path(".") if workflow_dir is None else workflow_dir
    if overrides:
        raw = apply_overrides(raw, overrides)
    document = parse_workflow(raw)
    steps = document.steps
    step_names = frozenset(steps)

    # ---- rule 4 (references) + derived dependency edges (§3) -------------- #
    overridden_raw: dict[str, dict[str, Any]] = {}
    deps: dict[str, set[str]] = {name: set() for name in steps}
    step_refs: dict[str, set[tuple[str, str]]] = {}
    for name, step in steps.items():
        for other in step.after:
            if other not in steps:
                raise WorkflowError(
                    4, f"'after' names unknown step {other!r}", path=f"steps.{name}"
                )
            deps[name].add(other)
        if isinstance(step, (SelectStep, PlotStep)):
            if step.from_ not in steps:
                raise WorkflowError(
                    4, f"'from' names unknown step {step.from_!r}", path=f"steps.{name}"
                )
            if not isinstance(steps[step.from_], ProtocolStep):
                raise WorkflowError(
                    4,
                    f"'from' must name a protocol step; {step.from_!r} is a "
                    f"{steps[step.from_].type} step",
                    path=f"steps.{name}",
                )
            deps[name].add(step.from_)
            if not step.table.endswith(".parquet"):
                raise WorkflowError(
                    8,
                    f"'table' names a .parquet output, got {step.table!r}",
                    path=f"steps.{name}",
                )
        if isinstance(step, PlotStep) and not step.file_path.endswith((".png", ".pdf")):
            raise WorkflowError(
                8,
                f"a plot file_path ends in .png/.pdf, got {step.file_path!r}",
                path=f"steps.{name}",
            )
        if isinstance(step, ProtocolStep):
            doc_path = (workflow_dir / step.document).resolve()
            if not doc_path.is_file():
                raise WorkflowError(
                    4, f"document {step.document!r} not found", path=f"steps.{name}"
                )
            try:
                inner_raw = apply_overrides(dict(load_text(doc_path)), step.set)
            except ParseError as err:
                raise WorkflowError(
                    9,
                    f"'set' override failed on {step.document!r}: {err}",
                    path=f"steps.{name}",
                ) from err
            overridden_raw[name] = inner_raw
            refs = _walk_step_refs(inner_raw, step_names)
            step_refs[name] = refs
            for artifact, _key in refs:
                deps[name].add(artifact.split("/", 1)[0])
            for run_path in _walk_run_tree_paths(inner_raw, step_names):
                deps[name].add(run_path.split("/", 1)[0])

    # ---- rule 5: acyclicity + the schedule --------------------------------- #
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(node: str, trail: tuple[str, ...]) -> None:
        mark = state.get(node)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join((*trail[trail.index(node) :], node))
            raise WorkflowError(5, f"the step graph has a cycle: {cycle}")
        state[node] = 0
        for dep in sorted(deps[node]):
            visit(dep, trail + (node,))
        state[node] = 1
        order.append(node)

    for name in steps:
        visit(name, ())

    depth: dict[str, int] = {}
    for name in order:
        depth[name] = 1 + max((depth[d] for d in deps[name]), default=-1)
    n_levels = 1 + max(depth.values(), default=0)
    levels = tuple(
        tuple(n for n in order if depth[n] == level) for level in range(n_levels)
    )

    # ---- inner loads (rule 4 tail) + representative substitution ----------- #
    inner: dict[str, LoadedProtocol] = {}
    inner_digests: dict[str, str] = {}
    inner_digest_kind: dict[str, str] = {}
    representatives: dict[tuple[str, str], Any] = {}

    def compute_representatives(select_name: str, select: SelectStep) -> None:
        producer = inner.get(select.from_)
        if producer is None:
            return  # the producer failed earlier; its error already raised
        axis_ids = {axis.id: axis for axis in producer.expansion.axes}
        for key, column in select.emit.items():
            if column in axis_ids:
                representatives[(select_name, key)] = axis_ids[column].values[0]
            elif column == select.value or column == "value":
                representatives[(select_name, key)] = 0.0
            else:
                raise WorkflowError(
                    7,
                    f"emit column {column!r} is neither a sweep axis of "
                    f"{select.from_!r}'s document nor its value column",
                    path=f"steps.{select_name}.emit",
                )

    for name in order:
        step = steps[name]
        if isinstance(step, SelectStep):
            compute_representatives(name, step)
            continue
        if not isinstance(step, ProtocolStep):
            continue
        refs = step_refs[name]
        deferred = bool(refs) or any(
            True for _ in _walk_run_tree_paths(overridden_raw[name], step_names)
        )
        load_env = env
        if deferred:
            load_env = ResolutionEnv(
                datasets=env.datasets,
                artifacts=DeferredArtifacts(
                    outer=env.artifacts,
                    step_names=step_names,
                    representatives=representatives,
                ),
                model_info=env.model_info,
            )
        try:
            loaded = load(
                overridden_raw[name],
                load_env,
                point_cap=step.max_points
                if step.max_points is not None
                else DEFAULT_POINT_CAP,
            )
        except ProtocolError as err:
            raise WorkflowError(
                4,
                f"document {step.document!r} does not load: {err}",
                path=f"steps.{name}",
            ) from err
        inner[name] = loaded
        if deferred:
            inner_digests[name] = _authored_digest(overridden_raw[name])
            inner_digest_kind[name] = "authored"
        else:
            inner_digests[name] = loaded.document_digest
            inner_digest_kind[name] = "campaign"

    # ---- rules 7 + 8: select/plot columns against the producer's axes ------ #
    def check_columns(
        name: str, from_: str, table: str, columns: Sequence[str]
    ) -> None:
        producer = inner[from_]
        outputs = {entry.file_path for entry in producer.document.save}
        if table not in outputs:
            raise WorkflowError(
                4,
                f"{from_!r} saves no {table!r} (has {sorted(outputs)})",
                path=f"steps.{name}.table",
            )
        axis_ids = {axis.id for axis in producer.expansion.axes}
        for column in columns:
            if column not in axis_ids and column != "value":
                raise WorkflowError(
                    7,
                    f"column {column!r} is neither a sweep axis of {from_!r}'s "
                    f"document ({sorted(axis_ids)}) nor 'value'",
                    path=f"steps.{name}",
                )

    for name, step in steps.items():
        if isinstance(step, SelectStep):
            check_columns(name, step.from_, step.table, list(step.emit.values()))
        elif isinstance(step, PlotStep):
            columns = [step.x] + [c for c in (step.y, step.series) if c is not None]
            check_columns(name, step.from_, step.table, columns)

    # ---- rule 4 tail: save entries name real outputs ------------------------ #
    def outputs_of(name: str) -> set[str]:
        step = steps[name]
        if isinstance(step, ProtocolStep):
            return {entry.file_path for entry in inner[name].document.save}
        if isinstance(step, SelectStep):
            return {"values.json"}
        return {step.file_path}

    for i, entry in enumerate(document.save):
        if entry.step not in steps:
            raise WorkflowError(
                4, f"save entry names unknown step {entry.step!r}", path=f"save[{i}]"
            )
        if entry.value not in outputs_of(entry.step):
            raise WorkflowError(
                4,
                f"{entry.step!r} produces no {entry.value!r} "
                f"(has {sorted(outputs_of(entry.step))})",
                path=f"save[{i}]",
            )

    # ---- rule 6: sinks ------------------------------------------------------ #
    consumed: set[str] = {entry.step for entry in document.save}
    for name in steps:
        consumed |= deps[name] - set(steps[name].after)  # `after` is not data flow
    for name in steps:
        if name not in consumed:
            raise WorkflowError(
                6,
                f"step {name!r} is dead: no later step consumes it and no save "
                "entry publishes it (§0)",
                path=f"steps.{name}",
            )

    # ---- canonical form + digest (§7) --------------------------------------- #
    canonical = _canonicalize(document, inner_digests, inner_digest_kind)
    digest = hashlib.sha256(canonical_bytes(canonical)).hexdigest()

    return LoadedWorkflow(
        document=document,
        workflow_dir=workflow_dir,
        order=tuple(order),
        levels=levels,
        dependencies={name: tuple(sorted(deps[name])) for name in steps},
        inner=inner,
        inner_digest_kind=inner_digest_kind,
        inner_digests=inner_digests,
        canonical=canonical,
        digest=digest,
    )


def _canonicalize(
    document: WorkflowDocument,
    inner_digests: Mapping[str, str],
    inner_digest_kind: Mapping[str, str],
) -> dict[str, Any]:
    steps_canonical: dict[str, Any] = {}
    for name, step in document.steps.items():
        entry: dict[str, Any] = {"type": step.type}
        if step.description is not None:
            entry["description"] = step.description
        if step.after:
            entry["after"] = sorted(step.after)
        if isinstance(step, ProtocolStep):
            entry["document"] = step.document
            if step.set:
                entry["set"] = dict(step.set)
            if step.max_points is not None:
                entry["max_points"] = step.max_points
            entry["document_digest"] = inner_digests[name]
            entry["digest_kind"] = inner_digest_kind[name]
        elif isinstance(step, SelectStep):
            entry.update(
                {
                    "from": step.from_,
                    "table": step.table,
                    "choose": step.choose,
                    "value": step.value,
                    "aggregate": "mean",  # v1's one aggregation, materialized
                    "emit": dict(step.emit),
                }
            )
        else:
            entry.update(
                {
                    "plot": step.plot,
                    "from": step.from_,
                    "table": step.table,
                    "x": step.x,
                    "value": step.value,
                    "file_path": step.file_path,
                }
            )
            if step.y is not None:
                entry["y"] = step.y
            if step.series is not None:
                entry["series"] = step.series
        steps_canonical[name] = entry
    out: dict[str, Any] = {"version": document.version}
    if document.description is not None:
        out["description"] = document.description
    out["steps"] = steps_canonical
    out["save"] = [dataclasses.asdict(entry) for entry in document.save]
    return out
