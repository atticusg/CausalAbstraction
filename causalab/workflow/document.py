"""The workflow-protocol document model (docs/workflow_protocol.md, v2).

Backend-free, like the rest of this package: parsing, the workflow load-error
checklist, the derived dependency graph and schedule, and the canonical form +
digest. Executing a loaded workflow is :mod:`causalab.workflow.runner`'s job.

**Two step types, one wiring mechanism.** ``protocol`` stays declarative
because that is where the load-time bite lives — inner-document validation,
sweep expansion, capability routing, shard dispatch. ``script`` is inputs → one
Python script → declared outputs, wide enough that a pipeline never has to
leave the record. Everything either one consumes is spelled as a *locator plus
an optional selector* (§3), so the dependency graph falls out of one place
instead of three.

Three load-time subtleties the spec commits to:

* **Step-dependent inner documents validate against declared
  representatives.** A protocol step whose document references another step's
  outputs (``{"artifact": "best", …}``, or a ``file_path`` under a step's run
  tree) cannot resolve those values before the run. The producing script step
  declares them — ``outputs.<slot>.keys`` maps each emitted name to a
  representative *value* (§2.3) — and the loader substitutes those, so the
  consumer type-checks honestly. Run-tree ``file_path`` loads defer their
  existence/identity checks to run time (the deferring store advertises
  :meth:`DeferredArtifacts.defers`).
* **A script is hashed, never imported.** ``validate``/``digest`` must not pull
  torch in through a user script, so load-time checking is the file existing,
  ``ast.parse`` succeeding, a module-level ``def main`` being present, and the
  bytes being hashed. Hashing needs no import, which is what lets the hash sit
  in the digest and keep ``--resume`` correct (§7).
* **Digests split by dependency** (§7): a protocol step with no in-run
  references stamps its document's full campaign digest; a step-dependent
  document stamps the digest of its overridden authored form, and the fully
  resolved digests land in the run manifest.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.bundles import (
    entry_key,
    entry_selection,
    select_entry,
    selector_slot,
)
from causalab.protocol.canonical import canonical_bytes
from causalab.protocol.errors import (
    ParseError,
    ProtocolError,
    ValidationError,
    suggest,
)
from causalab.protocol.loader import (
    LoadedProtocol,
    apply_overrides,
    flatten,
    load,
    load_text,
)
from causalab.protocol.resolve import ArtifactStore, ResolutionEnv
from causalab.protocol.schema import FEATURIZER_SLOTS
from causalab.protocol.sweep import (
    DEFAULT_POINT_CAP,
    coordinate_label,
    short_coords,
)
from causalab.protocol.tables import TABLE_SUFFIX

__all__ = [
    "COLUMN_DTYPES",
    "DeferredArtifacts",
    "LoadedWorkflow",
    "OutputDecl",
    "ProtocolStep",
    "Reference",
    "STEP_TYPES",
    "ScriptStep",
    "Step",
    "WorkflowDocument",
    "WorkflowError",
    "is_workflow",
    "load_workflow",
    "parse_workflow",
]

STEP_TYPES: tuple[str, ...] = ("intervention_protocol", "script")

#: Column dtypes a table output may declare. Deliberately narrow: the types
#: that survive a JSON round-trip and a strict re-parse by a consuming step.
COLUMN_DTYPES: tuple[str, ...] = ("int64", "float64", "bool", "string")

#: The two *record* formats: structured data and dense numerics (§2.5).
RECORD_SUFFIXES: tuple[str, ...] = (TABLE_SUFFIX, ".safetensors")

#: Visualization formats. These carry no record — a figure is a rendering of an
#: artifact rather than one itself — so they are legal outputs but may declare
#: no `columns`/`keys`. `png` is preferred over `pdf` unless a document asks for
#: pdf explicitly (``causalab.io.plots.figure_format``).
VISUALIZATION_SUFFIXES: tuple[str, ...] = (".png", ".pdf", ".html")

OUTPUT_SUFFIXES: tuple[str, ...] = RECORD_SUFFIXES + VISUALIZATION_SUFFIXES

_STEP_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")

#: Top-level sections in mandatory order (§1). No `save`: everything a step
#: declares is published where it lands (§0).
SECTION_ORDER: tuple[str, ...] = ("version", "description", "output_dir", "steps")

#: The highest checklist rule number (§5).
MAX_RULE = 11


class WorkflowError(ProtocolError):
    """A workflow document violates checklist rule ``rule``
    (docs/workflow_protocol.md §5); code ``W<rule>``."""

    def __init__(self, rule: int, message: str, *, path: str | None = None) -> None:
        if not 1 <= rule <= MAX_RULE:
            raise AssertionError(f"workflow checklist rule out of range: {rule}")
        self.rule = rule
        super().__init__(f"W{rule}", message, path=path)


# --------------------------------------------------------------------------- #
# object model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Reference:
    """One resolved-at-run-time input: a locator plus an optional selector (§3).

    Exactly one locator is set. ``step``+``file`` names a file in the run tree;
    ``path`` names a file on disk (absolute if it starts with ``/``, otherwise
    relative to the repo root). ``key`` selects a scalar out of a JSON values
    object; ``entry`` selects one tensor of a safetensors bundle. At most one
    selector, and each requires the matching format."""

    step: str | None = None
    file: str | None = None
    path: str | None = None
    key: str | None = None
    entry: Mapping[str, Any] | None = None
    #: which named tensor of a multi-slot bundle, when ``entry`` alone is ambiguous
    slot: str | None = None

    @property
    def target(self) -> str:
        """What this reference names, for an error message."""
        return f"{self.step}/{self.file}" if self.step is not None else str(self.path)

    @property
    def suffix(self) -> str:
        name = self.file if self.step is not None else self.path
        return Path(str(name)).suffix


@dataclasses.dataclass(frozen=True)
class OutputDecl:
    """One declared output: a filename, plus at most one shape promise.

    ``columns`` says "an array of row objects" and maps column name to a
    :data:`COLUMN_DTYPES` entry. ``keys`` says "one object mapping these names
    to values" and maps each name to a **representative value** — not a type,
    because a step-dependent inner document validates against it and a position
    spec has to type-check as a position spec (§2.3)."""

    file: str
    columns: Mapping[str, str] | None = None
    keys: Mapping[str, Any] | None = None

    @property
    def suffix(self) -> str:
        return Path(self.file).suffix


@dataclasses.dataclass(frozen=True)
class ProtocolStep:
    type: str
    document: str
    set: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    max_points: int | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ScriptStep:
    type: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, OutputDecl]
    #: exactly one of these is set — the script locator (§2.3)
    module: str | None = None
    path: str | None = None
    #: sha256 of the script's bytes — in the digest, so ``--resume`` is correct
    script_sha256: str = ""
    runtime: Mapping[str, Any] | None = None
    is_deterministic: bool = True
    after: tuple[str, ...] = ()
    description: str | None = None

    @property
    def script(self) -> str:
        """What the document said, for an error message or a manifest."""
        return self.module if self.module is not None else str(self.path)


Step = ProtocolStep | ScriptStep


@dataclasses.dataclass(frozen=True)
class WorkflowDocument:
    version: str
    output_dir: str
    steps: Mapping[str, Step]
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class LoadedWorkflow:
    """One loaded workflow: the parsed document, the derived schedule, the
    inner protocol loads (or authored-form info for step-dependent ones), the
    canonical form, and the digest."""

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
    #: ``{step: digest of its canonical entry}`` — the provenance unit a tensor
    #: a script step writes is stamped with (§7)
    step_digests: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: absolute `path` references, which load cannot existence-check (rule 4)
    unchecked_paths: tuple[str, ...] = ()

    @property
    def nondeterministic(self) -> tuple[str, ...]:
        """Steps that declared themselves not replayable (§7)."""
        return tuple(
            name
            for name, step in self.document.steps.items()
            if isinstance(step, ScriptStep) and not step.is_deterministic
        )


def is_workflow(raw: Mapping[str, Any]) -> bool:
    """A workflow document is distinguished by its ``steps`` section (§1)."""
    return "steps" in raw


# --------------------------------------------------------------------------- #
# parsing
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
            raise WorkflowError(1, f"missing required key {field!r}", path=path)


def _str_field(obj: Mapping[str, Any], field: str, path: str) -> str:
    value = obj[field]
    if not isinstance(value, str) or not value:
        raise WorkflowError(
            1, f"{field!r} is a non-empty string, got {value!r}", path=path
        )
    return value


def _after(obj: Mapping[str, Any], path: str) -> tuple[str, ...]:
    raw = obj.get("after", ())
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise WorkflowError(1, "'after' is a list of step names", path=path)
    return tuple(str(name) for name in raw)


def _bool_field(obj: Mapping[str, Any], field: str, default: bool, path: str) -> bool:
    if field not in obj:
        return default
    value = obj[field]
    if not isinstance(value, bool):
        raise WorkflowError(11, f"{field!r} is a boolean, got {value!r}", path=path)
    return value


def _contained(value: str, rule: int, path: str) -> None:
    """A relative path that cannot escape its directory (rules 2, 6, 7)."""
    if not isinstance(value, str) or not value:
        raise WorkflowError(rule, f"expected a path, got {value!r}", path=path)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WorkflowError(
            rule,
            f"{value!r} must be relative and stay inside its directory",
            path=path,
        )


def _parse_reference(value: Mapping[str, Any], path: str) -> Reference | None:
    """One ``inputs`` value as a reference, or ``None`` if it is a literal.

    References are recognized **only at the top level** of an inputs entry, so
    a nested object is always a literal and the loader never guesses (§3)."""
    locators = [key for key in ("step", "path") if key in value]
    if not locators:
        return None
    if len(locators) > 1:
        raise WorkflowError(
            1,
            "a reference has one locator: 'step'+'file', or 'path'",
            path=path,
        )
    allowed = ("step", "file", "path", "key", "entry", "slot")
    _check_keys(value, allowed, path)
    if "step" in value:
        _need(value, ("file",), path)
        ref = Reference(
            step=_str_field(value, "step", path),
            file=_str_field(value, "file", path),
        )
        _contained(str(ref.file), 4, path)
    else:
        ref = Reference(path=_str_field(value, "path", path))
    selectors = [key for key in ("key", "entry") if key in value]
    if len(selectors) > 1:
        raise WorkflowError(
            4,
            "a reference carries at most one selector: 'key' (JSON) or "
            "'entry' (safetensors)",
            path=path,
        )
    if "key" in value:
        ref = dataclasses.replace(ref, key=_str_field(value, "key", path))
    if "entry" in value:
        entry = value["entry"]
        if not isinstance(entry, Mapping):
            raise WorkflowError(1, "'entry' maps coordinate names to values", path=path)
        ref = dataclasses.replace(ref, entry=dict(entry))
    if "slot" in value:
        ref = dataclasses.replace(ref, slot=_str_field(value, "slot", path))
    # rule 4: a selector must match its locator's format, decidable from the
    # filename alone — which is exactly what having only two formats buys
    if ref.key is not None and ref.suffix != TABLE_SUFFIX:
        raise WorkflowError(
            4,
            f"'key' reads a {TABLE_SUFFIX} file; {ref.target!r} is "
            f"{ref.suffix or 'extensionless'}",
            path=path,
        )
    if (ref.entry is not None or ref.slot is not None) and ref.suffix != ".safetensors":
        raise WorkflowError(
            4,
            f"'entry'/'slot' reads a .safetensors bundle; {ref.target!r} is "
            f"{ref.suffix or 'extensionless'}",
            path=path,
        )
    return ref


def _parse_script_locator(raw: Any, path: str) -> tuple[str | None, str | None]:
    """``script`` as a locator: ``{"module": …}`` or ``{"path": …}`` (§2.3)."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(
            6,
            '\'script\' is a locator: {"module": "causalab.analysis.fit_pca"} '
            'or {"path": "scripts/probe.py"}',
            path=path,
        )
    _check_keys(raw, ("module", "path"), path)
    present = [key for key in ("module", "path") if key in raw]
    if len(present) != 1:
        raise WorkflowError(
            6,
            "'script' names exactly one of 'module' or 'path'",
            path=path,
        )
    if "module" in raw:
        module = _str_field(raw, "module", path)
        if not all(part.isidentifier() for part in module.split(".")):
            raise WorkflowError(
                6, f"script module {module!r} is not a dotted identifier", path=path
            )
        return module, None
    return None, _str_field(raw, "path", path)


def _parse_output(slot: str, raw: Any, path: str) -> OutputDecl:
    if isinstance(raw, str):
        decl = OutputDecl(file=raw)
    elif isinstance(raw, Mapping):
        _check_keys(raw, ("file", "columns", "keys"), path)
        _need(raw, ("file",), path)
        columns = raw.get("columns")
        keys = raw.get("keys")
        if columns is not None and keys is not None:
            raise WorkflowError(
                7,
                "'columns' and 'keys' are mutually exclusive: the first says "
                "an array of row objects, the second one values object",
                path=path,
            )
        if columns is not None:
            if not isinstance(columns, Mapping) or not columns:
                raise WorkflowError(
                    7, "'columns' maps column names to dtypes", path=path
                )
            for column, dtype in columns.items():
                if dtype not in COLUMN_DTYPES:
                    raise WorkflowError(
                        7,
                        f"column {column!r} has unknown dtype {dtype!r}"
                        f"{suggest(str(dtype), COLUMN_DTYPES)}",
                        path=path,
                    )
        if keys is not None and (not isinstance(keys, Mapping) or not keys):
            raise WorkflowError(
                7,
                "'keys' maps emitted names to a representative value each",
                path=path,
            )
        decl = OutputDecl(
            file=_str_field(raw, "file", path),
            columns=dict(columns) if columns is not None else None,
            keys=dict(keys) if keys is not None else None,
        )
    else:
        raise WorkflowError(
            1, f"output {slot!r} is a filename or an object with 'file'", path=path
        )
    _contained(decl.file, 7, path)
    if decl.suffix not in OUTPUT_SUFFIXES:
        raise WorkflowError(
            7,
            f"output {slot!r} is {decl.file!r} — every output ends in "
            f"{' or '.join(OUTPUT_SUFFIXES)} (§2.5)",
            path=path,
        )
    if decl.suffix != TABLE_SUFFIX and (
        decl.columns is not None or decl.keys is not None
    ):
        raise WorkflowError(
            7,
            f"output {slot!r} declares columns/keys but is not {TABLE_SUFFIX} — "
            "a shape promise only means something for a structured file",
            path=path,
        )
    return decl


def _parse_runtime(raw: Any, path: str) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkflowError(10, "'runtime' is an object", path=path)
    _check_keys(raw, ("isolate", "deps", "env"), path)
    isolate = raw.get("isolate", False)
    if not isinstance(isolate, bool):
        raise WorkflowError(10, "'runtime.isolate' is a boolean", path=path)
    deps = raw.get("deps", ())
    if isinstance(deps, str) or not isinstance(deps, (list, tuple)):
        raise WorkflowError(10, "'runtime.deps' is a list of requirements", path=path)
    env = raw.get("env", ())
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise WorkflowError(
            10, "'runtime.env' is a list of variable NAMES, never values", path=path
        )
    if isolate and not deps:
        raise WorkflowError(
            10,
            "an isolated step declares its 'deps' — otherwise isolation buys "
            "nothing and the subprocess would import a different environment "
            "than the document says",
            path=path,
        )
    out: dict[str, Any] = {"isolate": isolate, "deps": [str(d) for d in deps]}
    if env:
        out["env"] = [str(name) for name in env]
    return out


def _parse_step(name: str, raw: Any, path: str) -> Step:
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, f"step {name!r} is an object", path=path)
    _need(raw, ("type",), path)
    kind = raw["type"]
    if kind not in STEP_TYPES:
        raise WorkflowError(
            1,
            f"unknown step type {kind!r}{suggest(str(kind), STEP_TYPES)}",
            path=path,
        )
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise WorkflowError(1, "'description' is free text", path=path)
    if kind == "intervention_protocol":
        _check_keys(
            raw,
            ("type", "document", "set", "max_points", "after", "description"),
            path,
        )
        _need(raw, ("document",), path)
        overrides = raw.get("set", {})
        if not isinstance(overrides, Mapping):
            raise WorkflowError(1, "'set' maps dotted paths to values", path=path)
        max_points = raw.get("max_points")
        if max_points is not None and (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or max_points < 1
        ):
            raise WorkflowError(1, "'max_points' is a positive integer", path=path)
        return ProtocolStep(
            type="intervention_protocol",
            document=_str_field(raw, "document", path),
            set=dict(overrides),
            max_points=max_points,
            after=_after(raw, path),
            description=description,
        )

    _check_keys(
        raw,
        (
            "type",
            "script",
            "inputs",
            "outputs",
            "runtime",
            "is_deterministic",
            "after",
            "description",
        ),
        path,
    )
    _need(raw, ("script", "inputs", "outputs"), path)
    module, script_path = _parse_script_locator(raw["script"], f"{path}.script")
    inputs = raw["inputs"]
    if not isinstance(inputs, Mapping):
        raise WorkflowError(1, "'inputs' maps names to values", path=path)
    outputs = raw["outputs"]
    if not isinstance(outputs, Mapping) or not outputs:
        raise WorkflowError(
            7, "'outputs' is a non-empty map of slot to file", path=path
        )
    parsed_outputs = {
        slot: _parse_output(slot, decl, f"{path}.outputs.{slot}")
        for slot, decl in outputs.items()
    }
    files = [decl.file for decl in parsed_outputs.values()]
    duplicate = next((f for f in files if files.count(f) > 1), None)
    if duplicate is not None:
        raise WorkflowError(
            7,
            f"two outputs both write {duplicate!r} — one file, one slot",
            path=f"{path}.outputs",
        )
    parsed_inputs = {
        key: (
            _parse_reference(value, f"{path}.inputs.{key}") or value
            if isinstance(value, Mapping)
            else value
        )
        for key, value in inputs.items()
    }
    return ScriptStep(
        type="script",
        module=module,
        path=script_path,
        inputs=parsed_inputs,
        outputs=parsed_outputs,
        runtime=_parse_runtime(raw.get("runtime"), f"{path}.runtime"),
        is_deterministic=_bool_field(raw, "is_deterministic", True, path),
        after=_after(raw, path),
        description=description,
    )


def parse_workflow(raw: Mapping[str, Any]) -> WorkflowDocument:
    """Parse and structurally validate one workflow document (rules 1-3)."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "a workflow document is an object")
    _check_keys(raw, SECTION_ORDER, "")
    _need(raw, ("version", "output_dir", "steps"), "")

    present = [key for key in raw if key in SECTION_ORDER]
    expected = [key for key in SECTION_ORDER if key in raw]
    if present != expected:
        raise WorkflowError(
            2,
            f"sections must appear in the order {expected}, got {present}",
        )
    version = _str_field(raw, "version", "")
    if version != "1":
        raise WorkflowError(1, f"unsupported version {version!r} (expected '1')")

    output_dir = _str_field(raw, "output_dir", "")
    if not _SEGMENT.match(output_dir) or output_dir in {".", ".."}:
        raise WorkflowError(
            2,
            f"'output_dir' is one filesystem-safe path segment, got "
            f"{output_dir!r} — the CLI supplies the root it sits under (§1.1)",
        )

    steps_raw = raw["steps"]
    if not isinstance(steps_raw, Mapping) or not steps_raw:
        raise WorkflowError(1, "'steps' is a non-empty object", path="steps")
    steps: dict[str, Step] = {}
    for name, step_raw in steps_raw.items():
        if not _STEP_NAME.match(str(name)):
            raise WorkflowError(
                3,
                f"step name {name!r} is not filesystem-safe ([A-Za-z0-9_-]+)",
                path="steps",
            )
        if name == "workflow.json" or name.startswith("_"):
            raise WorkflowError(
                3,
                f"step name {name!r} is reserved — step directories sit beside "
                "the run manifest",
                path="steps",
            )
        steps[str(name)] = _parse_step(str(name), step_raw, f"steps.{name}")

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise WorkflowError(1, "'description' is free text")
    return WorkflowDocument(
        version=version,
        output_dir=output_dir,
        steps=steps,
        description=description,
    )


# --------------------------------------------------------------------------- #
# script resolution — hashed, never imported (§4.2)
# --------------------------------------------------------------------------- #


def resolve_script(step: ScriptStep, workflow_dir: Path, path: str) -> Path:
    """The file a step's ``script`` locator names — found, never imported (rule 6).

    Two locators, the same shape an ``inputs`` reference uses (§3):

    * ``{"module": "causalab.analysis.fit_pca"}`` — an importable module, found
      with :func:`importlib.util.find_spec`, which resolves a dotted name to a
      file **without executing it**. That is what lets a shipped script live
      wherever it belongs by subject (``causalab.analysis``,
      ``causalab.io.plots``, ``causalab.workflow.scripts``) instead of in one
      flat namespace with a search order.
    * ``{"path": "scripts/probe.py"}`` — a file beside the workflow document,
      contained, no parent escapes.

    v1 spelled a shipped script ``causalab:<name>``. That needed a registry —
    exactly the thing this layer removes — and it hid *which* code ran behind a
    lookup. A module path says it."""
    import importlib.util

    if step.module is not None:
        try:
            spec = importlib.util.find_spec(step.module)
        except (ImportError, ValueError) as err:
            # a missing PARENT package raises rather than returning None
            raise WorkflowError(
                6, f"script module {step.module!r} not found: {err}", path=path
            ) from err
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise WorkflowError(
                6,
                f"script module {step.module!r} does not resolve to a Python file",
                path=path,
            )
        return Path(spec.origin)
    _contained(str(step.path), 6, path)
    target = (workflow_dir / str(step.path)).resolve()
    if not target.is_file():
        raise WorkflowError(6, f"script {step.path!r} not found", path=path)
    return target


def check_script(target: Path, step: ScriptStep, path: str) -> str:
    """Rule 6: the script parses and declares ``main``. Returns its sha256.

    Deliberately shallow, and deliberately not an import: ``validate`` and
    ``digest`` must stay runnable without torch, and importing a user script
    would pull in whatever it links against (§4.2). Hashing needs no import,
    which is what lets the hash reach the digest."""
    source = target.read_bytes()
    try:
        tree = ast.parse(source, filename=str(target))
    except SyntaxError as err:
        raise WorkflowError(
            6, f"script {step.script!r} does not parse: {err}", path=path
        ) from err
    has_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in tree.body
    )
    if not has_main:
        raise WorkflowError(
            6,
            f"script {step.script!r} declares no module-level 'main' — the "
            "runner calls main(inputs, outputs) (§4)",
            path=path,
        )
    return hashlib.sha256(source).hexdigest()


# --------------------------------------------------------------------------- #
# load-time artifact store for step-dependent inner documents
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DeferredArtifacts:
    """Wraps an outer store for workflow-load-time inner validation:
    step-refs answer with declared representatives, and every ``file_path``
    check defers to run time — a representative-substituted document may carry
    representative values in exactly the fields an ArtifactIdentity is checked
    against (the site record), so a load-time identity comparison would be
    against the wrong document."""

    outer: ArtifactStore
    step_names: frozenset[str]
    representatives: Mapping[tuple[str, str], Any]

    def _head(self, ref: str) -> str:
        return ref.split("/", 1)[0]

    def defers(self, file_path: str) -> bool:
        del file_path
        return True  # all file checks re-run with real values at run time

    def read_value(self, artifact: str, key: str) -> Any:
        if self._head(artifact) in self.step_names:
            try:
                return self.representatives[(self._head(artifact), key)]
            except KeyError as err:
                raise KeyError(
                    f"step {artifact!r} declares no emitted key {key!r} — a "
                    "step whose values another document reads declares them "
                    "in outputs.<slot>.keys (workflow §2.3)"
                ) from err
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        del file_path
        return "0" * 64  # placeholder; the digest of a deferred doc is discarded

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        del file_path
        return None  # loader skips the check for deferring stores


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
    """Every ``file_path`` the inner document LOADS from a step's run tree.

    The IM spec has exactly two file_path *load* sites — featurizer specs and
    params entries (§2.5, §2.6); an inner document's ``save`` section also
    carries ``file_path`` keys, but those are outputs, never loads — walking
    them would fabricate dependency edges (and even self-cycles) out of a
    step's own products."""
    paths: set[str] = set()
    if not isinstance(node, Mapping):
        return paths
    for section in ("featurizers", "params"):
        table = node.get(section)
        if not isinstance(table, Mapping):
            continue
        for entry in table.values():
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("file_path")
            if isinstance(value, str) and value.split("/", 1)[0] in step_names:
                paths.add(value)
    return paths


def _authored_digest(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(raw)).hexdigest()


def _repo_root() -> Path:
    """The root a relative ``path`` reference resolves against (§3)."""
    import causalab

    return Path(causalab.__file__).resolve().parent.parent


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

    # ---- rule 6: scripts resolve, parse, declare main, and hash ------------ #
    script_hashes: dict[str, str] = {}
    for name, step in steps.items():
        if isinstance(step, ScriptStep):
            target = resolve_script(step, workflow_dir, f"steps.{name}.script")
            script_hashes[name] = check_script(target, step, f"steps.{name}.script")
    steps = {
        name: (
            dataclasses.replace(step, script_sha256=script_hashes[name])
            if isinstance(step, ScriptStep)
            else step
        )
        for name, step in steps.items()
    }
    document = dataclasses.replace(document, steps=steps)

    # ---- rule 4 (references) + derived dependency edges (§3) --------------- #
    overridden_raw: dict[str, dict[str, Any]] = {}
    inner_dirs: dict[str, Path] = {}
    #: per step, the method its document was composed from (§1.1) — the
    #: flatten happens here, so the provenance is attached back below
    inner_methods: dict[str, tuple[str | None, str | None]] = {}
    deps: dict[str, set[str]] = {name: set() for name in steps}
    step_refs: dict[str, set[tuple[str, str]]] = {}
    run_tree_loads: dict[str, set[str]] = {name: set() for name in steps}
    unchecked_paths: list[str] = []

    for name, step in steps.items():
        for other in step.after:
            if other not in steps:
                raise WorkflowError(
                    4, f"'after' names unknown step {other!r}", path=f"steps.{name}"
                )
            deps[name].add(other)

        if isinstance(step, ScriptStep):
            for slot, value in step.inputs.items():
                if not isinstance(value, Reference):
                    continue
                where = f"steps.{name}.inputs.{slot}"
                if value.step is not None:
                    if value.step not in steps:
                        raise WorkflowError(
                            4,
                            f"input {slot!r} names unknown step {value.step!r}"
                            f"{suggest(value.step, sorted(steps))}",
                            path=where,
                        )
                    deps[name].add(value.step)
                    continue
                target = str(value.path)
                if target.startswith("/"):
                    # rule 4: an absolute path is NOT existence-checked —
                    # validation and execution routinely run on different
                    # hosts, so checking here would fail a path that is
                    # perfectly good where the run happens
                    unchecked_paths.append(f"{name}.{slot}: {target}")
                    continue
                if not (_repo_root() / target).is_file():
                    raise WorkflowError(
                        4,
                        f"input {slot!r} names {target!r}, which does not exist "
                        "under the repo root (an absolute path would defer to "
                        "run time; a repo-relative one must be here now)",
                        path=where,
                    )
            continue

        doc_path = (workflow_dir / step.document).resolve()
        if not doc_path.is_file():
            raise WorkflowError(
                4, f"document {step.document!r} not found", path=f"steps.{name}"
            )
        try:
            # flatten a split inner document first (§1.1), so a step's `set`
            # paths are the composition's — one vocabulary for both authoring
            # forms
            flat, method_hash, method_ref = flatten(
                dict(load_text(doc_path)), base_dir=doc_path.parent
            )
            inner_methods[name] = (method_hash, method_ref)
            inner_raw = apply_overrides(flat, step.set)
        except ParseError as err:
            raise WorkflowError(
                8,
                f"'set' override failed on {step.document!r}: {err}",
                path=f"steps.{name}",
            ) from err
        overridden_raw[name] = inner_raw
        inner_dirs[name] = doc_path.parent
        refs = _walk_step_refs(inner_raw, step_names)
        step_refs[name] = refs
        run_tree_loads[name] = _walk_run_tree_paths(inner_raw, step_names)
        for artifact, key in refs:
            producer = artifact.split("/", 1)[0]
            deps[name].add(producer)
            # rule 4: a values reference must name a step that declares the
            # key. With `select` a script, the declaration is the contract —
            # and it is what the representative substitution below reads.
            _check_declares_key(steps[producer], producer, key, path=f"steps.{name}")
        for run_path in run_tree_loads[name]:
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

    # ---- representatives, then the inner loads (rule 4 tail, rule 8) ------- #
    representatives: dict[tuple[str, str], Any] = {}
    for name, step in steps.items():
        if not isinstance(step, ScriptStep):
            continue
        for decl in step.outputs.values():
            for key, value in (decl.keys or {}).items():
                representatives[(name, key)] = value

    inner: dict[str, LoadedProtocol] = {}
    inner_digests: dict[str, str] = {}
    inner_digest_kind: dict[str, str] = {}

    for name in order:
        step = steps[name]
        if not isinstance(step, ProtocolStep):
            continue
        deferred = bool(step_refs[name]) or bool(run_tree_loads[name])
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
                base_dir=inner_dirs.get(name),
                point_cap=step.max_points
                if step.max_points is not None
                else DEFAULT_POINT_CAP,
            )
        except ProtocolError as err:
            raise WorkflowError(
                8,
                f"document {step.document!r} does not load: {err}",
                path=f"steps.{name}",
            ) from err
        method_hash, method_ref = inner_methods.get(name, (None, None))
        if method_hash is not None:
            loaded = dataclasses.replace(
                loaded, method_digest=method_hash, method_ref=method_ref
            )
        inner[name] = loaded
        if deferred:
            inner_digests[name] = _authored_digest(overridden_raw[name])
            inner_digest_kind[name] = "authored"
        else:
            inner_digests[name] = loaded.document_digest
            inner_digest_kind[name] = "campaign"

    # ---- rule 4: every `file` names an output its producer really writes --- #
    def outputs_of(name: str) -> set[str]:
        step = steps[name]
        if isinstance(step, ProtocolStep):
            return {entry.file_path for entry in inner[name].document.save}
        return {decl.file for decl in step.outputs.values()}

    for name, step in steps.items():
        if not isinstance(step, ScriptStep):
            continue
        for slot, value in step.inputs.items():
            if not isinstance(value, Reference) or value.step is None:
                continue
            produced = outputs_of(value.step)
            if value.file not in produced:
                raise WorkflowError(
                    4,
                    f"input {slot!r} reads {value.target}, but "
                    f"{value.step!r} writes no {value.file!r} "
                    f"(has {sorted(produced)})",
                    path=f"steps.{name}.inputs.{slot}",
                )
            if value.key is not None:
                _check_declares_key(
                    steps[value.step],
                    value.step,
                    value.key,
                    path=f"steps.{name}.inputs.{slot}",
                    file=value.file,
                )
            if value.entry is not None or value.slot is not None:
                _check_script_entry(
                    name, slot, value, steps=steps, inner=inner, outputs_of=outputs_of
                )

    # ---- rule 9: run-tree loads inside protocol documents ------------------ #
    for name, paths in run_tree_loads.items():
        for run_path in sorted(paths):
            producer, _, rest = run_path.partition("/")
            produced = outputs_of(producer)
            if rest not in produced:
                raise WorkflowError(
                    4,
                    f"{name!r} loads {run_path!r}, but {producer!r} writes no "
                    f"{rest!r} (has {sorted(produced)}) — a run-tree file_path "
                    "must name a file the step actually saves",
                    path=f"steps.{name}",
                )
            if isinstance(steps[producer], ProtocolStep):
                _check_entry_selection(
                    consumer=inner[name],
                    producer=inner[producer],
                    run_path=run_path,
                    rest=rest,
                    step=name,
                )

    # ---- canonical form + digest (§7) ------------------------------------- #
    canonical = _canonicalize(document, inner_digests)
    digest = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    # a script step's provenance unit: the digest of its own canonical entry,
    # which is a pure function of script hash + inputs + outputs + runtime. It
    # is what a tensor it writes is stamped `produced_by` — the analogue of a
    # protocol point's digest.
    step_digests = {
        name: hashlib.sha256(canonical_bytes(canonical["steps"][name])).hexdigest()
        for name, step in steps.items()
        if isinstance(step, ScriptStep)
    }

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
        step_digests=step_digests,
        unchecked_paths=tuple(sorted(unchecked_paths)),
    )


def _check_declares_key(
    producer: Step,
    producer_name: str,
    key: str,
    *,
    path: str,
    file: str | None = None,
) -> None:
    """Rule 4: a ``key`` reference names a step that declares it.

    v1 could only ask this of a `select` step, whose `emit` table the loader
    read. v2 asks it of *any* step, because outputs are declared — which is why
    the check is stronger than the rule it replaces."""
    if not isinstance(producer, ScriptStep):
        raise WorkflowError(
            4,
            f"values reference reads a key of {producer_name!r}, which is a "
            "protocol step — a protocol document's outputs are tables and "
            "tensors, not a values object",
            path=path,
        )
    declared: dict[str, Any] = {}
    for decl in producer.outputs.values():
        if file is not None and decl.file != file:
            continue
        declared.update(decl.keys or {})
    if key not in declared:
        where = f"{producer_name}/{file}" if file else producer_name
        raise WorkflowError(
            4,
            f"{where} declares no emitted key {key!r} "
            f"({sorted(declared) or 'no keys declared'}) — a step whose values "
            "another step reads declares them in outputs.<slot>.keys (§2.3)"
            f"{suggest(key, sorted(declared))}",
            path=path,
        )


# --------------------------------------------------------------------------- #
# bundle-entry checking (rule 9)
# --------------------------------------------------------------------------- #


def _bundle_entries(
    producer: LoadedProtocol, file_path: str
) -> dict[str, dict[str, Any]] | None:
    """The tensor keys a producing document will write into ``file_path``, with
    their coordinates — derivable at load because sweeps expand
    deterministically (§3), which is what lets a wrong selection fail here
    instead of after the producing step has run."""
    entries: dict[str, dict[str, Any]] = {}
    for save_entry in producer.document.save:
        if save_entry.file_path != file_path:
            continue
        if save_entry.value in producer.document.metrics:
            return None  # a metric table, not a tensor bundle
        if save_entry.value in producer.document.reads:
            slots: tuple[str, ...] = (save_entry.value,)
        else:
            spec = producer.document.featurizers.get(save_entry.value)
            kind = spec.kind if spec is not None and isinstance(spec.kind, str) else ""
            slots = FEATURIZER_SLOTS.get(kind, ())
        if not slots:
            return None
        for point in producer.expansion.points:
            short = short_coords(point.coords, entry=save_entry.value)
            label = coordinate_label(point.coords, entry=save_entry.value)
            for slot in slots:
                entries[entry_key(slot, label)] = {"slot": slot, "coords": short}
    return entries or None


def _sole_bundle_slot(entries: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The one slot a bundle holds, or ``None`` when it holds several."""
    slots = {str(entry.get("slot")) for entry in entries.values()}
    return slots.pop() if len(slots) == 1 else None


def _check_script_entry(
    name: str,
    slot: str,
    ref: Reference,
    *,
    steps: Mapping[str, Step],
    inner: Mapping[str, LoadedProtocol],
    outputs_of: Any,
) -> None:
    """Rule 9 for a script input: the entry it selects is one the producer will
    write. Checkable only against a *protocol* producer, whose expansion is
    deterministic at load; against a script producer it is a run-time check."""
    producer = steps[str(ref.step)]
    if not isinstance(producer, ProtocolStep):
        return
    entries = _bundle_entries(inner[str(ref.step)], str(ref.file))
    if entries is None:
        return
    bundle_slot = ref.slot or _sole_bundle_slot(entries)
    if bundle_slot is None:
        held = sorted({str(e.get("slot")) for e in entries.values()})
        raise WorkflowError(
            9,
            f"input {slot!r} reads a bundle holding several slots ({held}) — "
            "name one with 'slot'",
            path=f"steps.{name}.inputs.{slot}",
        )
    try:
        select_entry(
            entries.keys(),
            bundle_slot,
            ref.entry,
            what=f"step {name!r}: input {slot!r} reads {ref.target}",
            coords_by_key=entries,
            implicit=False,
        )
    except ValidationError as err:
        raise WorkflowError(9, str(err), path=f"steps.{name}.inputs.{slot}") from err


def _check_entry_selection(
    *,
    consumer: LoadedProtocol,
    producer: LoadedProtocol,
    run_path: str,
    rest: str,
    step: str,
) -> None:
    """Rule 9, protocol-document half: the entry a load selects must be one the
    producer will write, for every point of the consuming document."""
    entries = _bundle_entries(producer, rest)
    if entries is None:
        return
    for expanded, point in zip(consumer.expansion.points, consumer.point_documents):
        loads: list[tuple[str, str, Any]] = []
        for fname, spec in point.featurizers.items():
            if spec.file_path == run_path:
                kind = spec.kind if isinstance(spec.kind, str) else "identity"
                slots = FEATURIZER_SLOTS.get(kind, ())
                if slots:
                    loads.append((fname, slots[0], spec.entry))
        for pname, pspec in point.params.items():
            if pspec.file_path == run_path:
                loads.append((pname, selector_slot(pspec.entry, "value"), pspec.entry))
        for name, slot, authored in loads:
            want, implicit = entry_selection(authored, expanded.coords, name)
            try:
                select_entry(
                    entries.keys(),
                    slot,
                    want,
                    what=f"step {step!r}: {name!r} loads {run_path!r}",
                    coords_by_key=entries,
                    implicit=implicit,
                )
            except ValidationError as err:
                raise WorkflowError(9, str(err), path=f"steps.{step}") from err


# --------------------------------------------------------------------------- #
# canonical form (§7)
# --------------------------------------------------------------------------- #


def _canon_reference(ref: Reference) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if ref.step is not None:
        entry["step"] = ref.step
        entry["file"] = ref.file
    else:
        entry["path"] = ref.path
    if ref.key is not None:
        entry["key"] = ref.key
    if ref.slot is not None:
        entry["slot"] = ref.slot
    if ref.entry:
        entry["entry"] = dict(ref.entry)
    return entry


def _canon_output(decl: OutputDecl) -> dict[str, Any]:
    entry: dict[str, Any] = {"file": decl.file}
    if decl.columns is not None:
        entry["columns"] = dict(decl.columns)
    if decl.keys is not None:
        entry["keys"] = dict(decl.keys)
    return entry


def _canonicalize(
    document: WorkflowDocument,
    inner_digests: Mapping[str, str],
) -> dict[str, Any]:
    """The canonical form: every default materialized, `after` sorted, each
    protocol step stamped with its document's digest, and each script step with
    its script's content hash.

    ``output_dir`` is **absent**: it names where the run lands, not what the run
    is, so moving a run tree must not change the workflow's identity (§1.1)."""
    canon_steps: dict[str, Any] = {}
    for name in sorted(document.steps):
        step = document.steps[name]
        entry: dict[str, Any] = {"type": step.type}
        if step.description is not None:
            entry["description"] = step.description
        if isinstance(step, ProtocolStep):
            entry["document"] = step.document
            if step.set:
                entry["set"] = dict(step.set)
            if step.max_points is not None:
                entry["max_points"] = step.max_points
            entry["document_digest"] = inner_digests[name]
        else:
            entry["script"] = (
                {"module": step.module}
                if step.module is not None
                else {"path": step.path}
            )
            entry["script_sha256"] = step.script_sha256
            entry["inputs"] = {
                key: (
                    _canon_reference(value) if isinstance(value, Reference) else value
                )
                for key, value in sorted(step.inputs.items())
            }
            entry["outputs"] = {
                slot: _canon_output(decl) for slot, decl in sorted(step.outputs.items())
            }
            if step.runtime is not None:
                entry["runtime"] = dict(step.runtime)
            entry["is_deterministic"] = step.is_deterministic
        if step.after:
            entry["after"] = sorted(step.after)
        canon_steps[name] = entry

    canonical: dict[str, Any] = {"version": document.version}
    if document.description is not None:
        canonical["description"] = document.description
    canonical["steps"] = canon_steps
    return canonical
