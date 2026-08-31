"""Task packages → serialized dataset tables (spec §2.2).

A protocol document names a dataset by ref and the resolver reads bytes
(:class:`causalab.protocol.resolve.FileDatasets`). This module is the other
half: it turns a task's causal model and counterfactual generator into those
bytes, *ahead of* the load. Nothing here runs at load or run time — which is
the point. Resolution stays stdlib-only, and a document's digest never
depends on importing task code or a tokenizer.

The one principle the row vocabulary encodes: **anything per-row or
task-semantic is computed here and serialized as a column** — the rendered
prompts, the answers, the post-intervention label, the equivalent answer
forms, the values that place a position per row. Documents reference
columns; they never compute.

Columns written per example (§2.2 names, the same ones the committed
fixtures use):

``input``
    The base prompt, the causal model's ``raw_input``.
``counterfactual_inputs``
    A one-element list — the counterfactual prompt. Documents select it as
    ``counterfactual_inputs[0]``.
``base_answer`` / ``cf_answer``
    Each prompt's own answer (``raw_output``).
``label``
    The answer *after* the interchange, from
    :meth:`CausalModel.label_counterfactual_data` — what an IIA metric
    scores against. It equals ``cf_answer`` only when the intervention
    replaces every variable the answer depends on, so the two are separate
    columns on purpose.
``<answer>_forms``
    The equivalent surface forms of each answer above, from the causal
    model's ``output_tokens`` declaration — the group a ``match`` metric
    consumes (§2.10).
``<variable>``
    Every causal-model variable of the *base* trace, stringified: the
    per-row values that ``{"variable": …}`` and ``{"column": …}`` positions
    resolve (§2.3).
``counterfactual_inputs_variables``
    The same variables for the counterfactual side, in the per-role
    ``<field>_variables`` convention position resolution reads.

Provenance lives beside the table in a ``<ref>.manifest.json`` sidecar that
resolution ignores: how the table was built, and the digest of the bytes it
describes. The table itself is the content-addressed unit (§7).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.causal.causal_model import CausalModel
from causalab.tasks.loader import load_task, load_task_counterfactuals

__all__ = [
    "RESERVED_COLUMNS",
    "SerializedDataset",
    "build_manifest",
    "config_class",
    "serialize_counterfactual_dataset",
    "serialize_examples",
    "table_bytes",
    "write_dataset_table",
]

#: Column names the row vocabulary owns. A causal-model variable colliding
#: with one of these is refused rather than silently overwriting it.
RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "input",
        "counterfactual_inputs",
        "counterfactual_inputs_variables",
        "base_answer",
        "base_answer_forms",
        "cf_answer",
        "cf_answer_forms",
        "label",
        "label_forms",
    }
)

#: Trace variables that are already columns in their own right.
_TEXT_VARIABLES: frozenset[str] = frozenset({"raw_input", "raw_output"})


@dataclasses.dataclass(frozen=True)
class SerializedDataset:
    """A built table plus what built it — the manifest's raw material."""

    rows: list[dict[str, Any]]
    task: str
    generator: str
    n: int
    #: ``None`` when the examples did not come from a seeded generator — an
    #: honest gap in the manifest is better than a number nothing reproduces
    #: (:func:`serialize_examples`).
    seed: int | None
    target_variables: tuple[str, ...]
    answer_variable: str | None
    #: The task's declared string-match mode for the answer variable
    #: (``exact`` / ``prefix``), recorded so a document author knows whether
    #: the answer space needs ``match``'s ``first_token`` mode (§2.10).
    match_mode: str | None


def serialize_counterfactual_dataset(
    task_name: str,
    *,
    n: int,
    seed: int,
    task_cfg: Any = None,
    target_variables: Sequence[str] | None = None,
    generator: str = "generate_dataset",
    answer_variable: str | None = None,
) -> SerializedDataset:
    """One task's counterfactual dataset as serializable rows.

    Args:
        task_name: A task package under ``causalab/tasks/``.
        n: Number of counterfactual pairs to generate.
        seed: Generator seed. The shipped generators snapshot and restore the
            global RNG, so the same (task, cfg, n, seed) yields the same rows.
        task_cfg: Config object for a factory task; ``None`` for a singleton.
        target_variables: The variables the interchange replaces. Defaults to
            the task's ``TARGET_VARIABLE``, which is what its own analyses use.
        generator: Which generator in the task's ``counterfactuals.py`` to
            call (e.g. ``generate_resample_dataset`` for a noise floor).
        answer_variable: The variable whose ``output_tokens`` declaration
            supplies the answer-form columns. Defaults to the sole variable
            the causal model declares forms for; ``None`` and no declaration
            means no ``_forms`` columns.

    Raises:
        ValueError: on a task whose generator produces more than one
            counterfactual per example (no v1 column vocabulary for it), a
            variable colliding with a reserved column name, or an answer
            value the task declares no forms for.
    """
    task = load_task(task_name, task_cfg=task_cfg)
    generators = load_task_counterfactuals(task_name)
    if not hasattr(generators, generator):
        raise ValueError(
            f"task {task_name!r} has no counterfactual generator {generator!r} "
            f"(has {sorted(name for name in dir(generators) if name.startswith('generate'))})"
        )
    targets = list(target_variables or ([task.intervention_variable] or []))
    if not targets or targets == [None]:
        raise ValueError(
            f"task {task_name!r} declares no TARGET_VARIABLE — pass "
            "target_variables explicitly so the label is well defined"
        )
    model: CausalModel = task.causal_model
    examples = getattr(generators, generator)(model, n, seed)
    return serialize_examples(
        model,
        examples,
        target_variables=targets,
        answer_variable=answer_variable,
        task_label=task_name,
        generator=generator,
        n=n,
        seed=seed,
    )


def serialize_examples(
    model: CausalModel,
    examples: Sequence[Mapping[str, Any]],
    *,
    target_variables: Sequence[str],
    answer_variable: str | None = None,
    task_label: str = "inline",
    generator: str = "inline",
    n: int | None = None,
    seed: int | None = None,
) -> SerializedDataset:
    """The same rows, from a causal model and an example list you already have.

    :func:`serialize_counterfactual_dataset` goes through
    :func:`~causalab.tasks.loader.load_task`, which needs a task **package**
    inside the library checkout. The causal protocol's step 3 tells an author
    to write ``models.py`` and ``counterfactuals.py`` in their own working
    directory — and then had nowhere to go: there was no public path from a
    hand-authored causal model to a serialized table, so the step dead-ended
    at "now build the task package".

    This is that path, and it is the *same* code: the package entry point
    resolves its task and then calls this, so the two cannot drift.

    Args:
        model: The causal model the examples were generated from.
        examples: Counterfactual examples — what a generator returns.
        target_variables: The variables the interchange replaces. Required
            here (there is no package to read a ``TARGET_VARIABLE`` from), and
            what the ``label`` column is computed against.
        answer_variable: As in :func:`serialize_counterfactual_dataset`.
        task_label: What the rows record as their task. Provenance only — no
            package of this name has to exist.
        generator, n, seed: Recorded on the result for the manifest. Leave
            them alone when the examples did not come from a seeded generator;
            ``None`` is more honest than a number nothing can reproduce.
    """
    targets = list(target_variables)
    if not targets or targets == [None]:
        raise ValueError(
            "target_variables is required: it is what the `label` column — the "
            "answer after the interchange — is computed against"
        )
    labeled = model.label_counterfactual_data(list(examples), targets)
    resolved_answer = _answer_variable(model, answer_variable)
    forms_of = _forms_lookup(model, answer_variable)
    return SerializedDataset(
        rows=[_row(example, forms_of, task_label) for example in labeled],
        task=task_label,
        generator=generator,
        n=len(labeled) if n is None else n,
        seed=seed,
        target_variables=tuple(targets),
        answer_variable=resolved_answer,
        match_mode=(model.match_modes or {}).get(resolved_answer or "", None),
    )


def _answer_variable(model: CausalModel, answer_variable: str | None) -> str | None:
    """The variable whose ``output_tokens`` declaration supplies answer forms:
    the caller's choice, or the sole declared one."""
    if answer_variable is not None:
        return answer_variable
    declared = model.output_tokens or {}
    return next(iter(declared)) if len(declared) == 1 else None


def _forms_lookup(model: CausalModel, answer_variable: str | None):
    """``(setting) -> forms | None`` for the answer-form columns.

    The forms come from the causal model's ``output_tokens`` — the task's own
    declaration of which surface strings count as one answer (§2.10). Keyed
    by the *answer variable's* value, not by the answer string, because that
    is how the declaration is keyed (``result`` → ``[" Friday", "Friday"]``).
    """
    declared: Mapping[str, Mapping[Any, list[str]]] = model.output_tokens or {}
    variable = answer_variable
    if variable is None:
        if len(declared) != 1:
            return lambda setting: None
        variable = next(iter(declared))
    if variable not in declared:
        raise ValueError(
            f"the causal model declares no output_tokens for {variable!r} "
            f"(declared: {sorted(declared)}) — no answer forms to serialize"
        )
    var_map = declared[variable]

    def forms(setting: Any) -> list[str]:
        value = setting[variable]
        key = tuple(value) if isinstance(value, list) else value
        if key not in var_map:
            raise ValueError(
                f"the causal model declares no output_tokens forms for "
                f"{variable}={key!r} — the answer space and the declaration "
                "disagree, which would silently mis-score a match metric"
            )
        return list(var_map[key])

    return forms


def _row(example: Mapping[str, Any], forms_of, task_name: str) -> dict[str, Any]:
    base = example["input"]
    counterfactuals = example["counterfactual_inputs"]
    if len(counterfactuals) != 1:
        raise ValueError(
            f"task {task_name!r} produced {len(counterfactuals)} counterfactuals "
            "for one example; v1 tables carry exactly one (the shipped "
            "generators' shape) — the column vocabulary for several is undefined"
        )
    counterfactual = counterfactuals[0]
    setting = example["setting"]
    row: dict[str, Any] = {
        "input": base["raw_input"],
        "counterfactual_inputs": [counterfactual["raw_input"]],
        "base_answer": base["raw_output"],
        "cf_answer": counterfactual["raw_output"],
        "label": example["label"],
    }
    base_forms = forms_of(base)
    if base_forms is not None:
        row["base_answer_forms"] = base_forms
        row["cf_answer_forms"] = forms_of(counterfactual)
        row["label_forms"] = forms_of(setting)
    row.update(_variable_columns(base, task_name))
    row["counterfactual_inputs_variables"] = [_variables(counterfactual)]
    return row


def _variables(trace: Any) -> dict[str, str]:
    """A trace's variables as strings — position resolution matches substrings
    of the row's text, so the serialized form is the string form."""
    values = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace)
    return {
        name: str(value)
        for name, value in sorted(values.items())
        if name not in _TEXT_VARIABLES
    }


def _variable_columns(trace: Any, task_name: str) -> dict[str, str]:
    columns = _variables(trace)
    collisions = sorted(set(columns) & RESERVED_COLUMNS)
    if collisions:
        raise ValueError(
            f"task {task_name!r} has causal-model variables {collisions} that "
            f"collide with the reserved row columns {sorted(RESERVED_COLUMNS)} — "
            "rename the variable or serialize this task by hand"
        )
    return columns


def table_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """The exact bytes a table serializes to: sorted keys, fixed indent,
    trailing newline. Deterministic on purpose — the content digest stamped
    into a canonical form (§7) has to be reproducible from the manifest's
    build parameters, on any machine."""
    return (json.dumps(list(rows), indent=1, sort_keys=True) + "\n").encode()


def write_dataset_table(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Write a table (and its manifest sidecar) and return its content digest
    — the sha256 of exactly the bytes
    :class:`~causalab.protocol.resolve.FileDatasets` will read back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table_bytes(rows)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if manifest is not None:
        sidecar = path.with_name(f"{path.stem}.manifest.json")
        sidecar.write_text(
            json.dumps({**manifest, "digest": digest}, indent=1, sort_keys=True) + "\n"
        )
    return digest


def build_manifest(
    dataset: SerializedDataset,
    *,
    task_cfg: Mapping[str, Any] | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """The provenance sidecar's content: how to rebuild this table.

    ``task_cfg`` is the caller's own JSON view of the config (the CLI's
    ``--set`` values), not an introspection of the resolved config object —
    those carry callables and derived value lists, and the reproducible input
    is what was *asked for*.
    """
    manifest: dict[str, Any] = {
        "built_by": "causalab.tasks.serialize",
        "task": dataset.task,
        "task_cfg": dict(task_cfg or {}),
        "generator": dataset.generator,
        "n": dataset.n,
        "seed": dataset.seed,
        "target_variables": list(dataset.target_variables),
        "n_rows": len(dataset.rows),
        "columns": sorted({key for row in dataset.rows for key in row}),
    }
    if dataset.answer_variable is not None:
        manifest["answer_variable"] = dataset.answer_variable
    if dataset.match_mode is not None:
        manifest["declared_match_mode"] = dataset.match_mode
    if commit is not None:
        manifest["causalab_commit"] = commit
    return manifest


def config_class(task_name: str) -> type | None:
    """The config dataclass of a factory task, by convention.

    ``causalab/tasks/<name>/config.py`` holds exactly one module-level
    dataclass whose name ends in ``Config`` (``NaturalDomainConfig``,
    ``SubjectObjectRelationsConfig``, …). Singleton tasks have none, and take
    no config. Same convention-over-registry approach as
    :mod:`causalab.tasks.loader`.
    """
    import importlib

    try:
        module = importlib.import_module(f"causalab.tasks.{task_name}.config")
    except ModuleNotFoundError:
        return None
    found = [
        value
        for name, value in vars(module).items()
        if name.endswith("Config")
        and dataclasses.is_dataclass(value)
        and isinstance(value, type)
        and value.__module__ == module.__name__
    ]
    if len(found) > 1:
        raise ValueError(
            f"task {task_name!r} has several config dataclasses "
            f"({sorted(cls.__name__ for cls in found)}) — pass the config object "
            "directly instead of relying on the convention"
        )
    return found[0] if found else None
