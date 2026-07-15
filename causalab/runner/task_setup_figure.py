"""Emit a ``task_setup.html`` figure for an experiment's task(s).

A thin runner-layer CLI: given one or more task specs, it loads each task,
samples a few base ``prompt -> expected answer`` examples from the task's
symbolic causal model (via :func:`generate_datasets` — no neural-model forward
pass), and hands the records to the stdlib renderer
:func:`causalab.io.plots.task_setup.render_task_setup_html`.

It sits in ``causalab.runner`` (not ``causalab.io``) because it imports the task
loader and dataset sampler — the ``io`` layer may not depend on ``tasks``/runner
helpers (layering invariant 3). The renderer it calls stays pure and io-layer.

Produces the experiment viewer's "Task setup" section. Tasks are passed as a
JSON list (inline or via ``@file``), each item::

    {"name": "comparative_degree", "target_variable": "degree",
     "config": {}, "description": "Rate A vs B on a 1-5 scale."}

``config`` (factory-task fields / ``score_by``) and ``description`` are optional.

CLI::

    python -m causalab.runner.task_setup_figure \\
        --tasks '[{"name": "comparative_degree", "target_variable": "degree"}]' \\
        --n 3 --out "${SESSION_DIR}/plan/figures/task_setup.html"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.io.plots.task_setup import write_task_setup_html
from causalab.runner.helpers import generate_datasets, resolve_task
from causalab.tasks.loader import Task


def examples_for_task(
    task: Task, *, n_examples: int, seed: int
) -> list[dict[str, str]]:
    """Sample up to ``n_examples`` base ``{prompt, answer}`` pairs from a task.

    Delegates to :func:`generate_datasets`, which runs the task's own (symbolic)
    causal model — no neural forward pass — and is the same generator the
    experiment uses, so the examples match what the model will be scored on. Only
    the *base* ``input`` of each example is read (``raw_input`` / ``raw_output``);
    the counterfactuals it also builds are ignored here. Deduped prompts may yield
    fewer than ``n_examples`` rows; the renderer shows what is available.
    """
    train, _ = generate_datasets(task, n_train=n_examples, n_test=0, seed=seed)
    rows: list[dict[str, str]] = []
    for ex in train[:n_examples]:
        inp = ex["input"]
        rows.append({"prompt": str(inp["raw_input"]), "answer": str(inp["raw_output"])})
    return rows


def record_for_task(
    task: Task,
    *,
    n_examples: int,
    seed: int,
    description: str | None = None,
) -> dict[str, Any]:
    """Build one renderer record from a resolved task."""
    return {
        "name": task.name,
        "target_variable": task.intervention_variable,
        "description": description,
        "examples": examples_for_task(task, n_examples=n_examples, seed=seed),
    }


def build_task_records(
    task_specs: Sequence[Mapping[str, Any]],
    *,
    n_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Resolve each spec into a task and build its renderer record.

    Each spec is ``{name, target_variable, config?, description?}``. ``config``
    holds factory-task fields / ``score_by`` and defaults to ``{}``.
    """
    records: list[dict[str, Any]] = []
    for spec in task_specs:
        name = spec["name"]
        target_variable = spec.get("target_variable")
        config = dict(spec.get("config") or {})
        task, _ = resolve_task(name, config, target_variable, seed=seed)
        records.append(
            record_for_task(
                task,
                n_examples=n_examples,
                seed=seed,
                description=spec.get("description"),
            )
        )
    return records


def _load_task_specs(tasks: str | None, tasks_file: str | None) -> list[dict[str, Any]]:
    """Parse the ``--tasks`` / ``--tasks-file`` JSON list of task specs."""
    if tasks_file:
        raw = Path(tasks_file).read_text(encoding="utf-8")
    elif tasks:
        raw = tasks
    else:
        raise ValueError("provide --tasks (inline JSON) or --tasks-file (a path)")
    specs = json.loads(raw)
    if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
        raise ValueError(
            "task specs must be a JSON list of {name, target_variable, ...} objects"
        )
    if not specs:
        raise ValueError("no task specs provided")
    return specs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.runner.task_setup_figure",
        description="Render a task_setup.html figure (task(s) + worked prompt->answer examples).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tasks",
        help='inline JSON list of task specs, e.g. \'[{"name":"t","target_variable":"v"}]\'',
    )
    src.add_argument(
        "--tasks-file", help="path to a JSON file holding the task-spec list"
    )
    parser.add_argument(
        "--n", type=int, default=3, help="examples sampled per task (default 3)"
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument("--out", required=True, help="output path for task_setup.html")
    parser.add_argument("--title", default="Task setup", help="page title")
    args = parser.parse_args(argv)

    specs = _load_task_specs(args.tasks, args.tasks_file)
    records = build_task_records(specs, n_examples=args.n, seed=args.seed)
    out = write_task_setup_html(records, args.out, title=args.title)
    n_examples = sum(len(r["examples"]) for r in records)
    print(
        f"[task_setup_figure] wrote {out} ({len(records)} task(s), {n_examples} examples)"
    )


if __name__ == "__main__":
    main()
