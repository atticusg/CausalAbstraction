"""Generate a comprehensive LaTeX table from all evaluation results.

Scans all ``causalab/tasks/*/outputs/*/path_steering/`` directories and
prints one consolidated LaTeX table to stdout.

Usage::

    uv run python -m causalab.io.plots.latex_table

    # Filter to specific tasks / model
    uv run python -m causalab.io.plots.latex_table \
        --tasks weekdays,months,years --model llama31_8b

    # Filter metrics / path_modes / subspace / manifold
    uv run python -m causalab.io.plots.latex_table \
        --metrics isometry,coherence \
        --path-modes geometric \
        --subspace pca_k8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TASKS_ROOT = os.path.join("causalab", "tasks")


@dataclass
class EvalRecord:
    """One row of evaluation results."""

    task: str
    model: str
    subspace: str
    manifold: str
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def collect_all_eval_records(
    model_filter: str | None = None,
    subspace_filter: str | None = None,
    manifold_filter: str | None = None,
    task_filter: list[str] | None = None,
) -> list[EvalRecord]:
    """Scan all tasks/models for evaluate/ metadata.json files."""
    records: list[EvalRecord] = []
    if not os.path.isdir(TASKS_ROOT):
        logger.warning("Tasks root not found: %s", TASKS_ROOT)
        return records

    for task_name in sorted(os.listdir(TASKS_ROOT)):
        if task_filter and task_name not in task_filter:
            continue
        outputs_dir = os.path.join(TASKS_ROOT, task_name, "outputs")
        if not os.path.isdir(outputs_dir):
            continue

        for model_id in sorted(os.listdir(outputs_dir)):
            if model_filter and model_id != model_filter:
                continue
            eval_root = os.path.join(outputs_dir, model_id, "path_steering")
            if not os.path.isdir(eval_root):
                continue

            for ss_sub in sorted(os.listdir(eval_root)):
                ss_path = os.path.join(eval_root, ss_sub)
                if not os.path.isdir(ss_path):
                    continue
                if subspace_filter and ss_sub != subspace_filter:
                    continue

                for m_sub in sorted(os.listdir(ss_path)):
                    m_path = os.path.join(ss_path, m_sub)
                    if not os.path.isdir(m_path):
                        continue
                    if manifold_filter and m_sub != manifold_filter:
                        continue

                    meta_path = os.path.join(m_path, "metadata.json")
                    if not os.path.isfile(meta_path):
                        continue

                    with open(meta_path) as f:
                        meta = json.load(f)

                    records.append(
                        EvalRecord(
                            task=task_name,
                            model=model_id,
                            subspace=ss_sub,
                            manifold=m_sub,
                            metrics=meta.get("criteria", {}),
                        )
                    )

    return records


def _fmt_val(val: dict[str, float] | list[float] | float) -> str:
    """Format a metric value dict as a compact LaTeX string.

    For dicts with mean/std keys: 'mean ± std'.
    For other dicts: 'key=val / key=val / ...'.
    Legacy tuple/list support for old metadata files.
    """
    if isinstance(val, dict):
        if "mean" in val and "std" in val:
            m, s = val["mean"], val["std"]
            if math.isnan(m):
                return "---"
            if math.isnan(s):
                return f"{m:.3f}"
            return f"{m:.3f} {{\\scriptsize$\\pm$ {s:.3f}}}"
        # Non mean/std dict (e.g. isometry): show all keys
        parts = []
        for k, v in val.items():
            if isinstance(v, float) and math.isnan(v):
                return "---"
            parts.append(f"{v:.3f}")
        return " / ".join(parts)
    # Legacy: tuple/list from old metadata
    if isinstance(val, (list, tuple)) and len(val) == 2:
        a, b = val
        if math.isnan(a) or math.isnan(b):
            return "---"
        return f"{a:.3f} {{\\scriptsize$\\pm$ {b:.3f}}}"
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return "---"
        return f"{val:.3f}"
    return str(val)


def _metric_display(name: str) -> str:
    """Pretty name for a metric key like 'coherence/geometric'."""
    return name.replace("_", " ").replace("/", " / ").title()


def build_latex_table(
    records: list[EvalRecord],
    metrics_filter: list[str] | None = None,
    path_modes_filter: list[str] | None = None,
) -> str:
    """Build a LaTeX table string from collected records.

    Pivoted layout — path modes become columns, metrics become rows.
    Each (task, model, subspace, manifold) combo gets multiple rows
    (one per metric), with ID columns using ``\\multirow`` to avoid
    repetition.

    Columns: Task | Model | Subspace | Manifold | Metric | <path_mode>…
    Non-path metrics (e.g. isometry) place their value in the first
    path-mode column with ``---`` in the rest.
    """
    if not records:
        return "% No evaluation records found.\n"

    # Discover all base metrics and path modes across records
    base_metrics_ordered: list[str] = []
    path_modes_ordered: list[str] = []
    base_seen: set[str] = set()
    pm_seen: set[str] = set()

    for r in records:
        for k in r.metrics:
            if "/" in k:
                base, pm = k.split("/", 1)
                if base not in base_seen:
                    base_seen.add(base)
                    base_metrics_ordered.append(base)
                if pm not in pm_seen:
                    pm_seen.add(pm)
                    path_modes_ordered.append(pm)
            else:
                if k not in base_seen:
                    base_seen.add(k)
                    base_metrics_ordered.append(k)

    # Apply filters
    if metrics_filter:
        base_metrics_ordered = [m for m in base_metrics_ordered if m in metrics_filter]
    if path_modes_filter:
        path_modes_ordered = [
            pm for pm in path_modes_ordered if pm in path_modes_filter
        ]

    if not base_metrics_ordered:
        return "% No matching metrics after filtering.\n"
    if not path_modes_ordered:
        return "% No matching path modes after filtering.\n"

    n_pm = len(path_modes_ordered)
    n_metrics = len(base_metrics_ordered)
    col_spec = "llll" + "l" + "c" * n_pm

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header_cells = ["Task", "Model", "Subspace", "Manifold", "Metric"]
    for pm in path_modes_ordered:
        header_cells.append(_metric_display(pm))
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    prev_task = None
    for ri, r in enumerate(records):
        if prev_task is not None and r.task != prev_task:
            # Replace trailing \addlinespace with \midrule at task boundary
            if lines[-1] == r"\addlinespace":
                lines[-1] = r"\midrule"
            else:
                lines.append(r"\midrule")
        prev_task = r.task

        id_labels = [
            r.task.replace("_", r"\_"),
            r.model.replace("_", r"\_"),
            r.subspace.replace("_", r"\_"),
            r.manifold.replace("_", r"\_"),
        ]

        for mi, metric in enumerate(base_metrics_ordered):
            row: list[str] = []

            # ID columns: multirow on first metric row, empty on rest
            if mi == 0:
                for label in id_labels:
                    row.append(rf"\multirow{{{n_metrics}}}{{*}}{{{label}}}")
            else:
                row.extend([""] * 4)

            # Metric name
            row.append(_metric_display(metric))

            # Check if this is a path-dependent metric
            has_path = any(f"{metric}/{pm}" in r.metrics for pm in path_modes_ordered)

            if has_path:
                for pm in path_modes_ordered:
                    key = f"{metric}/{pm}"
                    val = r.metrics.get(key)
                    if val is None:
                        row.append("---")
                    else:
                        row.append(_fmt_val(val))
            else:
                # Non-path metric: value in first column, dashes in rest
                val = r.metrics.get(metric)
                if val is None:
                    row.append("---")
                else:
                    row.append(_fmt_val(val))
                row.extend(["---"] * (n_pm - 1))

            row_sep = r" \\" if mi == n_metrics - 1 else r" \\[-2pt]"
            lines.append(" & ".join(row) + row_sep)

        # Thin space between record groups
        if ri < len(records) - 1:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Evaluation results.}")
    lines.append(r"\label{tab:eval-results}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def _parse_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated string into a list, or return None."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from evaluation results.",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter")
    parser.add_argument("--model", default=None, help="Model ID filter")
    parser.add_argument("--subspace", default=None, help="Subspace filter")
    parser.add_argument("--manifold", default=None, help="Manifold filter")
    parser.add_argument(
        "--metrics", default=None, help="Comma-separated metrics filter"
    )
    parser.add_argument(
        "--path-modes", default=None, help="Comma-separated path modes filter"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = collect_all_eval_records(
        model_filter=args.model,
        subspace_filter=args.subspace,
        manifold_filter=args.manifold,
        task_filter=_parse_csv(args.tasks),
    )

    if not records:
        logger.warning("No evaluation results found under %s", TASKS_ROOT)
        return

    latex = build_latex_table(
        records=records,
        metrics_filter=_parse_csv(args.metrics),
        path_modes_filter=_parse_csv(args.path_modes),
    )
    logger.info("%s", latex)


if __name__ == "__main__":
    main()
