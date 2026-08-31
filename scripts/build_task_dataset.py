"""Build a serialized dataset table from a task package (spec §2.2).

Protocol documents name datasets by ref, and refs resolve by reading bytes —
so a task's counterfactual dataset becomes usable by writing it out once,
here, rather than by generating it during a load. The bytes are
deterministic, so a table is reproducible from the parameters recorded in the
``<ref>.manifest.json`` sidecar this writes beside it.

Usage::

    # The weekdays interchange table the end-to-end IIA pin runs on.
    uv run python scripts/build_task_dataset.py \\
        --task natural_domains_arithmetic --set domain_type=weekdays \\
        --n 4 --seed 0 --target-variable result \\
        --out tests/protocol/fixtures/data/weekdays/task_n4_s0.json

    # A relation of the subject_object_relations factory.
    uv run python scripts/build_task_dataset.py \\
        --task subject_object_relations --set relation=word_first_letter \\
        --n 64 --seed 0 --out /tmp/word_first_letter.json

    # Rebuild in place and fail if the bytes moved (a determinism check).
    uv run python scripts/build_task_dataset.py ... --check

``--set k=v`` values are parsed as JSON when they parse, else kept as
strings, and are passed as keyword arguments to the task's config dataclass
(the convention :func:`causalab.tasks.serialize.config_class` resolves).
Singleton tasks take none.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from causalab.tasks.serialize import (
    build_manifest,
    config_class,
    serialize_counterfactual_dataset,
    table_bytes,
    write_dataset_table,
)


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    """``k=v`` pairs into config kwargs — same value semantics as the CLI's
    ``--set`` (JSON when it parses, else a bare string)."""
    out: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set takes key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _repo_commit() -> str | None:
    """The causalab commit the table was built at — provenance only, so a
    missing git (an installed wheel, a tarball) is not an error."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent.parent),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_task_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="a package under causalab/tasks/")
    parser.add_argument("--out", required=True, type=Path, help="table path (.json)")
    parser.add_argument("--n", required=True, type=int, help="counterfactual pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a config field for a factory task (repeatable)",
    )
    parser.add_argument(
        "--target-variable",
        action="append",
        default=[],
        metavar="NAME",
        help="variable(s) the interchange replaces; defaults to the task's "
        "TARGET_VARIABLE",
    )
    parser.add_argument(
        "--generator",
        default="generate_dataset",
        help="which generator in the task's counterfactuals.py to call",
    )
    parser.add_argument(
        "--answer-variable",
        default=None,
        help="variable whose output_tokens declaration supplies the answer-form "
        "columns (default: the only one declared)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the table would change — the "
        "determinism guard for committed tables",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = _parse_set(args.set)
    cls = config_class(args.task)
    if overrides and cls is None:
        raise SystemExit(
            f"task {args.task!r} has no config dataclass, so --set has nothing "
            "to configure"
        )
    task_cfg = cls(**overrides) if cls is not None and overrides else None

    dataset = serialize_counterfactual_dataset(
        args.task,
        n=args.n,
        seed=args.seed,
        task_cfg=task_cfg,
        target_variables=args.target_variable or None,
        generator=args.generator,
        answer_variable=args.answer_variable,
    )
    if args.check:
        fresh = table_bytes(dataset.rows)
        on_disk = args.out.read_bytes() if args.out.is_file() else None
        if on_disk != fresh:
            reason = "does not exist" if on_disk is None else "differs"
            print(f"CHANGED {args.out} {reason} — rerun without --check to write")
            return 1
        print(f"unchanged {args.out} ({len(dataset.rows)} rows)")
        return 0

    manifest = build_manifest(dataset, task_cfg=overrides, commit=_repo_commit())
    digest = write_dataset_table(dataset.rows, args.out, manifest=manifest)
    print(f"wrote {args.out} ({len(dataset.rows)} rows, digest {digest[:12]}…)")
    if dataset.match_mode == "prefix":
        print(
            f"note: the task declares match_mode 'prefix' for "
            f"{dataset.answer_variable!r} — a match metric over this table wants "
            '"mode": "first_token" (spec §2.10)'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
