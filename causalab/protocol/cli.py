"""The ``causalab`` CLI (spec §9): ``run · validate · explain · digest``.

Every verb loads through :func:`causalab.protocol.loader.load` against a
resolution environment built from ``--data-root`` / ``--artifacts-root``
(both default to the current directory). ``--set path=value`` applies an
ad-hoc override before loading — exploration only; promote anything that
matters into the file.

``run`` needs an execution backend; the reference backend
(:mod:`causalab.neural.pytorch_hooks`) is imported lazily so the pure
verbs stay torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from causalab.protocol.backend import requires
from causalab.protocol.errors import ProtocolError
from causalab.protocol.loader import LoadedProtocol, check_data_columns, load
from causalab.protocol.plan import plan_point
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.sweep import coordinate_label

__all__ = ["main"]


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set takes path=value, got {item!r}")
        dotted, _, raw_value = item.partition("=")
        try:
            overrides[dotted] = json.loads(raw_value)
        except json.JSONDecodeError:
            overrides[dotted] = raw_value  # a bare word is a string
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalab",
        description="Intervention protocols: run, validate, explain, digest "
        "(docs/intervention_protocol.md).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb, help_text in (
        ("run", "validate, expand, plan, execute, stamp"),
        ("validate", "the §5 load-error checklist"),
        (
            "explain",
            "models, forward plan, point count, requires, digest, save products",
        ),
        ("digest", "the campaign digest"),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("document", type=Path, help="a protocol JSON (or YAML) file")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
        p.add_argument("--data-root", type=Path, default=Path("."))
        p.add_argument("--artifacts-root", type=Path, default=Path("."))
        p.add_argument(
            "--max-points",
            type=int,
            default=None,
            help="override the sweep point cap (§5.14)",
        )
        if verb == "validate":
            p.add_argument(
                "--data", action="store_true", help="also check column references"
            )
        if verb == "run":
            p.add_argument(
                "--out", type=Path, required=True, help="run output directory"
            )
    return parser


def _env(args: argparse.Namespace) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=args.data_root),
        artifacts=FileArtifacts(root=args.artifacts_root),
    )


def _load(args: argparse.Namespace, env: ResolutionEnv) -> LoadedProtocol:
    from causalab.protocol.sweep import DEFAULT_POINT_CAP

    return load(
        args.document,
        env,
        overrides=_parse_set(args.set),
        point_cap=args.max_points if args.max_points is not None else DEFAULT_POINT_CAP,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env = _env(args)
    try:
        loaded = _load(args, env)
        if args.verb == "validate":
            if args.data:
                check_data_columns(loaded, env)
            n = len(loaded.expansion.points)
            print(
                f"OK: {args.document} — {n} point{'s' if n != 1 else ''}, "
                f"digest {loaded.document_digest[:16]}…"
            )
            return 0
        if args.verb == "digest":
            print(loaded.document_digest)
            return 0
        if args.verb == "explain":
            _explain(loaded)
            return 0
        # run — the reference backend is an optional, lazily-imported extra
        # so the pure verbs stay torch-free (importlib keeps the layering
        # honest: protocol/ never links against an execution engine)
        import importlib

        try:
            hooks = importlib.import_module("causalab.neural.pytorch_hooks")
        except ModuleNotFoundError as err:
            print(
                f"refused: no execution backend available ({err}) — 'run' needs "
                "the reference backend causalab.neural.pytorch_hooks",
                file=sys.stderr,
            )
            return 1
        result = _run(loaded, env, hooks.PytorchHooksBackend(), args.out)
        for manifest_path, disk_path in sorted(result.files.items()):
            print(f"saved {manifest_path} -> {disk_path}")
        return 0
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


def _run(loaded: LoadedProtocol, env: ResolutionEnv, backend: Any, out: Path) -> Any:
    from causalab.protocol.backend import ExecutionRequest, choose_backend

    chosen = choose_backend(loaded.point_documents[0], [backend])
    request = ExecutionRequest(
        points=tuple(p.raw for p in loaded.expansion.points),
        canonical=loaded.canonical_points,
        digests=loaded.point_digests,
        coords=tuple(p.coords for p in loaded.expansion.points),
        document_digest=loaded.document_digest,
        env=env,
        output_dir=out,
    )
    return chosen.execute(request)


def _explain(loaded: LoadedProtocol) -> None:
    doc = loaded.point_documents[0]
    axes = loaded.expansion.axes
    print(f"digest    {loaded.document_digest}")
    if axes:
        print(
            f"axes      {', '.join(f'{a.id} ({len(a.values)} values)' for a in axes)}"
        )
    print(f"points    {len(loaded.expansion.points)}")
    print(f"requires  {sorted(requires(doc)) or 'nothing beyond a forward pass'}")
    plan = plan_point(doc)
    print(f"forwards  {plan.num_forwards} per point")
    for group in plan.groups:
        taps = ", ".join(t.read for t in group.taps) or "(no reads — operands only)"
        print(f"  {group.model} on {group.input}: {taps}")
    print("save")
    for entry in doc.save:
        binding = (
            f"model={entry.model}, input={entry.input}"
            if entry.site is None
            else f"site={entry.site}"
        )
        print(f"  {entry.value} ({binding}) -> {entry.file_path}")
    if axes:
        first = loaded.expansion.points[0]
        print(
            f"first point {coordinate_label(first.coords)} digest {loaded.point_digests[0][:16]}…"
        )


if __name__ == "__main__":
    raise SystemExit(main())
