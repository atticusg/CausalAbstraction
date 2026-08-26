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

from causalab.protocol.backend import requires_campaign
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
            p.add_argument(
                "--device",
                default="cpu",
                help="torch device string for the reference backend "
                "(cpu, cuda, cuda:1, mps)",
            )
            p.add_argument(
                "--dtype",
                choices=("fp32", "bf16", "fp16"),
                default="fp32",
                help="model dtype for the reference backend",
            )
            p.add_argument(
                "--points",
                default=None,
                metavar="START:STOP",
                help="execute only this half-open point-index range of the "
                "expanded campaign — the seam external schedulers shard on "
                "(document runs only; digests and stamps are unaffected)",
            )
    return parser


def _parse_points(spec: str, n_points: int) -> range:
    """The --points shard selector: a half-open [start, stop) index range
    into the campaign's expanded points, refused rather than clamped when
    it falls outside [0, n_points] or selects nothing."""
    try:
        start_text, stop_text = spec.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError:
        raise ProtocolError("P4", f"--points {spec!r} is not START:STOP") from None
    if not (0 <= start < stop <= n_points):
        raise ProtocolError(
            "P4",
            f"--points {spec!r} is outside the campaign's {n_points} points "
            "or selects none",
        )
    return range(start, stop)


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


def _ensure_model_registered(args: argparse.Namespace) -> None:
    """The run verb touches the model anyway, so an unregistered key is
    resolved from its HF config and registered before canonicalization —
    the pure verbs stay registry-only so digests never depend on the
    network."""
    from causalab.protocol.loader import apply_overrides, load_text

    raw = apply_overrides(dict(load_text(args.document)), _parse_set(args.set))
    _register_model_key(raw)


def _register_model_key(raw: dict[str, Any]) -> None:
    from causalab.protocol.registry import (
        get_model_info,
        model_info_from_hf_config,
        register_model,
    )

    model = raw.get("model", raw.get("neural_model", {}))
    key = model.get("key") if isinstance(model, dict) else None
    if not isinstance(key, str):
        return
    try:
        get_model_info(key)
    except ProtocolError:
        from transformers import AutoConfig

        revision = model.get("revision", "main") if isinstance(model, dict) else "main"
        config = AutoConfig.from_pretrained(key, revision=revision)
        register_model(model_info_from_hf_config(key, config))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env = _env(args)
    try:
        from causalab.protocol.loader import load_text as _load_text
        from causalab.protocol.workflow import is_workflow

        if is_workflow(_load_text(args.document)):
            return _workflow_main(args, env)
        if args.verb == "run":
            _ensure_model_registered(args)
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
        result = _run(
            loaded,
            env,
            hooks.PytorchHooksBackend(device=args.device, dtype=args.dtype),
            args.out,
            points=args.points,
        )
        for manifest_path, disk_path in sorted(result.files.items()):
            print(f"saved {manifest_path} -> {disk_path}")
        return 0
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


def _run(
    loaded: LoadedProtocol,
    env: ResolutionEnv,
    backend: Any,
    out: Path,
    *,
    points: str | None = None,
) -> Any:
    from causalab.protocol.backend import ExecutionRequest, choose_backend

    chosen = choose_backend(list(loaded.point_documents), [backend])
    # --points slices every per-point tuple in lockstep; the campaign
    # digest is untouched — a shard's artifacts still stamp and dedup as
    # members of the whole campaign, so an external scheduler can fan
    # shards out and recombine by digest.
    selected = (
        _parse_points(points, len(loaded.expansion.points))
        if points is not None
        else range(len(loaded.expansion.points))
    )
    request = ExecutionRequest(
        points=tuple(loaded.expansion.points[i].raw for i in selected),
        canonical=tuple(loaded.canonical_points[i] for i in selected),
        digests=tuple(loaded.point_digests[i] for i in selected),
        coords=tuple(loaded.expansion.points[i].coords for i in selected),
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
    needed = requires_campaign(list(loaded.point_documents))
    print(f"requires  {sorted(needed) or 'nothing beyond a forward pass'}")
    plan = plan_point(doc)
    print(f"forwards  {plan.num_forwards} per point")
    for group in plan.groups:
        taps = ", ".join(t.read for t in group.taps) or "(no reads — operands only)"
        print(f"  {group.model} on {group.input}: {taps}")
        if group.decode_depth:
            # print what the decode obliges, so the bill of a document is
            # readable before it runs — the mechanism stays the backend's
            print(f"    decode {group.decode_depth} tokens (greedy)")
            for item in group.materialize:
                needs = (
                    "distribution per addressed position"
                    if item.needs_distribution
                    else "no distribution — ids and activations only"
                )
                print(f"    {item.read} at {item.site}: {needs}")
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


# --------------------------------------------------------------------------- #
# workflow documents (docs/workflow_protocol.md §9) — dispatch on `steps`
# --------------------------------------------------------------------------- #


def _workflow_main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    from causalab.protocol.loader import apply_overrides as _apply
    from causalab.protocol.loader import load_text as _load_text
    from causalab.protocol.workflow import ProtocolStep, load_workflow

    if args.verb == "run":
        if args.points is not None:
            print(
                "refused: --points shards a single document's expanded "
                "campaign; a workflow schedules whole steps — shard the "
                "inner document runs instead",
                file=sys.stderr,
            )
            return 1
        # the run verb touches models anyway: pre-register every inner model
        # key BEFORE loading (canonicalization derives widths from the registry)
        raw_wf = _apply(dict(_load_text(args.document)), _parse_set(args.set))
        workflow_dir = args.document.resolve().parent
        for step_raw in raw_wf.get("steps", {}).values():
            if not isinstance(step_raw, dict) or step_raw.get("type") != "protocol":
                continue
            document = step_raw.get("document")
            if not isinstance(document, str):
                continue  # malformed shapes refuse properly in load_workflow
            doc_path = (workflow_dir / document).resolve()
            if not doc_path.is_file():
                continue
            try:
                inner_raw = _apply(
                    dict(_load_text(doc_path)), dict(step_raw.get("set", {}) or {})
                )
            except ProtocolError:
                continue
            _register_model_key(inner_raw)

    loaded = load_workflow(args.document.resolve(), env, overrides=_parse_set(args.set))
    if args.verb == "validate":
        if getattr(args, "data", False):
            for name in loaded.order:
                inner = loaded.inner.get(name)
                if inner is not None:
                    check_data_columns(inner, env)
        print(
            f"OK: {args.document} — {len(loaded.document.steps)} steps, "
            f"digest {loaded.digest[:16]}…"
        )
        return 0
    if args.verb == "digest":
        print(loaded.digest)
        return 0
    if args.verb == "explain":
        print(f"digest    {loaded.digest}")
        print(f"schedule  {len(loaded.levels)} levels")
        for i, level in enumerate(loaded.levels):
            print(f"  level {i}: {', '.join(level)}")
        for name in loaded.order:
            step = loaded.document.steps[name]
            if isinstance(step, ProtocolStep):
                inner = loaded.inner[name]
                kind = loaded.inner_digest_kind[name]
                print(
                    f"  {name}: protocol {step.document} — "
                    f"{len(inner.expansion.points)} point(s), "
                    f"{kind} digest {loaded.inner_digests[name][:16]}…"
                )
            else:
                print(f"  {name}: {step.type}")
        print("save")
        for entry in loaded.document.save:
            print(f"  {entry.step}/{entry.value} -> {entry.file_path}")
        return 0
    # run
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
    from causalab.workflow import run_workflow

    result = run_workflow(
        loaded,
        env,
        args.out,
        [hooks.PytorchHooksBackend(device=args.device, dtype=args.dtype)],
    )
    for file_path, disk_path in sorted(result.published.items()):
        print(f"published {file_path} -> {disk_path}")
    print(f"manifest {args.out / 'workflow.json'}")
    return 0
