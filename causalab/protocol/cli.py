"""The CLI verbs for an **intervention protocol** document (spec §9).

Every verb loads through :func:`causalab.protocol.loader.load` against a
resolution environment built by :mod:`causalab.cli`, which also owns argument
parsing and the dispatch between document types. This module therefore links
against nothing in the workflow layer.

``run`` needs an execution backend; the reference backend
(:mod:`causalab.neural.pytorch_hooks`) is imported lazily so the pure verbs stay
torch-free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from causalab.protocol.backend import requires_campaign
from causalab.protocol.errors import ProtocolError
from causalab.protocol.loader import LoadedProtocol, check_data_columns, load
from causalab.protocol.plan import plan_point
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.sweep import coordinate_label

__all__ = ["main"]


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


def _load(args: argparse.Namespace, env: ResolutionEnv) -> LoadedProtocol:
    from causalab.protocol.sweep import DEFAULT_POINT_CAP

    return load(
        args.document,
        env,
        overrides=dict(args.parsed_set),
        point_cap=args.max_points if args.max_points is not None else DEFAULT_POINT_CAP,
    )


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    """Run one verb against an **intervention protocol** document.

    Dispatch between document types lives in :mod:`causalab.cli`, so this module
    — and the whole ``protocol/`` package — links against nothing in the
    workflow layer. That is what lets someone use the intervention protocol on
    its own."""
    try:
        if args.verb == "run":
            from causalab.cli import ensure_model_registered

            ensure_model_registered(args)
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
