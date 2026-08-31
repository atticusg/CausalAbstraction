"""The CLI verbs for an **intervention protocol** document (spec §9).

Every verb loads through :func:`causalab.protocol.loader.load` against a
resolution environment built by :mod:`causalab.cli`, which also owns argument
parsing and the dispatch between document types. This module therefore links
against nothing in the workflow layer.

``run`` needs an execution engine; the reference engine
(:mod:`causalab.neural.engines.pytorch_hooks`) is imported lazily so the pure verbs stay
torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from causalab.protocol.engine import requires_campaign
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.loader import (
    LoadedProtocol,
    check_data_columns,
    load,
    load_text,
)
from causalab.protocol.plan import plan_point
from causalab.protocol.method import (
    document_type,
    method_digest,
    parse_method,
)
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.schema import MODEL_DTYPE_DEFAULT
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
    its own.

    A **method** file (§1.1) is dispatched here rather than there: it is a
    protocol-family document, and telling it apart needs `document_type`,
    which is this package's."""
    try:
        raw = dict(load_text(args.document))
        if document_type(raw) == "method":
            return _method_main(args, raw)
        from causalab.cli import ensure_model_registered, wants_hf_registration

        if wants_hf_registration(args):
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
            _explain_engine(loaded, getattr(args, "engine", None))
            return 0
        # run — engines are optional, lazily-imported extras so the pure
        # verbs stay torch-free; --engine picks the list, choose_engine routes
        from causalab.cli import load_engines

        result = _run(
            loaded,
            env,
            load_engines(getattr(args, "engine", "pytorch_hooks"), args.device),
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
    engines: list[Any],
    out: Path,
    *,
    points: str | None = None,
) -> Any:
    from causalab.protocol.engine import ExecutionRequest, choose_engine

    chosen = choose_engine(list(loaded.point_documents), engines)
    # --points slices every per-point tuple in lockstep; the campaign
    # digest is untouched — a shard's artifacts still stamp and dedup as
    # members of the whole campaign, so an external scheduler can fan
    # shards out and recombine by digest.
    selected = (
        _parse_points(points, len(loaded.expansion.points))
        if points is not None
        else range(len(loaded.expansion.points))
    )
    _write_run_record(loaded, out, selected)
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


def _method_main(args: argparse.Namespace, raw: dict[str, Any]) -> int:
    """The verbs on a method file (§1.1).

    A method has no inputs, so there is nothing to plan, expand or run: what
    it can answer is "is this a well-formed method", "what does it hash to"
    and "what must I bind to use it" — the last being the thing a reader of a
    shared method actually needs.
    """
    if args.verb == "run":
        print(
            "refused: this is a method file — it names no network, no data and "
            "no addresses. Bind it from a document's `application` half "
            "(§1.1), and run that.",
            file=sys.stderr,
        )
        return 1
    method = parse_method(raw)
    if args.verb == "digest":
        print(method_digest(raw))
        return 0
    if args.verb == "validate":
        print(
            f"OK: {args.document} — method, digest {method_digest(raw)[:16]}…, "
            f"{len(method.signature.lines())} binding"
            f"{'s' if len(method.signature.lines()) != 1 else ''} to supply"
        )
        return 0
    print(f"digest    {method_digest(raw)}")
    if method.description:
        print(f"about     {method.description.splitlines()[0]}")
    print("binds     the application half must supply")
    for line in method.signature.lines():
        print(f"  {line}")
    print("save")
    for entry in raw.get("save", []):
        print(f"  {entry.get('value')} -> {entry.get('file_path')}")
    return 0


def _write_run_record(loaded: LoadedProtocol, out: Path, selected: range) -> Path:
    """``<out>/protocol.json`` — the record of what ran.

    The saved tables say what the numbers are; this says what produced them:
    the canonical document (every default materialized, dtype and
    quantization included), its digest, the per-point provenance digests, and
    the method this document was composed from. It is what someone reproducing
    the run reads first, and it is written before execution so a crashed run
    still says what it was.
    """
    record = {
        "document_digest": loaded.document_digest,
        "canonical": loaded.canonical_document,
        "points": [
            {
                "index": index,
                "digest": loaded.point_digests[index],
                "coords": dict(loaded.expansion.points[index].coords),
            }
            for index in selected
        ],
    }
    if loaded.method_digest is not None:
        record["method"] = {
            "digest": loaded.method_digest,
            "ref": loaded.method_ref,
        }
    out.mkdir(parents=True, exist_ok=True)
    target = out / "protocol.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return target


def _explain_engine(loaded: LoadedProtocol, choice: str | None) -> None:
    """Print which engine ``choose_engine`` would pick, or the §8 refusal.

    ``explain`` printed ``requires`` and stopped there, so routing could not be
    pre-flighted at all — and routing is exactly what is not obvious on a model
    where one family of components is hooks-only and another is nnsight-only.
    The refusal is the *more* useful answer of the two, so it is printed rather
    than raised.

    Opt-in because engines are heavy: without ``--engine`` nothing here loads,
    and the pure verbs stay torch-free (``test_load_is_torch_free``). The
    import is inside ``main`` for the same reason ``run``'s is — ``protocol/``
    never links against an engine at module scope.
    """
    if choice is None:
        return
    from causalab.cli import load_engines
    from causalab.protocol.engine import choose_engine

    engines = load_engines(choice, "cpu")
    try:
        print(f"engine    {choose_engine(list(loaded.point_documents), engines).name}")
    except ValidationError as err:
        print(f"engine    refused: {err}")


def _explain(loaded: LoadedProtocol) -> None:
    doc = loaded.point_documents[0]
    axes = loaded.expansion.axes
    print(f"digest    {loaded.document_digest}")
    if loaded.method_digest is not None:
        ref = f" ({loaded.method_ref})" if loaded.method_ref else " (inline)"
        print(f"method    {loaded.method_digest}{ref}")
    model = doc.model
    realization = f"{model.key}@{model.revision} {model.dtype or MODEL_DTYPE_DEFAULT}"
    if model.quantization is not None:
        realization += f" + {model.quantization.scheme} ({model.quantization.method})"
    print(f"model     {realization}")
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
            # readable before it runs — the mechanism stays the engine's
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
