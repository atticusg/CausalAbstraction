"""The ``causalab`` CLI: ``run · validate · explain · digest``.

One entry point over **two document types**. Argument parsing and the
resolution environment are shared; the verbs themselves are not:

* an **intervention protocol** document → :mod:`causalab.protocol.cli`
* a **workflow** document → :mod:`causalab.workflow.cli`

Dispatch is on the document's ``steps`` section (workflow spec §1). Keeping it
here is what lets ``protocol/`` carry no workflow code and ``workflow/`` depend
on ``protocol/`` one way only — so the intervention protocol is usable on its
own, which is the point of having two packages.

``run`` needs an execution engine; the reference engine
(:mod:`causalab.neural.engines.pytorch_hooks`) is imported lazily by whichever half
needs it, so the pure verbs stay torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import PRECISION_DTYPES

__all__ = ["ensure_model_registered", "main", "register_model_key"]


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


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """``--set`` overrides plus the ``--dtype`` shorthand, which is one of
    them: dtype belongs to the document, so the only way to change it from
    the command line is the way every other field changes (§9)."""
    overrides = _parse_set(args.set)
    dtype = getattr(args, "dtype", None)
    if dtype is None:
        return overrides
    already = overrides.get("model.dtype")
    if already is not None and already != dtype:
        raise SystemExit(
            f"--dtype {dtype} contradicts --set model.dtype={already} — "
            "they set the same field"
        )
    overrides["model.dtype"] = dtype
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
                "--out",
                type=Path,
                required=True,
                help="run output directory; for a workflow, the ROOT under "
                "which the document's own output_dir is created (§1.1)",
            )
            p.add_argument(
                "--resume",
                action="store_true",
                help="skip a step whose outputs exist with a matching stamped "
                "digest (workflow documents only)",
            )
            p.add_argument(
                "--reuse-nondeterministic",
                action="store_true",
                help="with --resume, also reuse steps declaring "
                "is_deterministic: false",
            )
            p.add_argument(
                "--device",
                default="cpu",
                help="torch device string for the reference engine "
                "(cpu, cuda, cuda:1, mps)",
            )
            p.add_argument(
                "--dtype",
                choices=PRECISION_DTYPES,
                default=None,
                help="shorthand for --set model.dtype=… — precision is a "
                "document fact (§2.1), so an override enters the digest and "
                "the record never lies about what produced the numbers",
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


def _env(args: argparse.Namespace) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=args.data_root),
        artifacts=FileArtifacts(root=args.artifacts_root),
    )


def ensure_model_registered(args: argparse.Namespace) -> None:
    """The run verb touches the model anyway, so an unregistered key is
    resolved from its HF config and registered before canonicalization —
    the pure verbs stay registry-only so digests never depend on the
    network."""
    from causalab.protocol.loader import apply_overrides, flatten, load_text

    # flatten first: in a split document the model lives in the `application`
    # half (§1.1), and `--set model.key=…` addresses the composition
    raw = dict(load_text(args.document))
    try:
        raw, _, _ = flatten(raw, base_dir=args.document.resolve().parent)
    except ProtocolError:
        return  # a malformed document refuses properly in the real load
    register_model_key(apply_overrides(raw, dict(args.parsed_set)))


def register_model_key(raw: dict[str, Any]) -> None:
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
    """Parse, build the environment, and dispatch on document type."""
    args = _build_parser().parse_args(argv)
    args.parsed_set = _overrides(args)
    env = _env(args)
    try:
        from causalab.protocol.loader import load_text
        from causalab.workflow.document import is_workflow

        if is_workflow(load_text(args.document)):
            from causalab.workflow import cli as workflow_cli

            return workflow_cli.main(args, env)
        from causalab.protocol import cli as protocol_cli

        return protocol_cli.main(args, env)
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
