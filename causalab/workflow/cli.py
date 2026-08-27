"""The CLI verbs for **workflow** documents (docs/workflow_protocol.md §9).

Split out of ``protocol/cli.py`` so the protocol package carries no workflow
code: someone who wants only the intervention protocol imports only that.
Dispatch between the two document types is :mod:`causalab.cli`.
"""

from __future__ import annotations

import argparse
import sys

from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.loader import check_data_columns

__all__ = ["main"]


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    from causalab.protocol.loader import apply_overrides as _apply
    from causalab.protocol.loader import load_text as _load_text
    from causalab.cli import register_model_key
    from causalab.workflow.document import ProtocolStep, load_workflow

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
        raw_wf = _apply(dict(_load_text(args.document)), dict(args.parsed_set))
        workflow_dir = args.document.resolve().parent
        for step_raw in raw_wf.get("steps", {}).values():
            if (
                not isinstance(step_raw, dict)
                or step_raw.get("type") != "intervention_protocol"
            ):
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
            register_model_key(inner_raw)

    loaded = load_workflow(
        args.document.resolve(), env, overrides=dict(args.parsed_set)
    )
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
                    f"  {name}: intervention_protocol {step.document} — "
                    f"{len(inner.expansion.points)} point(s), "
                    f"{kind} digest {loaded.inner_digests[name][:16]}…"
                )
            else:
                marks = []
                if not step.is_deterministic:
                    marks.append("non-deterministic")
                if step.runtime and step.runtime.get("isolate"):
                    marks.append("isolated")
                suffix = f" [{', '.join(marks)}]" if marks else ""
                print(
                    f"  {name}: script {step.script} -> "
                    f"{', '.join(sorted(d.file for d in step.outputs.values()))}"
                    f"{suffix}"
                )
        if loaded.nondeterministic:
            # §7: explain names the steps that make a run unreplayable, so the
            # gap is visible before anyone trusts a rerun
            print(
                "not replayable: "
                + ", ".join(loaded.nondeterministic)
                + " (is_deterministic: false)"
            )
        if loaded.unchecked_paths:
            # rule 4: an absolute path is not existence-checked at load,
            # because validation and execution routinely run on different hosts
            print("unchecked absolute paths (verified at run time):")
            for item in loaded.unchecked_paths:
                print(f"  {item}")
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
        resume=getattr(args, "resume", False),
        reuse_nondeterministic=getattr(args, "reuse_nondeterministic", False),
    )
    for name, entry in sorted(result.manifest["steps"].items()):
        files = ", ".join(entry.get("files", ()))
        print(f"{entry.get('status', 'completed')} {name}: {files}")
    print(f"manifest {result.run_root / 'workflow.json'}")
    return 0
