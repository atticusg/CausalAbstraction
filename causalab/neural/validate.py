"""nnterp model-load validation gate for model onboarding / CI.

Two layers of validation, cheapest first:

* :func:`validate_model_load` — the per-model onboarding gate. Constructs a
  ``nnterp.StandardizedTransformer(..., check_renaming=True)``, which runs
  nnterp's load-time renaming + IO-shape checks (``check_model_renaming`` /
  ``check_io``): the standardized module tree resolves, layer outputs are
  ``(batch, seq, hidden)``, and embeddings -> layers -> logits shapes line up.
  Cheap, CPU-friendly, and the check F1 (#392) relied on to validate
  ``tiny-random`` / ``gpt2`` / ``Llama-3.1-8B`` / ``chat-coherent``.

* :func:`run_nnterp_tests` — the deeper CI layer. Wraps nnterp's own pytest
  runner (``python -m nnterp run_tests``) over a set of checkpoints. Heavier
  (downloads models, runs nnterp's ~5 test modules), so it is opt-in.

Alongside the load gate, each passing report records whether the checkpoint
supports the zero-compute ``scan()`` preflight (CAP5, #458 —
:func:`causalab.neural.preflight.check_scan_support`): nnterp's checks fall
back ``scan → trace`` under ``allow_dispatch``, so only this explicit probe
distinguishes "scan-clean" from "scan unsupported for this model".

``python -m causalab.neural.validate --model-names ...`` runs the gate over a
list of checkpoints and exits non-zero if any fails, with ``--run-tests`` to
also invoke the deeper layer.

Why nnterp is pinned to a git rev (see ``pyproject.toml`` ``[tool.uv.sources]``):
the nnterp PyPI wheels up to 1.3.0 omit ``nnterp/data/`` (their ``package-data``
globs ``tests/*`` but not ``data/*``), so ``python -m nnterp run_tests`` crashed
on import (``ModuleNotFoundError: No module named 'nnterp.data'``). The fix is
merged upstream (ndif-team/nnterp#49); the pin is the upstream merge commit,
awaiting a PyPI release that ships it (#391 / #415).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError

from causalab.neural.pipeline import assert_architecture_supported
from causalab.neural.preflight import check_scan_support
from causalab.neural.site import hf_text_config

logger = logging.getLogger(__name__)

__all__ = [
    "ModelValidationReport",
    "ModelValidationError",
    "validate_model_load",
    "run_nnterp_tests",
]


class ModelValidationError(RuntimeError):
    """Raised by :meth:`ModelValidationReport.raise_if_failed` on a failed gate."""


@dataclasses.dataclass
class ModelValidationReport:
    """Outcome of :func:`validate_model_load` for a single checkpoint.

    On success (``ok=True``) the introspection fields carry the standardized
    facts nnterp exposes (``num_layers`` / ``num_heads`` / ``hidden_size`` /
    ``vocab_size``) plus ``head_dim`` / ``num_kv_heads`` / ``decoupled_head_dim``
    read best-effort from the HF config — the columns of F1's load-spike table.
    On failure (``ok=False``) only ``error`` is populated.

    ``scan_supported`` records whether a bare fake-mode forward
    (``model.scan()``) runs on the checkpoint — whether the CAP5 plan
    preflight (:mod:`causalab.neural.preflight`) can validate plans against
    it. nnterp's own load checks fall back ``scan → trace`` under
    ``allow_dispatch``, so a passing load gate does **not** imply scan
    support; this field makes the distinction explicit (``scan_error``
    carries the cause when unsupported).
    """

    model_name: str
    ok: bool
    error: str | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    head_dim: int | None = None
    num_kv_heads: int | None = None
    decoupled_head_dim: bool | None = None
    scan_supported: bool | None = None
    scan_error: str | None = None

    def raise_if_failed(self) -> None:
        """Raise :class:`ModelValidationError` if this checkpoint failed the gate."""
        if not self.ok:
            raise ModelValidationError(
                f"{self.model_name} failed nnterp load-time validation: {self.error}"
            )

    def summary_row(self) -> str:
        """One-line human-readable summary for the CLI table."""
        if not self.ok:
            return f"[FAIL] {self.model_name}: {self.error}"
        decoupled = " decoupled-head_dim" if self.decoupled_head_dim else ""
        scan = ""
        if self.scan_supported is not None:
            scan = " scan=ok" if self.scan_supported else " scan=UNSUPPORTED"
        return (
            f"[ OK ] {self.model_name}: "
            f"layers={self.num_layers} heads={self.num_heads} "
            f"kv_heads={self.num_kv_heads} hidden={self.hidden_size} "
            f"head_dim={self.head_dim} vocab={self.vocab_size}{decoupled}{scan}"
        )


def validate_model_load(
    model_name: str,
    *,
    allow_dispatch: bool = True,
    rename_config: Any | None = None,
    **load_kwargs: Any,
) -> ModelValidationReport:
    """Run nnterp's load-time validation gate on ``model_name``.

    Runs the same architecture preflight as ``LMPipeline``
    (:func:`causalab.neural.pipeline.assert_architecture_supported` — a clear
    verdict, naming the transformers-version cause, for model types the
    installed transformers cannot load), then constructs a
    :class:`~nnterp.StandardizedTransformer` with ``check_renaming=True``;
    nnterp raises :class:`~nnterp.rename_utils.RenamingError`
    (or a plain ``ValueError`` for a missing config key) when the module tree or
    IO shapes don't standardize. Those are caught and reported as ``ok=False``;
    any other exception propagates (a real bug, not a validation verdict).

    Args:
        model_name: HF checkpoint id (or local path).
        allow_dispatch: passed through to nnterp; when True its IO check may fall
            back from ``scan`` to a real ``trace`` forward.
        rename_config: optional ``nnterp.RenameConfig`` for architectures not in
            nnterp's default family set.
        **load_kwargs: forwarded to ``StandardizedTransformer`` (e.g. ``device_map``).

    Returns:
        A :class:`ModelValidationReport`.
    """
    try:
        assert_architecture_supported(model_name, token=load_kwargs.get("token"))
        model = StandardizedTransformer(
            model_name,
            check_renaming=True,
            allow_dispatch=allow_dispatch,
            rename_config=rename_config,
            **load_kwargs,
        )
    except (RenamingError, ValueError) as exc:
        logger.info("nnterp validation failed for %s: %s", model_name, exc)
        return ModelValidationReport(
            model_name=model_name, ok=False, error=f"{type(exc).__name__}: {exc}"
        )

    try:
        num_heads = model.num_heads
        hidden_size = model.hidden_size
        # Read the GQA/head fields nnterp does not standardize from the *text*
        # config (`hf_text_config` follows nnterp's ``text_config`` nesting
        # rule) — the raw top-level config silently reports the wrong values
        # on nesting models (e.g. Gemma3).
        hf_config = hf_text_config(model)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None and hidden_size and num_heads:
            head_dim = hidden_size // num_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
        decoupled = (
            head_dim is not None
            and num_heads
            and hidden_size
            and head_dim != hidden_size // num_heads
        )
        # The CAP5 preflight column: nnterp's own checks may have passed via
        # the trace fallback, so record the scan verdict explicitly.
        scan_error = check_scan_support(model)
        return ModelValidationReport(
            model_name=model_name,
            ok=True,
            num_layers=model.num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=model.vocab_size,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            decoupled_head_dim=bool(decoupled),
            scan_supported=scan_error is None,
            scan_error=scan_error,
        )
    finally:
        # Release the model / device memory; the gate only needs the verdict.
        del model


def run_nnterp_tests(
    model_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    *,
    extra_pytest_args: Sequence[str] | None = None,
) -> int:
    """Run nnterp's own pytest suite (``python -m nnterp run_tests``) over checkpoints.

    The deeper CI layer: exercises nnterp's ~5 test modules against the given
    models. Runs in a scratch working directory so pytest does **not** discover
    causalab's ``pyproject.toml`` / ``conftest.py`` (which enforce causalab's tier
    markers and would otherwise error on nnterp's own tests).

    Args:
        model_names: checkpoints to test (nnterp's ``--model-names``); all
            available models if None.
        class_names: toy-model classes to test (nnterp's ``--class-names``).
        extra_pytest_args: extra flags forwarded to the underlying pytest.

    Returns:
        The subprocess return code (0 = all nnterp tests passed).
    """
    cmd = [sys.executable, "-m", "nnterp", "run_tests"]
    if model_names:
        cmd += ["--model-names", *model_names]
    if class_names:
        cmd += ["--class-names", *class_names]
    if extra_pytest_args:
        cmd += list(extra_pytest_args)
    with tempfile.TemporaryDirectory(prefix="nnterp-run-tests-") as neutral_cwd:
        logger.info("running nnterp run_tests: %s (cwd=%s)", " ".join(cmd), neutral_cwd)
        return subprocess.run(cmd, cwd=neutral_cwd, check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.neural.validate",
        description="Validate that nnterp can standardize model checkpoints "
        "(model-onboarding / CI gate).",
    )
    parser.add_argument(
        "--model-names",
        "-m",
        nargs="+",
        required=True,
        help="HF checkpoint id(s) to validate.",
    )
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Disable nnterp's trace-based IO-check fallback (allow_dispatch=False).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="After the load gate passes, also run `python -m nnterp run_tests` "
        "over the checkpoints (heavier: downloads models, runs nnterp's suite).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = all checkpoints valid)."""
    args = _build_parser().parse_args(argv)
    reports = [
        validate_model_load(name, allow_dispatch=not args.no_dispatch)
        for name in args.model_names
    ]
    for report in reports:
        print(report.summary_row())
    exit_code = 0 if all(r.ok for r in reports) else 1
    if args.run_tests and exit_code == 0:
        print("--- nnterp run_tests ---")
        exit_code = run_nnterp_tests(args.model_names) or exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
