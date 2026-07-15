"""Runner completion helper for smoke / numerical tier tests.

Provides ``assert_runner_completed`` — a small assertion bundle used by
end-to-end runner tests to confirm that a Hydra-composed runner produced its
expected artifacts under the supplied output directory.

The helper deliberately uses ``assert`` statements (not ``raise``) so that
test failures point at the specific check that failed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _dir_listing(p: Path) -> str:
    return (
        repr(sorted(q.name for q in p.iterdir()))
        if p.is_dir()
        else "<output_dir does not exist>"
    )


def assert_runner_completed(
    cfg: Any,
    output_dir: Path | str,
    expected_artifacts: Iterable[str] = ("metrics.json",),
) -> None:
    """Assert that a runner produced its expected artifacts.

    For each ``name`` in ``expected_artifacts``:
      - asserts the file exists under ``output_dir`` (one assertion per artifact);
      - if the name ends in ``.json``, parses the file with ``json.load`` and
        asserts the parse succeeds (one additional assertion per JSON artifact);
      - otherwise asserts the file is non-empty (one additional assertion per file).

    Parameters
    ----------
    cfg:
        The composed runner config (e.g. a Hydra ``DictConfig``). **Currently
        unused at the helper level** — accepted for forward-compatibility so
        later tiers may look up runner-specific expected artifacts (Phase 7
        goldens, per-runner determinism contracts) without changing the call
        sites added in Phase 2/3. Phase 2 enforces nothing via ``cfg``.
    output_dir:
        Directory the runner wrote its artifacts to. Resolved as a ``Path``.
    expected_artifacts:
        Iterable of artifact filenames expected directly under ``output_dir``.
        Defaults to ``("metrics.json",)``.
    """
    del cfg  # forward-compat parameter; intentionally unused in Phase 2

    output_dir = Path(output_dir)
    for name in expected_artifacts:
        artifact = output_dir / name
        assert artifact.is_file(), (
            f"expected runner artifact missing: {artifact} "
            f"(output_dir contents: {_dir_listing(output_dir)})"
        )
        if name.endswith(".json"):
            try:
                with artifact.open() as f:
                    json.load(f)
                parse_ok = True
                parse_err: str | None = None
            except json.JSONDecodeError as exc:
                parse_ok = False
                parse_err = str(exc)
            assert parse_ok, (
                f"runner artifact {artifact} is not valid JSON: {parse_err}"
            )
        else:
            assert artifact.stat().st_size > 0, (
                f"runner artifact {artifact} exists but is empty"
            )
