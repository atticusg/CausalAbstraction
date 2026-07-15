"""Guard test for docs/CODEBASE.md invariant 3.

`io/` is the lowest application layer above third-party libs: it may depend only
on `neural/`, `tasks/`, `causal/`, and third-party libs. It must NOT import from
`methods/`, `analyses/`, or `runner/`. Both `methods/` and `analyses/` consume
`io/`, so an upward edge from `io/` creates a circular dependency.

This test parses every module under `causalab/io/` and fails if any of them
imports from a forbidden higher layer. It's static (no GPU, no model load).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import causalab.io

# Static structural guard — pure AST inspection, no model load (see module docstring).
pytestmark = pytest.mark.unit

# Higher layers that io/ must never import from (invariant 3).
FORBIDDEN_PREFIXES = (
    "causalab.methods",
    "causalab.analyses",
    "causalab.runner",
)

IO_DIR = Path(causalab.io.__file__).parent


def _forbidden_imports(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, module) for every absolute import into a forbidden layer."""
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (within io/) — always allowed.
            if node.level == 0 and node.module is not None:
                module = node.module
            else:
                continue
        elif isinstance(node, ast.Import):
            # `import a, b` — check each alias.
            for alias in node.names:
                if any(
                    alias.name == p or alias.name.startswith(p + ".")
                    for p in FORBIDDEN_PREFIXES
                ):
                    violations.append((node.lineno, alias.name))
            continue
        else:
            continue
        if any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
            violations.append((node.lineno, module))
    return violations


def test_io_has_no_upward_imports():
    """No file under causalab/io/ may import from methods/, analyses/, or runner/."""
    offenders: list[str] = []
    for py_file in sorted(IO_DIR.rglob("*.py")):
        for lineno, module in _forbidden_imports(py_file):
            rel = py_file.relative_to(IO_DIR.parent.parent)
            offenders.append(f"{rel}:{lineno} imports {module}")

    assert not offenders, (
        "docs/CODEBASE.md invariant 3 violated — io/ must not import from "
        "methods/, analyses/, or runner/:\n  " + "\n  ".join(offenders)
    )
