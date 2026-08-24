"""Static guards for the layering docs/CODEBASE.md §1 states.

Three invariants, all checked by parsing the source — no model load, no GPU:

1. **`io/` has no upward imports.** It is the lowest application layer above
   third-party libs, and the layers above it consume it, so an upward edge
   would be a cycle.
2. **`transform/` is torch-free at module level.** The op *records* are what
   the pure CLI verbs read to refuse a bad document, so importing the registry
   must not drag in numerics. Ops keep their heavy imports inside the function
   body — the idiom the runner already uses for pandas and matplotlib.
3. **`protocol/` keeps no module-level edge to `transform/`.** The document
   layer links against nothing that executes; the workflow loader reaches the
   registry through a function-local import, as `cli.py` does for the backend.

Invariant 2 is a *static* check. Its behavioural counterpart — that a real
``causalab validate`` of a transform workflow leaves torch out of
``sys.modules`` — lives in ``tests/transform/test_load_is_torch_free.py``,
because ``tests/conftest.py`` imports torch at session scope and an in-process
check could never see the difference.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import causalab.io
import causalab.protocol
import causalab.transform

# Static structural guard — pure AST inspection, no model load (see docstring).
pytestmark = pytest.mark.unit

#: Layers `io/` must never import from. The pre-refactor entries
#: (`causalab.methods`, `causalab.analyses`, `causalab.runner`) named packages
#: that no longer exist, so the guard had stopped guarding anything.
FORBIDDEN_PREFIXES = ("causalab.transform", "causalab.workflow")

#: Numerics no module under `causalab/transform/` may import at module level.
HEAVY_MODULES = ("torch", "numpy", "pandas", "scipy", "sklearn", "safetensors")

IO_DIR = Path(causalab.io.__file__).parent
TRANSFORM_DIR = Path(causalab.transform.__file__).parent
PROTOCOL_DIR = Path(causalab.protocol.__file__).parent


def _module_level_imports(path: Path) -> list[tuple[int, str]]:
    """Every absolute import executed at *module* scope.

    Imports nested inside a function body are deliberately not reported:
    deferring a heavy import into the call is exactly the discipline these
    guards exist to permit."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nested: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                nested.add(id(inner))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in nested:
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
    return found


def _matches(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _offenders(root: Path, prefixes: tuple[str, ...]) -> list[str]:
    return [
        f"{path.relative_to(root.parent.parent)}:{lineno} imports {module}"
        for path in sorted(root.rglob("*.py"))
        for lineno, module in _module_level_imports(path)
        if _matches(module, prefixes)
    ]


def test_io_has_no_upward_imports():
    offenders = _offenders(IO_DIR, FORBIDDEN_PREFIXES)
    assert not offenders, (
        "docs/CODEBASE.md invariant 3 violated — io/ must not import from a "
        "higher layer:\n  " + "\n  ".join(offenders)
    )


def test_transform_is_torch_free_at_module_level():
    """An op's numerics belong inside its function body.

    Without this, one stray top-level ``import torch`` in a new op would make
    ``causalab validate`` pay for the whole numerics stack — silently, since
    every test process has torch loaded already."""
    offenders = _offenders(TRANSFORM_DIR, HEAVY_MODULES)
    assert not offenders, (
        "causalab/transform/ must stay importable without numerics — move the "
        "import inside the function that needs it:\n  " + "\n  ".join(offenders)
    )


def test_protocol_does_not_link_against_transform_at_module_level():
    """``protocol/`` is the backend-free document layer. It *reads* op records
    at load, but through a function-local import, so ``import causalab.protocol``
    still pulls in nothing that executes."""
    offenders = _offenders(PROTOCOL_DIR, ("causalab.transform",))
    assert not offenders, (
        "protocol/ must not import causalab.transform at module level:\n  "
        + "\n  ".join(offenders)
    )
