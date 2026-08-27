"""Static guards for the layering docs/CODEBASE.md §1 states.

Three invariants, all checked by parsing the source — no model load, no GPU:

1. **`io/` has no upward imports.** It is the lowest application layer above
   third-party libs, and the layers above it consume it, so an upward edge
   would be a cycle.
2. **`steps/` is torch-free at module level.** A step script's numerics belong
   inside its ``main``, so listing the shipped scripts (for a did-you-mean) and
   hashing one cost nothing but stdlib. The runner already uses the same idiom
   for pandas and matplotlib.
3. **`protocol/` keeps no module-level edge to `steps/` or `workflow/`.** The
   document layer links against nothing that executes; the workflow loader
   reaches the shipped-script directory through a function-local import, as
   `cli.py` does for the backend.

Invariant 2 is a *static* check. Its behavioural counterpart — that a real
``causalab validate`` of a script workflow leaves torch out of ``sys.modules``
— lives in ``tests/protocol/test_load_is_torch_free.py``, because
``tests/conftest.py`` imports torch at session scope and an in-process check
could never see the difference.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import causalab.io
import causalab.protocol
import causalab.steps

# Static structural guard — pure AST inspection, no model load (see docstring).
pytestmark = pytest.mark.unit

#: Layers `io/` must never import from. The pre-refactor entries
#: (`causalab.methods`, `causalab.analyses`, `causalab.runner`) named packages
#: that no longer exist, so the guard had stopped guarding anything.
FORBIDDEN_PREFIXES = ("causalab.steps", "causalab.workflow")

#: Numerics no module under `causalab/steps/` may import at module level.
HEAVY_MODULES = ("torch", "numpy", "pandas", "scipy", "sklearn", "safetensors")

IO_DIR = Path(causalab.io.__file__).parent
STEPS_DIR = Path(causalab.steps.__file__).parent
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


def test_steps_are_torch_free_at_module_level():
    """A step script's numerics belong inside its ``main``.

    Without this, one stray top-level ``import torch`` in a new shipped script
    would make ``causalab validate`` pay for the whole numerics stack —
    silently, since every test process has torch loaded already."""
    offenders = _offenders(STEPS_DIR, HEAVY_MODULES)
    assert not offenders, (
        "causalab/steps/ must stay importable without numerics — move the "
        "import inside the function that needs it:\n  " + "\n  ".join(offenders)
    )


def test_protocol_does_not_link_against_steps_at_module_level():
    """``protocol/`` is the backend-free document layer. It resolves and hashes
    a step script, but reaches the shipped-script directory through a
    function-local import, so ``import causalab.protocol`` still pulls in
    nothing that executes."""
    offenders = _offenders(PROTOCOL_DIR, ("causalab.steps", "causalab.workflow"))
    assert not offenders, (
        "protocol/ must not import causalab.steps at module level:\n  "
        + "\n  ".join(offenders)
    )
