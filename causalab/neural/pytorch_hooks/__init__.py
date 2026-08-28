"""Deprecated import path — the package moved to
:mod:`causalab.neural.engines.pytorch_hooks`.

This shim keeps ``causalab.neural.pytorch_hooks`` (and its submodules)
importable for one deprecation beat; new code imports the ``engines`` path.
Delete it once nothing external references the old path.
"""

from __future__ import annotations

import importlib
import sys

_NEW = "causalab.neural.engines.pytorch_hooks"

_package = importlib.import_module(_NEW)
# Alias the package and its submodules so `import causalab.neural.
# pytorch_hooks.loading` keeps resolving without a second copy of any module.
sys.modules[__name__] = _package
for _sub in (
    "attention_probs",
    "encoding",
    "engine",
    "executor",
    "featurizers",
    "layout",
    "loading",
    "mechanisms",
    "metrics",
    "outputs",
    "sites",
    "train",
):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"{_NEW}.{_sub}")
