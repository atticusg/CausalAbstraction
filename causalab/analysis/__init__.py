"""Numerical analysis a workflow ``script`` step can run.

Each module here is one deterministic step script — ``main(inputs, outputs)`` —
addressed from a workflow document as
``{"script": {"module": "causalab.analysis.<name>"}}``. They are ordinary
Python, importable and testable without the workflow layer, which is the point:
a document names them, but nothing about them depends on being named.

| module | what it computes | produces |
|---|---|---|
| ``fit_pca`` | a principal basis over a saved read, by full SVD | tensor + table |
| ``harvest_difference`` | a steering direction as the difference of two harvest means | tensor + table |
| ``head_stats`` | mean and spread of a metric per (layer, head) cell | table |
| ``paired_ttest`` | a two-sided paired t-test of two metric tables | table |

These are **not** the retired ``causalab/methods/`` — that was
interventions-as-Python, and interventions are documents now
(``docs/intervention_protocol.md``). What lives here is the numerics that no
intervention vocabulary can express: fits, statistics, and the operands a
later intervention consumes.

Numerics are imported inside each ``main``, so listing or hashing a script
costs nothing but stdlib (``tests/test_architecture_layering.py``).
"""

from __future__ import annotations

__all__: list[str] = []
