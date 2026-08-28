"""The nnsight tracing engine: intervention protocols over nnsight 0.8.

The second implementation of the protocol's engine contract (plan
``causalab-nnsight-engine-plan`` §4): the same shared services, site map and
write math as the reference engine, executed through ``model.trace`` on
envoys instead of registered hooks. Its reason to exist is the surface
*beyond* module boundaries — the ``.source`` interiors of the Qwen3.6
hookpoints (DeltaNet state, expert interiors, attention internals) — which
arrive phase by phase (N5–N8); this skeleton serves the module-boundary
vocabulary and proves itself against the reference engine's answers.

Requires the ``nnsight`` extra (``pip install 'causalab[nnsight]'``), pinned
to the 0.8-branch rev verified on the real 35B.
"""

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine

__all__ = ["NnsightEngine"]
