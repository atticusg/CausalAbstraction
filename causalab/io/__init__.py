"""Artifact I/O and plotting primitives.

This package is the single source of truth for code that touches disk:
- ``artifacts``: JSON / safetensors / pickle save/load, metadata, intervention
  and training result writers.
- ``plots``: shared figure rendering helpers.
- ``counterfactuals``: counterfactual dataset save/load.
- ``configs``: runner config save/load for notebook workflows.
- ``pipelines``: LMPipeline and analysis-result loaders.

Dependency rule: ``causalab.io`` imports only from ``causalab.neural`` (for
typing) and third-party libs. It must not import from ``causalab.methods``,
``causalab.analyses``, or ``causalab.runner``.
"""
