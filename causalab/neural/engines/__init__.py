"""Execution engines — one package per implementation of the protocol's
:class:`~causalab.protocol.engine.Engine` contract.

The reference engine lives in :mod:`.pytorch_hooks`; the seam it implements is
documented in ``docs/intervention_protocol.md`` §8. Engines share the protocol
vocabulary (components, layouts, refusal tables) — nothing in here may fork it.
"""
