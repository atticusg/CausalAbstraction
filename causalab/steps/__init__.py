"""Step scripts: the Python a workflow ``script`` step runs.

A script step is ``inputs → one Python script → declared outputs``
(docs/workflow_protocol.md §2.3). This package holds two things:

- :mod:`causalab.steps.io` — reading inputs and writing outputs: JSON metric
  tables, safetensors bundles with ``slot``/``entry`` addressing, and the
  identity a tensor output inherits. A script imports it; nothing forces it to.
- ``builtin/`` — the scripts causalab ships, reached from a document as
  ``causalab:<name>``. ``select`` and ``plot`` live here, so the reductions
  every pipeline wants stay one line to author even though they are no longer
  step *types*.

The contract is one function:

.. code-block:: python

    def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None: ...

The script creates every declared output file; the runner verifies they exist,
checks a declared table's columns, and stamps ArtifactIdentity on safetensors
outputs so a script cannot forget provenance.
"""

from __future__ import annotations

__all__: list[str] = []
