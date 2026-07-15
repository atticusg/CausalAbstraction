"""Direct ``is_original`` plumbing tests for ``causalab.neural.units``.

These cases were extracted from the legacy ``tests/neural/test_is_original_flag.py``
during the Wave-2 test refactor:

* The ``TokenPosition``-specific ``is_original`` cases were migrated into
  ``tests/neural/test_token_positions.py`` by PR #113.
* The ``TestComponentIndexerUnit`` / ``TestAtomicModelUnitUnit`` classes in
  ``tests/neural/test_units.py`` (PR #108) cover the ``ComponentIndexer``
  dispatch and ``AtomicModelUnit`` construction surface, but they do not
  exercise the cross-cutting ``is_original`` round-trips below:

  - flag-aware indexer receives BOTH ``True`` and ``False`` per call,
  - batched indexing propagates ``is_original`` to every element,
  - the default (unspecified) ``is_original`` does not crash a flag-aware
    indexer that declares its own default,
  - ``AtomicModelUnit.index_component`` forwards ``is_original`` through
    to the wrapped :class:`ComponentIndexer`.

These tests guard the contract that the arrow-syntax interchange path
(``var1<-var2``) relies on to select different token positions in
``base`` vs. ``source`` inputs without re-tokenising.
"""

from __future__ import annotations

import pytest

from causalab.neural.featurizer import Featurizer
from causalab.neural.units import AtomicModelUnit, ComponentIndexer

pytestmark = pytest.mark.unit


class TestComponentIndexerIsOriginal:
    """``ComponentIndexer`` must forward ``is_original`` to indexers that
    accept it and silently no-op for legacy indexers that do not."""

    def test_indexer_with_is_original_parameter(self) -> None:
        """Flag-aware indexer receives both ``True`` and ``False``."""

        def position_indexer(input, is_original=True):
            return [0, 1, 2] if is_original else [3, 4, 5]

        indexer = ComponentIndexer(position_indexer, id="test_indexer")
        assert indexer.index({"text": "test"}, is_original=True) == [0, 1, 2]
        assert indexer.index({"text": "test"}, is_original=False) == [3, 4, 5]

    def test_indexer_without_is_original_parameter(self) -> None:
        """Legacy indexer (no ``is_original`` arg) still works under both
        flag values: signature detection sees no flag, so it's called without
        one (no ``TypeError``-swallowing fallback involved)."""

        def old_indexer(input):
            return [0, 1, 2]

        indexer = ComponentIndexer(old_indexer, id="old_indexer")
        assert indexer.index({"text": "test"}, is_original=True) == [0, 1, 2]
        assert indexer.index({"text": "test"}, is_original=False) == [0, 1, 2]

    def test_batch_indexing_with_is_original(self) -> None:
        """``batch=True`` must propagate ``is_original`` to every element."""

        def position_indexer(input, is_original=True):
            base_pos = input.get("pos", 0)
            return [base_pos] if is_original else [base_pos + 10]

        indexer = ComponentIndexer(position_indexer, id="batch_indexer")
        batch = [{"pos": 0}, {"pos": 1}, {"pos": 2}]
        assert indexer.index(batch, batch=True, is_original=True) == [[0], [1], [2]]
        assert indexer.index(batch, batch=True, is_original=False) == [[10], [11], [12]]

    def test_default_is_original_omits_kwarg(self) -> None:
        """When ``is_original`` is not passed, the wrapper must NOT inject
        it; the indexer's own default takes effect."""

        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="default_test")
        # No is_original passed → indexer's default (True) wins.
        assert indexer.index({"text": "test"}) == [0]


class TestAtomicModelUnitIsOriginal:
    """``AtomicModelUnit.index_component`` must forward ``is_original`` to
    the wrapped :class:`ComponentIndexer`."""

    def test_index_component_passes_is_original(self) -> None:
        def position_indexer(input, is_original=True):
            return [0] if is_original else [1]

        indexer = ComponentIndexer(position_indexer, id="test")
        unit = AtomicModelUnit(
            layer=5,
            component_type="block_input",
            indices_func=indexer,
            unit="pos",
            featurizer=Featurizer(),
            id="test_unit",
        )
        assert unit.index_component({"text": "test"}, is_original=True) == [0]
        assert unit.index_component({"text": "test"}, is_original=False) == [1]
