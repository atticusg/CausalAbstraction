"""Tests for ``causalab.io.centroids`` categorical-target handling (#260).

The geometry/centroid pipeline turns causal parameters into numeric
coordinates. A categorical (string) target with no embedding used to hit a bare
``float()`` and surface the opaque ``could not convert string to float: 'A'`` —
crashing the spline-fitting path (activation_manifold) and producing a cryptic
skip warning in the viz paths (subspace / output_manifold).

These tests pin the fix: the chokepoint raises a clear
:class:`CategoricalParameterError` (a ``ValueError`` subclass, so existing
``except`` viz handlers still skip gracefully) naming the variable and the
``EMBEDDINGS`` remedy; the embedding escape hatch still works; a model-side
embedding is never silently ignored when a partial ``embeddings`` dict is
passed; numeric extraction is unchanged; and the subspace viz wrapper does not
hard-crash on a categorical target.
"""

from __future__ import annotations

import pytest
import torch

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.io.centroids import (
    CategoricalParameterError,
    coerce_param_to_float,
    extract_parameters_from_dataset,
)
from tests._helpers.tiny import tiny_chain_model

pytestmark = pytest.mark.unit


def _ord_embed(v: str) -> list[float]:
    """Minimal ordinal embedding mapping a single-char label to one float."""
    return [float(ord(v))]


def _categorical_model(embeddings: dict | None = None) -> CausalModel:
    """A 1-input ``letter`` model whose target is a categorical string."""
    values = {"letter": ["A", "B", "C"], "raw_input": None, "raw_output": None}
    mechanisms = {
        "letter": input_var(["A", "B", "C"]),
        "raw_input": Mechanism(
            parents=["letter"], compute=lambda t: f"L={t['letter']}"
        ),
        "raw_output": Mechanism(parents=["letter"], compute=lambda t: t["letter"]),
    }
    return CausalModel(
        mechanisms, values, id="categorical_letter", embeddings=embeddings
    )


def _dataset(model: CausalModel, var: str, vals: list) -> list[dict]:
    return [
        {"input": model.new_trace({var: v}), "counterfactual_inputs": []} for v in vals
    ]


def _tuple_categorical_dataset() -> list[dict]:
    """Dataset whose target is a tuple of strings (exercises the tuple branch)."""
    pairs = [("A", "B"), ("C", "D")]
    values = {"pair": pairs, "raw_input": None, "raw_output": None}
    mechanisms = {
        "pair": input_var(pairs),
        "raw_input": Mechanism(parents=["pair"], compute=lambda t: str(t["pair"])),
        "raw_output": Mechanism(parents=["pair"], compute=lambda t: str(t["pair"])),
    }
    model = CausalModel(mechanisms, values, id="pair_categorical")
    return _dataset(model, "pair", pairs)


class TestCoerceParamToFloat:
    """The shared scalar conversion helper used at every numeric chokepoint."""

    def test_numeric_values_convert(self):
        assert coerce_param_to_float("x", 2) == 2.0
        assert coerce_param_to_float("x", "3.5") == 3.5

    def test_categorical_raises_clear_error(self):
        with pytest.raises(CategoricalParameterError) as exc:
            coerce_param_to_float("ease_label", "A")
        msg = str(exc.value)
        # Names the variable, the offending value, and the EMBEDDINGS remedy —
        # not the raw "could not convert string to float".
        assert "ease_label" in msg
        assert "'A'" in msg
        assert "EMBEDDINGS" in msg
        assert "could not convert string to float" not in msg


class TestExtractParametersCategorical:
    """Categorical targets at the ``extract_parameters_from_dataset`` chokepoint."""

    def test_scalar_categorical_raises_clear_error(self):
        ds = _dataset(_categorical_model(), "letter", ["A", "B", "C"])
        with pytest.raises(CategoricalParameterError) as exc:
            extract_parameters_from_dataset(ds)
        msg = str(exc.value)
        assert "letter" in msg
        assert "EMBEDDINGS" in msg

    def test_tuple_categorical_raises_clear_error(self):
        ds = _tuple_categorical_dataset()
        with pytest.raises(CategoricalParameterError) as exc:
            extract_parameters_from_dataset(ds)
        assert "pair" in str(exc.value)

    def test_explicit_embedding_bypasses_guard(self):
        ds = _dataset(_categorical_model(), "letter", ["A", "B", "C"])
        out = extract_parameters_from_dataset(ds, embeddings={"letter": _ord_embed})
        assert set(out.keys()) == {"letter"}
        assert torch.allclose(out["letter"], torch.tensor([65.0, 66.0, 67.0]))

    def test_model_embedding_used_when_no_explicit_dict(self):
        model = _categorical_model(embeddings={"letter": _ord_embed})
        ds = _dataset(model, "letter", ["A", "B"])
        out = extract_parameters_from_dataset(ds, causal_model=model)
        assert torch.allclose(out["letter"], torch.tensor([65.0, 66.0]))

    def test_model_embedding_not_shadowed_by_partial_explicit_dict(self):
        # Footgun guard: a partial explicit dict (missing the target) must not
        # hide a model-side embedding — they are merged, explicit wins.
        model = _categorical_model(embeddings={"letter": _ord_embed})
        ds = _dataset(model, "letter", ["A", "B"])
        out = extract_parameters_from_dataset(
            ds, embeddings={"unrelated": _ord_embed}, causal_model=model
        )
        assert torch.allclose(out["letter"], torch.tensor([65.0, 66.0]))


class TestNumericUnaffected:
    """Numeric extraction is unchanged by the guard (regression)."""

    def test_numeric_target_extracts_tensors(self):
        model = tiny_chain_model()
        ds = [
            {"input": model.new_trace({"A": 0}), "counterfactual_inputs": []},
            {"input": model.new_trace({"A": 1}), "counterfactual_inputs": []},
        ]
        out = extract_parameters_from_dataset(ds)
        assert "A" in out
        assert torch.allclose(out["A"], torch.tensor([0.0, 1.0]))
        assert out["A"].dtype == torch.float32


class TestErrorContract:
    """``CategoricalParameterError`` must stay catchable by existing handlers."""

    def test_is_valueerror_subclass(self):
        assert issubclass(CategoricalParameterError, ValueError)

    def test_caught_by_broad_handlers(self):
        # The viz paths skip via ``except Exception`` / ``except ValueError``;
        # the dedicated type must remain catchable by both.
        for catch in (Exception, ValueError):
            try:
                coerce_param_to_float("x", "A")
            except catch:
                caught = True
            else:  # pragma: no cover - guard
                caught = False
            assert caught


class TestVizDoesNotHardCrash:
    """Subspace viz wrapper degrades to a warning on a categorical target."""
