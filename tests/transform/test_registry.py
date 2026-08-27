"""The op registry's refusals, and the invariants every record must hold.

The registry is a closed vocabulary like the metric kinds and the plot
kinds, so it is held to the same standard: an unknown value is refused at
load with a suggestion, and the *record* is checked as strictly as the
document that names it.
"""

from __future__ import annotations

import pytest

from causalab.transform.registry import lookup, op_ids
from causalab.transform.schema import (
    COLUMN_DTYPES,
    Bool,
    Float,
    Int,
    Str,
    Table,
    Tensor,
    TransformError,
    validate_params,
)

pytestmark = pytest.mark.unit


class TestLookup:
    def test_every_seed_op_is_registered(self) -> None:
        assert op_ids() == ("fit_pca@1", "head_stats@1", "paired_ttest@1")

    def test_a_known_op_resolves(self) -> None:
        op = lookup("fit_pca@1")
        assert op.name == "fit_pca" and op.version == 1
        assert op.id == "fit_pca@1"

    def test_unknown_op_suggests(self) -> None:
        with pytest.raises(TransformError) as err:
            lookup("fit_pcaa@1")
        assert "unknown op 'fit_pcaa'" in str(err.value)
        assert "fit_pca" in str(err.value)  # the suggestion

    def test_unknown_version_of_a_known_op_says_which_exist(self) -> None:
        """A did-you-mean over the whole id would answer 'fit_pca@1' without
        saying why; the version list is what the author needs."""
        with pytest.raises(TransformError) as err:
            lookup("fit_pca@7")
        assert "has no version 7" in str(err.value)
        assert "version 1" in str(err.value)

    def test_an_unversioned_op_is_refused(self) -> None:
        with pytest.raises(TransformError) as err:
            lookup("fit_pca")
        assert "name@version" in str(err.value)

    def test_a_non_numeric_version_is_refused(self) -> None:
        with pytest.raises(TransformError) as err:
            lookup("fit_pca@one")
        assert "not an integer" in str(err.value)


class TestRecords:
    """Invariants that make a record safe to validate a document against."""

    @pytest.mark.parametrize("op_id", op_ids())
    def test_output_tables_declare_typed_columns(self, op_id: str) -> None:
        for slot, decl in lookup(op_id).outputs.items():
            if isinstance(decl, Table):
                assert decl.columns, f"{op_id}:{slot} declares no columns"
                for column, dtype in decl.columns.items():
                    assert dtype in COLUMN_DTYPES, f"{op_id}:{slot}.{column}"

    @pytest.mark.parametrize("op_id", op_ids())
    def test_slots_have_the_right_file_suffixes(self, op_id: str) -> None:
        op = lookup(op_id)
        for decl in (*op.inputs.values(), *op.outputs.values()):
            expected = ".json" if isinstance(decl, Table) else ".safetensors"
            assert decl.suffix == expected

    @pytest.mark.parametrize("op_id", op_ids())
    def test_identity_mappings_name_declared_params(self, op_id: str) -> None:
        op = lookup(op_id)
        for param in op.identity_from_params.values():
            assert param in op.params

    @pytest.mark.parametrize("op_id", op_ids())
    def test_the_callable_takes_inputs_and_params(self, op_id: str) -> None:
        """The runner calls every op the same way, so the signature is part of
        the contract, not a per-op convention."""
        import inspect

        signature = inspect.signature(lookup(op_id).fn)
        assert set(signature.parameters) == {"inputs", "params"}
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )

    @pytest.mark.parametrize("op_id", op_ids())
    def test_a_tensor_output_can_prove_its_provenance(self, op_id: str) -> None:
        """A tensor a protocol step may load has to be identity-stamped, and
        the only identity an op can add beyond what it inherits is from its
        params — so an op whose tensor output is meant to be loaded declares
        that mapping. `fit_pca@1` is the case; assert the rule holds where it
        applies rather than asserting it vacuously everywhere."""
        op = lookup(op_id)
        writes_tensor = any(isinstance(d, Tensor) for d in op.outputs.values())
        reads_tensor = any(isinstance(d, Tensor) for d in op.inputs.values())
        if writes_tensor:
            assert reads_tensor or op.identity_from_params, (
                f"{op_id} writes a tensor but can neither inherit nor declare "
                "an identity for it"
            )


class TestParamValidation:
    def test_defaults_are_materialized(self) -> None:
        schema = {"k": Int(default=4), "mode": Str(default="a")}
        assert validate_params(schema, {}, path="p") == {"k": 4, "mode": "a"}

    def test_a_required_parameter_is_required(self) -> None:
        with pytest.raises(TransformError) as err:
            validate_params({"k": Int(min=1)}, {}, path="p")
        assert "missing required parameter 'k'" in str(err.value)

    def test_unknown_parameter_suggests(self) -> None:
        with pytest.raises(TransformError) as err:
            validate_params({"seed": Int(default=0)}, {"seeed": 1}, path="p")
        assert "unknown parameter 'seeed'" in str(err.value)
        assert "seed" in str(err.value)

    def test_wrong_type_is_refused(self) -> None:
        with pytest.raises(TransformError) as err:
            validate_params({"k": Int(min=1)}, {"k": "8"}, path="p")
        assert "is an integer" in str(err.value)

    def test_a_boolean_is_not_an_integer(self) -> None:
        """`bool` subclasses `int` in Python, so `{"k": true}` would silently
        mean k = 1 without this check."""
        with pytest.raises(TransformError) as err:
            validate_params({"k": Int(min=1)}, {"k": True}, path="p")
        assert "boolean" in str(err.value)

    def test_bounds_are_enforced(self) -> None:
        with pytest.raises(TransformError):
            validate_params({"k": Int(min=1)}, {"k": 0}, path="p")
        with pytest.raises(TransformError):
            validate_params({"k": Int(default=1, max=3)}, {"k": 4}, path="p")

    def test_choices_are_a_closed_enum(self) -> None:
        schema = {"mode": Str(default="mean", choices=("mean", "median"))}
        with pytest.raises(TransformError) as err:
            validate_params(schema, {"mode": "meen"}, path="p")
        assert "mean" in str(err.value)  # the suggestion

    def test_an_int_is_accepted_for_a_float_and_normalized(self) -> None:
        out = validate_params({"x": Float(default=0.0)}, {"x": 2}, path="p")
        assert out == {"x": 2.0} and isinstance(out["x"], float)

    def test_a_boolean_parameter_takes_a_boolean(self) -> None:
        assert validate_params({"b": Bool(default=False)}, {"b": True}, path="p") == {
            "b": True
        }
        with pytest.raises(TransformError):
            validate_params({"b": Bool(default=False)}, {"b": 1}, path="p")
