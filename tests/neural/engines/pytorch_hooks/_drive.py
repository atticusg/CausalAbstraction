"""Drive a protocol document through the reference engine in-process.

Test documents bypass dataset resolution: rows are handed to the executor
directly (the executor's own seam), while the document still parses and
validates through the real loader path — a test document is exactly as
valid as a real one."""

from __future__ import annotations

from typing import Any

from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.protocol.schema import parse_document
from causalab.protocol.validate import validate_document

from tests.protocol._docs import in_order


def bundle_loader(files: dict[str, dict[str, Any]]) -> Any:
    """A ``load_tensors`` over in-memory bundles: path -> {slot: tensor}.

    Tests that hand-build bundles carry no ``entries`` table, which is the
    same shape an external (hand-made) artifact has — selection then falls
    back to the entry keys themselves."""
    from causalab.neural.engines.pytorch_hooks.loading import TensorBundle

    def load(path: str) -> TensorBundle:
        return TensorBundle(tensors=files[path], entry_coords={})

    return load


def executor_for(
    doc_raw: dict[str, Any],
    bundle: ModelBundle,
    *,
    base_texts: list[str],
    counterfactual_texts: list[str] | None = None,
    extra_columns: dict[str, list[Any]] | None = None,
    load_tensors: Any = None,
    grad_enabled: bool = False,
) -> PointExecutor:
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(base_texts):
        row: dict[str, Any] = {"input": text}
        if counterfactual_texts is not None:
            row["counterfactual_inputs"] = [counterfactual_texts[i]]
        for column, values in (extra_columns or {}).items():
            row[column] = values[i]
        rows.append(row)
    role_rows: dict[str, list[dict[str, Any]]] = {"base": rows}
    role_fields = {"base": "input"}
    if counterfactual_texts is not None:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return PointExecutor(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=load_tensors
        or (lambda path: (_ for _ in ()).throw(KeyError(path))),
        grad_enabled=grad_enabled,
    )


def base_data_section(with_counterfactual: bool) -> dict[str, Any]:
    data: dict[str, Any] = {"base": {"dataset": "inline", "field": "input"}}
    if with_counterfactual:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data
