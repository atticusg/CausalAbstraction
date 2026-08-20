"""Task packages → dataset tables (spec §2.2): the column vocabulary, the
determinism the content digest depends on, and the loud failures.

LM-free: the serializer never touches a model or a tokenizer — that is the
property that lets a document's digest be computed without either.
"""

from __future__ import annotations

import json

import pytest

from causalab.protocol.resolve import FileDatasets
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.tasks.serialize import (
    RESERVED_COLUMNS,
    build_manifest,
    config_class,
    serialize_counterfactual_dataset,
    table_bytes,
    write_dataset_table,
)

pytestmark = pytest.mark.unit


def _weekdays(n: int = 4, seed: int = 0):
    return serialize_counterfactual_dataset(
        "natural_domains_arithmetic",
        n=n,
        seed=seed,
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
        target_variables=["result"],
    )


def test_row_vocabulary_is_the_documented_one():
    dataset = _weekdays()
    assert len(dataset.rows) == 4
    row = dataset.rows[0]
    # the columns documents reference (§2.2)
    for column in (
        "input",
        "counterfactual_inputs",
        "base_answer",
        "cf_answer",
        "label",
        "counterfactual_inputs_variables",
    ):
        assert column in row, column
    # answer forms from the task's own output_tokens declaration (§2.10)
    assert row["label_forms"] == [row["label"], row["label"].strip()]
    # every causal-model variable is a per-row column (§2.3), stringified
    assert row["entity"] in row["input"]
    assert row["number"] in row["input"]
    # ... and the counterfactual side lives in the per-role sibling
    (cf_variables,) = row["counterfactual_inputs_variables"]
    assert cf_variables["entity"] in row["counterfactual_inputs"][0]


def test_label_is_the_post_intervention_answer_not_the_counterfactuals_own():
    """The trap this column vocabulary exists to avoid: `label` is what the
    causal model outputs *after* the interchange, `cf_answer` is what the
    counterfactual prompt answers on its own. They coincide only when the
    intervention replaces every variable the answer depends on — as it does
    for a full `result` swap, which is why both are serialized and documented
    rather than aliased."""
    dataset = _weekdays()
    assert all(row["label"] == row["cf_answer"] for row in dataset.rows)

    # Intervening on `number` alone: the answer recomputes from the base
    # entity and the counterfactual number, so it is neither prompt's answer.
    partial = serialize_counterfactual_dataset(
        "natural_domains_arithmetic",
        n=8,
        seed=0,
        task_cfg=NaturalDomainConfig(domain_type="weekdays"),
        target_variables=["number"],
    )
    assert any(
        row["label"] not in (row["base_answer"], row["cf_answer"])
        for row in partial.rows
    )


def test_same_parameters_are_byte_identical():
    """What the content digest in a canonical form (§7) rests on."""
    assert table_bytes(_weekdays().rows) == table_bytes(_weekdays().rows)
    assert table_bytes(_weekdays(seed=1).rows) != table_bytes(_weekdays(seed=0).rows)


def test_written_table_resolves_through_the_seam(tmp_path):
    """The other half of the contract: what the builder writes is exactly what
    ``FileDatasets`` reads back, digest included."""
    dataset = _weekdays()
    out = tmp_path / "weekdays" / "train.json"
    digest = write_dataset_table(
        dataset.rows, out, manifest=build_manifest(dataset, task_cfg={"x": 1})
    )
    resolver = FileDatasets(root=tmp_path)
    assert resolver.digest("weekdays/train") == digest
    assert resolver.rows("weekdays/train") == json.loads(out.read_text())
    assert "label_forms" in resolver.columns("weekdays/train")

    manifest = json.loads((out.parent / "train.manifest.json").read_text())
    assert manifest["digest"] == digest
    assert manifest["task"] == "natural_domains_arithmetic"
    assert manifest["n"] == 4 and manifest["seed"] == 0
    assert manifest["target_variables"] == ["result"]
    assert manifest["task_cfg"] == {"x": 1}


def test_manifest_is_not_part_of_the_table(tmp_path):
    """Provenance must not change the content address: the sidecar is written
    beside the table, never into it."""
    dataset = _weekdays()
    plain = write_dataset_table(dataset.rows, tmp_path / "a" / "t.json")
    stamped = write_dataset_table(
        dataset.rows, tmp_path / "b" / "t.json", manifest=build_manifest(dataset)
    )
    assert plain == stamped


def test_missing_generator_names_the_alternatives():
    with pytest.raises(ValueError, match="generate_dataset"):
        serialize_counterfactual_dataset(
            "natural_domains_arithmetic",
            n=1,
            seed=0,
            task_cfg=NaturalDomainConfig(domain_type="weekdays"),
            target_variables=["result"],
            generator="generate_nonsense",
        )


def test_undeclared_answer_variable_refuses():
    with pytest.raises(ValueError, match="output_tokens"):
        serialize_counterfactual_dataset(
            "natural_domains_arithmetic",
            n=1,
            seed=0,
            task_cfg=NaturalDomainConfig(domain_type="weekdays"),
            target_variables=["result"],
            answer_variable="entity",
        )


def test_reserved_columns_cover_the_row_vocabulary():
    """A causal-model variable named like a row column would overwrite it, so
    the guard's list has to stay in step with what ``_row`` writes."""
    dataset = _weekdays()
    variables = set(dataset.rows[0]["counterfactual_inputs_variables"][0])
    assert not variables & RESERVED_COLUMNS
    assert {"input", "label", "cf_answer"} <= RESERVED_COLUMNS


def test_config_class_follows_the_task_convention():
    assert config_class("natural_domains_arithmetic") is NaturalDomainConfig
    assert config_class("MCQA") is None  # a singleton task takes no config


def test_a_generated_table_carries_no_model_dependency():
    """The seam's load-time property, asserted where it is created: a table is
    text and variable strings — no token ids, no tokenizer, no device."""
    dataset = _weekdays()
    for row in dataset.rows:
        for value in row.values():
            assert isinstance(value, (str, list)), value
