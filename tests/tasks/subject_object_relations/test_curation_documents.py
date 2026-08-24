"""The curation sweep, as a protocol document — the shape, not the run.

The per-relation base-accuracy table in this task's README was measured by a
producer the protocol refactor deleted (`data/curation_sweep.py`), while its
numbers stay load-bearing: they pick `config.py`'s default relation and the
relation a pinned tier would use. Recomputing it needs two things this seam
adds — task-generated tables, and a `match` that can grade a multi-token
object by its first token (the task's own `match_modes={"object": "prefix"}`,
spec §2.10).

This test asserts the campaign is *expressible and valid* end to end on CPU:
tables for several relations, one document sweeping `data.base.dataset` over
them, every column reference checked. Running it is a GPU campaign and belongs
with the coherent-model tier, not here — what would silently rot without a
test is the seam, and that is what this covers.
"""

from __future__ import annotations

import json

import pytest

from causalab.protocol.loader import check_data_columns, load
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks.serialize import (
    build_manifest,
    serialize_counterfactual_dataset,
    write_dataset_table,
)
from causalab.tasks.subject_object_relations.config import SubjectObjectRelationsConfig

pytestmark = pytest.mark.unit

#: A slice of the curation's own selections: the strongest "green" relation
#: (single-letter answers), a two-object bias relation, and a flagged relation
#: whose objects are multi-token — the case that needs first-token grading.
RELATIONS = ["word_first_letter", "name_gender", "country_capital_city"]

MODEL = "meta-llama/Llama-3.1-8B"


def _baseline_document(refs: list[str]) -> dict:
    """A no-intervention baseline: read the answer-position logits and score
    the declared answer forms, swept over one table per relation."""
    return {
        "version": "1",
        "description": "Per-relation base accuracy: the curation sweep as a document.",
        "model": {"key": MODEL, "revision": "main"},
        "causal_model": {"key": "subject_object_relations"},
        "data": {"base": {"dataset": {"sweep": refs}, "field": "input"}},
        "positions": {"answer_tok": {"index": -1}},
        "sites": {"lm_head": {"component": "lm_head"}},
        "reads": {
            "logits": {
                "site": "lm_head",
                "pos": "answer_tok",
                "model": "original",
                "input": "base",
            }
        },
        "metrics": {
            "accuracy": {
                "kind": "match",
                "of": "logits",
                "expected": "base_answer_forms",
                "mode": "first_token",
            }
        },
        "save": [
            {
                "value": "accuracy",
                "model": "original",
                "input": "base",
                "file_path": "accuracy.parquet",
            }
        ],
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[ResolutionEnv, list[str], dict]:
    """Tables for the sampled relations, in a scratch data root.

    Deliberately not committed: 35 relations × 64 rows is a build product, and
    the manifest beside each table is what makes it reproducible.
    """
    root = tmp_path_factory.mktemp("sor_data")
    refs, manifests = [], {}
    for relation in RELATIONS:
        dataset = serialize_counterfactual_dataset(
            "subject_object_relations",
            n=8,
            seed=0,
            task_cfg=SubjectObjectRelationsConfig(relation=relation),
        )
        ref = f"subject_object_relations/{relation}"
        digest = write_dataset_table(
            dataset.rows,
            root / f"{ref}.json",
            manifest=build_manifest(dataset, task_cfg={"relation": relation}),
        )
        refs.append(ref)
        manifests[relation] = json.loads(
            (
                root / "subject_object_relations" / f"{relation}.manifest.json"
            ).read_text()
        )
        assert manifests[relation]["digest"] == digest
    env = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    return env, refs, manifests


def test_the_swept_baseline_loads_validates_and_expands(built):
    env, refs, _ = built
    loaded = load(_baseline_document(refs), env)
    # one point per relation...
    assert len(loaded.expansion.points) == len(refs)
    # ...each stamping its own table's content digest (§2.2), so the points
    # are distinct provenance units rather than one document run three times
    stamped = {point["data"]["base"]["digest"] for point in loaded.canonical_points}
    assert len(stamped) == len(refs)
    assert len(set(loaded.point_digests)) == len(refs)


def test_every_column_reference_resolves(built):
    env, refs, _ = built
    refs_checked = check_data_columns(load(_baseline_document(refs), env), env)
    assert "base_answer_forms" in refs_checked  # the answer-form group column


def test_the_manifest_records_the_declared_prefix_mode(built):
    """Why the document needs ``first_token``: the task declares its answer
    match mode as ``prefix`` (multi-token objects like "Washington D.C."), and
    the builder records that next to the table so an author does not have to
    rediscover it."""
    _, _, manifests = built
    assert all(m["declared_match_mode"] == "prefix" for m in manifests.values())
    assert all(m["answer_variable"] == "object" for m in manifests.values())
