"""Canonicalization: the canonical-stamp principle (spec §7)."""

from __future__ import annotations

import copy

import pytest

from causalab.protocol.canonical import canonical_bytes, canonicalize, digest
from causalab.protocol.loader import load

from tests.protocol._env import CORPUS_DIR
from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def test_pos_sugar_and_alias_canonicalize(env):
    raw = base_doc()
    canonical = canonicalize(raw, env)
    assert canonical["reads"]["v_cf"]["pos"] == {"index": -1}
    # dtype materializes like every other default: an authored document may
    # be silent about precision, a canonical one never is (§2.1)
    assert canonical["model"] == {"key": "gpt2", "revision": "main", "dtype": "fp32"}


def test_all_pos_sugar_canonicalizes(env):
    """Both spellings land on one canonical form, so a document authored
    either way digests identically."""
    sugar, explicit = base_doc(), base_doc()
    sugar["reads"]["v_cf"]["pos"] = "all"
    explicit["reads"]["v_cf"]["pos"] = {"all": True}
    assert canonicalize(sugar, env)["reads"]["v_cf"]["pos"] == {"all": True}
    assert digest(canonicalize(sugar, env)) == digest(canonicalize(explicit, env))


def test_all_positions_changes_the_digest(env):
    """The position is part of the address, so it is part of the record."""
    raw = base_doc()
    raw["reads"]["v_cf"]["pos"] = "all"
    assert digest(canonicalize(raw, env)) != digest(canonicalize(base_doc(), env))


def test_dataset_digest_stamped(env):
    canonical = canonicalize(base_doc(), env)
    stamped = canonical["data"]["base"]["digest"]
    assert stamped == env.datasets.digest("weekdays/train")
    assert len(stamped) == 64


def test_im_write_lists_sorted(env):
    raw = base_doc()
    raw["writes"]["another"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "v_cf", "alpha": 1.0}},
    }
    raw["intervened_models"]["patched"]["writes"] = ["patch", "another"]
    canonical = canonicalize(raw, env)
    assert canonical["intervened_models"]["patched"]["writes"] == ["another", "patch"]


def test_train_defaults_materialized(env):
    loaded = load(CORPUS_DIR / "04_das_im.json", env)
    train = loaded.canonical_document["train"]
    assert train["optimizer"]["betas"] == [0.9, 0.999]
    assert train["optimizer"]["eps"] == 1e-8
    assert train["optimizer"]["schedule"] == "constant"
    assert train["precision"] == {"feature": "fp32", "loss": "fp32"}
    # the model's own precision has one home, and it is the model section
    assert loaded.canonical_document["model"]["dtype"] == "bf16"
    assert "digest" in train["eval"]  # eval.split is a dataset ref too


def test_featurizer_widths_derived(env):
    loaded = load(CORPUS_DIR / "04_das_im.json", env)
    rot = loaded.canonical_document["featurizers"]["rot"]
    assert rot["width"] == 4096
    assert rot["params"] == {"weight": [4096, 8]}
    assert rot["dtype"] == "fp32"


def test_gate_width_derived(env):
    loaded = load(CORPUS_DIR / "05_dbm_im.json", env)
    gate = loaded.canonical_document["featurizers"]["gate"]
    assert gate["params"] == {"theta": [4096]}


def test_loaded_featurizer_hashed_not_shaped(env):
    loaded = load(CORPUS_DIR / "09_das_apply_im.json", env)
    rot = loaded.canonical_document["featurizers"]["rot"]
    assert "content_digest" in rot and len(rot["content_digest"]) == 64
    assert "params" not in rot  # loaded bundles are identified by their bytes


def test_swept_document_keeps_wrappers(env):
    loaded = load(CORPUS_DIR / "08_weekdays_das_sweep_im.json", env)
    doc_form = loaded.canonical_document
    assert doc_form["featurizers"]["rot"]["k"] == {"sweep": [8, 16, 32]}
    point_form = loaded.canonical_points[0]
    assert point_form["featurizers"]["rot"]["k"] == 8
    assert point_form["featurizers"]["rot"]["params"] == {"weight": [4096, 8]}


def test_description_is_part_of_the_digest(env):
    raw = base_doc()
    first = digest(canonicalize(raw, env))
    raw["description"] = "same experiment, different words"
    ordered = {"version": raw.pop("version"), "description": raw.pop("description")}
    ordered.update(raw)
    assert digest(canonicalize(ordered, env)) != first


def test_canonical_bytes_are_sorted_and_minimal(env):
    canonical = canonicalize(base_doc(), env)
    blob = canonical_bytes(canonical).decode()
    assert ": " not in blob and ", " not in blob
    assert blob.index('"data"') < blob.index('"model"')  # sorted keys


def test_out_of_range_layer_refused(env):
    raw = base_doc()
    raw["sites"]["tgt"]["layer"] = 40  # gpt2 has 12 layers
    with pytest.raises(Exception) as err:
        canonicalize(raw, env)
    assert "[V4]" in str(err.value)


def test_match_mode_default_materialized(env):
    """Optional metric fields are materialized like ``train.optimizer``
    defaults (§2.10): the two spellings of "exact" are one canonical form, so
    adding the field cannot split the digest of documents that omit it."""
    omitted = base_doc()
    omitted["metrics"]["m"] = {"kind": "match", "of": "logits", "expected": "label"}
    omitted["save"].append(
        {"value": "m", "model": "patched", "input": "base", "file_path": "m.json"}
    )
    spelled = {
        **omitted,
        "metrics": {
            **omitted["metrics"],
            "m": {**omitted["metrics"]["m"], "mode": "exact"},
        },
    }
    assert canonicalize(omitted, env)["metrics"]["m"]["mode"] == "exact"
    assert digest(canonicalize(omitted, env)) == digest(canonicalize(spelled, env))


def test_first_token_mode_is_a_different_document(env):
    """...and a real semantic choice still moves the digest."""
    exact = base_doc()
    exact["metrics"]["m"] = {"kind": "match", "of": "logits", "expected": "label"}
    exact["save"].append(
        {"value": "m", "model": "patched", "input": "base", "file_path": "m.json"}
    )
    first = {
        **exact,
        "metrics": {
            **exact["metrics"],
            "m": {**exact["metrics"]["m"], "mode": "first_token"},
        },
    }
    assert digest(canonicalize(exact, env)) != digest(canonicalize(first, env))


def test_column_position_canonicalizes_verbatim(env):
    """A column position is data the canonical form carries as authored — no
    derivation, so the digest names the column the document reads."""
    raw = base_doc()
    raw["positions"] = {"subj": {"column": "entity"}}
    raw["reads"]["v_cf"]["pos"] = "subj"
    canonical = canonicalize(in_order(raw), env)
    assert canonical["positions"]["subj"] == {"column": "entity"}


def test_generated_frame_canonicalizes_verbatim(env):
    """The frame selector is authored data with no defaults to materialize,
    so it carries through untouched — and a different budget is a different
    document, because the position enters the read's closure."""
    raw = base_doc()
    raw["positions"] = {"tail": {"generated": {"max_new_tokens": 8}, "index": -1}}
    raw["reads"]["v_cf"]["pos"] = "tail"
    canonical = canonicalize(in_order(raw), env)
    assert canonical["positions"]["tail"] == {
        "generated": {"max_new_tokens": 8},
        "index": -1,
    }
    # deep-copied: canonicalize passes this section through by reference, so
    # mutating a shared nested dict would rewrite the form just measured
    longer = copy.deepcopy(in_order(raw))
    longer["positions"]["tail"]["generated"]["max_new_tokens"] = 16
    assert digest(canonical) != digest(canonicalize(longer, env))


def test_prompt_frame_documents_digest_unchanged(env):
    """The new field is absent, not defaulted, on every position that does
    not ask for a continuation — the reason no existing digest moves."""
    canonical = canonicalize(base_doc(), env)
    assert "generated" not in canonical["reads"]["v_cf"]["pos"]
