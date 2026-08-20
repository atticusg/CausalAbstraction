"""Canonicalization: the canonical-stamp principle (spec §7)."""

from __future__ import annotations

import pytest

from causalab.protocol.canonical import canonical_bytes, canonicalize, digest
from causalab.protocol.loader import load

from tests.protocol._env import CORPUS_DIR
from tests.protocol._docs import base_doc

pytestmark = pytest.mark.unit


def test_pos_sugar_and_alias_canonicalize(env):
    raw = base_doc()
    canonical = canonicalize(raw, env)
    assert canonical["reads"]["v_cf"]["pos"] == {"index": -1}
    assert canonical["model"] == {"key": "gpt2", "revision": "main"}


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
    assert train["precision"] == {"feature": "fp32", "loss": "fp32", "model": "bf16"}
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
