"""Tests for the manifold_bundle_ingest analysis (issue #265).

Covers the ``(bundle_dir, community_id) -> characterize input set`` producer:

- ``TestExemplarParsing`` (``unit``) — the robust exemplar parser, exercising
  the two failure modes earlier sessions hit by hand (multi-line spans and
  spans with embedded ``"`` quotes), plus marker/ellipsis stripping and the
  section boundary.
- ``TestBundleIngest`` (``unit``) — the end-to-end input-set build over a tiny
  synthetic fixture bundle (built on the fly, no committed fixtures): the Hydra
  ``main(cfg)`` entry, rotation round-trip through ``load_subspace``, the
  exemplar ``step1_dataset``, manifest field mapping, the ``hypotheses.jsonl``
  word-boundary scan, the pick fallback, and the mislabel warning. This class is
  the acceptance test for #265 ("fixture bundle + community id -> valid input
  set, no hand-editing"). The analysis is model-less and deterministic, so this
  regression lives in the ``unit`` tier, not ``golden`` (the coherent-model tier).
"""

from __future__ import annotations

import json
import logging
import pickle

import pytest
import torch
from omegaconf import OmegaConf

from causalab.analyses.characterize_subspace.loading import load_subspace
from causalab.analyses.manifold_bundle_ingest import main as mbi_main
from causalab.analyses.manifold_bundle_ingest.ingest import (
    build_characterize_inputs,
    extract_manifold_exemplars,
    find_community_record,
)


# A significance_description shaped like the real bundle's: a topology section,
# a 4x4 exemplar grid, and a trailing section. The grid deliberately includes
# the spans that broke naive `"(.*?)"` extraction — a multi-line span and a
# span with embedded quotes — plus a `<|begin_of_text|>` token, `>>>...<<<`
# activation markers, and `...` truncation.
SIGNIFICANCE = (
    "# Community [12212, 13649] — manifold significance\n"
    "\n"
    "## Shape (topology)\n"
    "a multi-loop product structure  ·  intrinsic dim ~3\n"
    "\n"
    "## Manifold exemplars (sampled along the geometry, grid layout)\n"
    '[grid_0_0] " the inclusion of animals is still scarce and>>> false<<<.\n'
    'It is worrying that life continues..."\n'
    '[grid_0_1] " it seems like "Launch a token and the market will come". To have>>> real<<< success and impact..."\n'
    '[grid_0_2] "<|begin_of_text|>I think this is>>> genuine<<< care and concern"\n'
    '[grid_0_3] " tell our own>>> true<<<st stories"\n'
    '[grid_1_0] " a>>> sure<<< way of doing this"\n'
    '[grid_1_1] " both of these people are>>> way<<< too professional"\n'
    '[grid_1_2] " if Amazon were a>>> real<<< jungle"\n'
    '[grid_1_3] " a>>> actual<<< saucy minx"\n'
    '[grid_2_0] " someone that>>> actually<<< knows their neighbors"\n'
    '[grid_2_1] " really couple of really>>> very<<< entirely"\n'
    '[grid_2_2] " to have>>> real<<< success"\n'
    '[grid_2_3] " the two games will be>>> locking<<< horns"\n'
    '[grid_3_0] " you could>>> truly<<< feel vibrations"\n'
    '[grid_3_1] " an unreliable push shaft"\n'
    '[grid_3_2] " the design is relatively>>> simple<<<"\n'
    '[grid_3_3] " one small step in building that>>> world<<<"\n'
    "\n"
    "## Axes\n"
    "[axis_0] this axis line must NOT be parsed as an exemplar\n"
)


def _record(
    *,
    comm: int,
    theme: str,
    n_dims: int,
    d_model: int,
    significance: str = SIGNIFICANCE,
    layer_index: int = 19,
) -> dict:
    """A minimal hypotheses.jsonl-style record (deterministic projection_matrix)."""
    rows = torch.arange(n_dims * d_model, dtype=torch.float32).reshape(n_dims, d_model)
    return {
        "subspace": {
            "projection_matrix": rows.tolist(),
            "n_dims": n_dims,
        },
        "config": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B",
            "layer_index": layer_index,
            "hook_site": "residual",
        },
        "topology_characterization": {
            "shape": "product",
            "description": "a multi-loop product structure (e.g. a torus)",
            "intrinsic_dim": 3,
            "n_components": 1,
            "n_loops": 2,
            "n_voids": 0,
        },
        "significance_description": significance,
        "hypothesis": {"theme": theme, "manifold_guess": theme},
        "comm": comm,
    }


@pytest.fixture()
def bundle_dir(tmp_path):
    """A tiny synthetic manifold-SAE bundle on disk.

    - ``hypotheses.jsonl`` holds comm 12 (the rich record) and comm 1 (used to
      check the word-boundary scan: ``comm 1`` must not match ``comm 12``).
    - ``picks/comm12_finance.json`` is *mislabelled* (the record is about
      authenticity, not finance) and carries ``member_features`` (the source of
      member-feature ids for provenance).
    - ``picks/comm1_numbers.json`` is consistent with its record.
    - ``picks/comm7_security.json`` has no ``hypotheses.jsonl`` entry, exercising
      the pick fallback source.
    - ``communities.pkl`` holds *decoy* ids and must be ignored — ingestion does
      not unpickle it; member ids come from the pick's ``member_features``.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    picks = bundle / "picks"
    picks.mkdir()

    rec12 = _record(
        comm=12,
        theme="lexical emphasis on authenticity/truth: true, real, genuine, actual",
        n_dims=4,
        d_model=8,
    )
    rec1 = _record(
        comm=1,
        theme="numbers and numeric magnitude words",
        n_dims=2,
        d_model=8,
    )
    with open(bundle / "hypotheses.jsonl", "w", encoding="utf-8") as fh:
        # comm 12 first on purpose: scanning for comm 1 must skip this line.
        fh.write(json.dumps(rec12) + "\n")
        fh.write(json.dumps(rec1) + "\n")

    pick12 = dict(rec12)
    pick12["member_features"] = [
        {"feature": 101, "label": "words about truth and authenticity"},
        {"feature": 102, "label": "the word genuine"},
        {"feature": 103, "label": "really / actually intensifiers"},
        {"feature": 104, "label": "the word real"},
    ]
    (picks / "comm12_finance.json").write_text(json.dumps(pick12), encoding="utf-8")
    (picks / "comm1_numbers.json").write_text(json.dumps(rec1), encoding="utf-8")

    rec7 = _record(comm=7, theme="security and access control", n_dims=3, d_model=8)
    (picks / "comm7_security.json").write_text(json.dumps(rec7), encoding="utf-8")

    # Decoy: ingestion must NOT unpickle this. If it ever does, the member-id
    # assertion below (which expects the pick's ids) will fail.
    communities = [[] for _ in range(13)]
    communities[12] = [999, 998]
    communities[1] = [201, 202]
    with open(bundle / "communities.pkl", "wb") as fh:
        pickle.dump(communities, fh)

    return bundle


class TestExemplarParsing:
    """Robust extraction of the per-manifold exemplar spans."""

    pytestmark = pytest.mark.unit

    def test_recovers_every_grid_entry(self):
        spans = extract_manifold_exemplars(SIGNIFICANCE)
        # All 16 grid cells survive — including the multi-line and embedded-quote
        # spans that a naive `"(.*?)"` regex dropped (issue #265).
        assert len(spans) == 16

    def test_multiline_span_is_one_entry(self):
        spans = extract_manifold_exemplars(SIGNIFICANCE)
        first = spans[0]
        assert "scarce and false." in first
        assert "It is worrying that life continues" in first  # the second line
        assert not first.endswith("...")

    def test_embedded_quotes_preserved(self):
        spans = extract_manifold_exemplars(SIGNIFICANCE)
        embedded = next(s for s in spans if "Launch a token" in s)
        assert '"Launch a token and the market will come"' in embedded
        assert "real success" in embedded

    def test_markers_and_special_tokens_stripped(self):
        spans = extract_manifold_exemplars(SIGNIFICANCE)
        joined = "\n".join(spans)
        assert ">>>" not in joined
        assert "<<<" not in joined
        assert "<|begin_of_text|>" not in joined
        # the peak tokens themselves are kept
        assert any("genuine care and concern" in s for s in spans)

    def test_section_boundary_excludes_other_sections(self):
        spans = extract_manifold_exemplars(SIGNIFICANCE)
        assert all("must NOT be parsed" not in s for s in spans)

    def test_missing_section_returns_empty(self):
        assert extract_manifold_exemplars("# heading\n\nno exemplars here") == []
        assert extract_manifold_exemplars("") == []
        assert extract_manifold_exemplars(None) == []


class TestBundleIngest:
    """End-to-end build of a characterize input set from (bundle, comm)."""

    pytestmark = pytest.mark.unit

    def test_main_produces_valid_input_set(self, bundle_dir, tmp_path):
        # Drive the Hydra entry point exactly as the runner would.
        out = tmp_path / "out12"
        cfg = OmegaConf.create(
            {
                "experiment_root": str(tmp_path / "art"),
                "manifold_bundle_ingest": {
                    "_name_": "manifold_bundle_ingest",
                    "_output_dir": str(out),
                    "bundle": str(bundle_dir),
                    "community_id": 12,
                },
            }
        )
        result = mbi_main.main(cfg)
        manifest = result["manifest"]
        assert result["output_dir"] == str(out)

        # Manifest fields mapped from the record's config + hypothesis + topology.
        assert manifest["model"] == "meta-llama/Meta-Llama-3.1-8B"
        assert manifest["layer"] == 19
        assert manifest["site"] == "residual"
        assert manifest["k_features_hint"] == 4
        assert manifest["significance"]["hypothesis_text"].startswith(
            "lexical emphasis"
        )
        assert manifest["significance"]["topology_description"].startswith("product")
        assert manifest["significance"]["figure_path"] is None

        # Rotation safetensors round-trips through the analysis loader at the
        # right orientation (d_model, k) and matches the source projection_matrix.
        projector = load_subspace(
            manifest["subspace_artifact"], k_features_hint=manifest["k_features_hint"]
        )
        assert projector.d_model == 8
        assert projector.k == 4
        expected = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).t()
        assert torch.allclose(projector.rotation, expected)

        # step1_dataset is a list[str] of the exemplar spans, clearing the gate's
        # >=8 floor.
        spans = json.loads((out / "step1_dataset.json").read_text(encoding="utf-8"))
        assert isinstance(spans, list)
        assert all(isinstance(s, str) for s in spans)
        assert len(spans) == 16

        # The manifest is also persisted for provenance, with no hand-editing.
        assert (out / "subspace_manifest.json").is_file()
        assert manifest["provenance"]["source"] == "hypotheses.jsonl"
        assert manifest["provenance"]["n_exemplars"] == 16
        # Member ids come from the pick's member_features (JSON), not from the
        # communities.pkl decoy ([999, 998]) — ingestion never unpickles it.
        assert manifest["provenance"]["member_feature_ids"] == [101, 102, 103, 104]

    def test_main_requires_bundle_and_community(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "manifold_bundle_ingest": {
                    "_output_dir": str(tmp_path),
                    "community_id": 1,
                }
            }
        )
        with pytest.raises(ValueError, match="bundle is required"):
            mbi_main.main(cfg)

    def test_mislabelled_pick_warns(self, bundle_dir, caplog):
        with caplog.at_level(logging.WARNING):
            _, provenance = find_community_record(str(bundle_dir), 12)
        assert provenance["pick_label"] == "finance"
        assert provenance["warnings"], "expected a mislabel warning for comm12_finance"
        assert any("finance" in w for w in provenance["warnings"])
        assert any("finance" in r.message for r in caplog.records)

    def test_consistent_pick_does_not_warn(self, bundle_dir):
        _, provenance = find_community_record(str(bundle_dir), 1)
        assert provenance["pick_label"] == "numbers"
        assert provenance["warnings"] == []

    def test_jsonl_scan_respects_word_boundary(self, bundle_dir):
        # Requesting comm 1 must return the comm-1 record, not comm 12, even
        # though "comm": 12 contains the digit 1.
        record, provenance = find_community_record(str(bundle_dir), 1)
        assert record["comm"] == 1
        assert record["subspace"]["n_dims"] == 2
        assert provenance["source"] == "hypotheses.jsonl"

    def test_pick_fallback_when_absent_from_jsonl(self, bundle_dir, tmp_path):
        out = tmp_path / "out7"
        manifest = build_characterize_inputs(
            bundle_dir=str(bundle_dir), community_id=7, out_dir=str(out)
        )
        assert manifest["provenance"]["source"] == "pick"
        assert manifest["k_features_hint"] == 3

    def test_unknown_community_raises(self, bundle_dir, tmp_path):
        with pytest.raises(KeyError):
            build_characterize_inputs(
                bundle_dir=str(bundle_dir),
                community_id=999,
                out_dir=str(tmp_path / "x"),
            )
