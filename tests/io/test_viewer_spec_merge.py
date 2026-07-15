"""Unit tests for the idempotent viewer_spec section-merge helper.

Pure in-memory + filesystem work — no GPU, no model load. Mirrors the
experiment pipeline flow: a plan-written base spec (a few §D nodes) is
merged with the pipeline's fragment (a leading "Characterization evidence" section +
the fixed-subspace IIA geometry) and must keep the base selection intact, lead
with the fragment, dedup on re-run, and still render.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from causalab.io.artifact_viewer import build_viewer, load_spec
from causalab.io.viewer_spec_merge import (
    merge_sections,
    merge_spec_file,
)

pytestmark = pytest.mark.unit


def _plan_base_spec() -> dict:
    """A base spec shaped like what the plan stage writes (§D default selection)."""
    return {
        "title": "Artifact Viewer",
        "sections": [
            {
                "heading": "Baseline",
                "items": [
                    {
                        "caption": "accuracy",
                        "candidates": ["**/baseline/**/accuracy.png"],
                    }
                ],
            },
            {
                "heading": "Locate",
                "items": [
                    {
                        "caption": "layer scan",
                        "candidates": ["**/locate/**/heatmap.png"],
                    }
                ],
            },
            {
                "heading": "Subspace",
                "items": [
                    {
                        "caption": "geometry",
                        "candidates": ["**/subspace/**/features_3d.html"],
                    }
                ],
            },
        ],
    }


def _pipeline_fragment() -> list[dict]:
    """The fragment the pipeline merges in (characterization + fixed-IIA)."""
    return [
        {
            "heading": "Characterization evidence",
            "items": [
                {
                    "caption": "Subspace projection explorer",
                    "candidates": [
                        "**/characterize_subspace/**/projection_explorer.html",
                        "**/plan/figures/projection_explorer.html",
                    ],
                },
                {
                    "caption": "Webtext peak-token distribution",
                    "candidates": [
                        "**/characterize_subspace/**/projection_distribution.html",
                        "**/plan/figures/projection_distribution.html",
                    ],
                },
            ],
        },
        {
            "heading": "Fixed-subspace interchange-IIA",
            "items": [
                {
                    "caption": "Threaded-subspace geometry",
                    "candidates": ["**/subspace/fixed_k*/**/features_3d.html"],
                }
            ],
        },
    ]


def _headings(spec: dict) -> list[str]:
    return [s.get("heading") for s in spec["sections"]]


def test_prepend_leads_and_preserves_base() -> None:
    base = _plan_base_spec()
    merged = merge_sections(base, _pipeline_fragment(), position="prepend")

    # Fragment leads; every base section survives (no clobber); nothing dropped.
    assert _headings(merged) == [
        "Characterization evidence",
        "Fixed-subspace interchange-IIA",
        "Baseline",
        "Locate",
        "Subspace",
    ]
    # The base spec object is not mutated.
    assert _headings(base) == ["Baseline", "Locate", "Subspace"]
    # Non-sections top-level keys carry through.
    assert merged["title"] == "Artifact Viewer"


def test_append_position() -> None:
    merged = merge_sections(_plan_base_spec(), _pipeline_fragment(), position="append")
    assert _headings(merged) == [
        "Baseline",
        "Locate",
        "Subspace",
        "Characterization evidence",
        "Fixed-subspace interchange-IIA",
    ]


def test_idempotent_on_rerun() -> None:
    base = _plan_base_spec()
    once = merge_sections(base, _pipeline_fragment(), position="prepend")
    twice = merge_sections(once, _pipeline_fragment(), position="prepend")
    # Re-running the merge is a no-op (dedup by heading) — no duplicated sections.
    assert twice == once
    assert len(_headings(twice)) == len(set(_headings(twice)))


def test_dedup_replaces_existing_heading_in_place() -> None:
    # A base that already contains a fragment heading (e.g. a prior partial merge):
    # the stale copy is removed and the fragment version re-leads, not duplicated.
    base = _plan_base_spec()
    base["sections"].append(
        {
            "heading": "Fixed-subspace interchange-IIA",
            "items": [{"caption": "stale", "candidates": ["**/old.png"]}],
        }
    )
    merged = merge_sections(base, _pipeline_fragment(), position="prepend")
    headings = _headings(merged)
    assert headings.count("Fixed-subspace interchange-IIA") == 1
    assert headings[1] == "Fixed-subspace interchange-IIA"
    # The surviving section is the fragment's, not the stale base one.
    fixed = next(
        s
        for s in merged["sections"]
        if s["heading"] == "Fixed-subspace interchange-IIA"
    )
    assert fixed["items"][0]["caption"] == "Threaded-subspace geometry"


def test_sections_without_heading_are_preserved() -> None:
    # A base repeat block is kept when the fragment carries no matching repeat.
    base = {
        "title": "t",
        "sections": [
            {
                "repeat": {
                    "over": "*/",
                    "label_from": {"kind": "dirname"},
                    "heading": "{label}",
                    "sections": [],
                }
            },
            {
                "heading": "Baseline",
                "items": [{"caption": "c", "candidates": ["**/a.png"]}],
            },
        ],
    }
    merged = merge_sections(base, _pipeline_fragment(), position="prepend")
    assert any("repeat" in s for s in merged["sections"])
    assert _headings(merged)[:2] == [
        "Characterization evidence",
        "Fixed-subspace interchange-IIA",
    ]


def _repeat_fragment() -> list[dict]:
    """A fragment shaped like the CEP canonical one: a heading + a per-task repeat."""
    return [
        {
            "heading": "Task setup",
            "items": [
                {"caption": "tasks", "candidates": ["plan/figures/task_setup.html"]}
            ],
        },
        {
            "repeat": {
                "over": "artifacts/*/",
                "label_from": {"kind": "dirname"},
                "heading": "Task: {label}",
                "sections": [
                    {
                        "heading": "Baseline",
                        "items": [
                            {
                                "caption": "confusion",
                                "candidates": ["**/baseline/**/confusion_heatmap.*"],
                            }
                        ],
                    }
                ],
            }
        },
    ]


def test_idempotent_with_repeat_block_in_fragment() -> None:
    # The canonical CEP fragment carries a per-task `repeat` block. Re-merging it
    # must NOT duplicate the repeat (repeat blocks dedup by their `over` glob).
    base = _plan_base_spec()
    once = merge_sections(base, _repeat_fragment(), position="prepend")
    twice = merge_sections(once, _repeat_fragment(), position="prepend")
    assert twice == once
    repeats = [s for s in twice["sections"] if "repeat" in s]
    assert len(repeats) == 1
    assert repeats[0]["repeat"]["over"] == "artifacts/*/"


def test_repeat_with_different_over_is_preserved() -> None:
    # A base repeat over a *different* glob is not matched by the fragment's
    # repeat (keyed on `over`), so it survives alongside the fragment's.
    base = _plan_base_spec()
    base["sections"].insert(
        0,
        {
            "repeat": {
                "over": "models/*/",
                "label_from": {"kind": "dirname"},
                "heading": "{label}",
                "sections": [
                    {
                        "heading": "X",
                        "items": [{"caption": "c", "candidates": ["**/x.png"]}],
                    }
                ],
            }
        },
    )
    merged = merge_sections(base, _repeat_fragment(), position="prepend")
    overs = sorted(s["repeat"]["over"] for s in merged["sections"] if "repeat" in s)
    assert overs == ["artifacts/*/", "models/*/"]


def test_fragment_as_mapping_with_sections_key() -> None:
    # The helper accepts either a bare list or a {sections: [...]} mapping.
    frag_mapping = {"sections": _pipeline_fragment()}
    merged = merge_sections(_plan_base_spec(), frag_mapping, position="prepend")
    assert _headings(merged)[0] == "Characterization evidence"


def test_invalid_fragment_raises() -> None:
    with pytest.raises(ValueError):
        merge_sections(_plan_base_spec(), "not-a-list-or-sections-mapping")


def test_invalid_position_raises() -> None:
    with pytest.raises(ValueError):
        merge_sections(_plan_base_spec(), _pipeline_fragment(), position="sideways")


def test_merge_result_revalidated_against_schema() -> None:
    # A structurally broken fragment is caught at merge time, not deferred to render.
    bad_fragment = [{"heading": "X", "items": [{"caption": "no candidates"}]}]
    with pytest.raises(ValueError):
        merge_sections(_plan_base_spec(), bad_fragment)


def test_file_roundtrip_writes_valid_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "viewer_spec.yaml"
    frag_path = tmp_path / "fragment.yaml"
    spec_path.write_text(yaml.safe_dump(_plan_base_spec()), encoding="utf-8")
    frag_path.write_text(yaml.safe_dump(_pipeline_fragment()), encoding="utf-8")

    merge_spec_file(spec_path, frag_path, position="prepend")

    # The rewritten file reloads + revalidates through the viewer's own loader.
    reloaded = load_spec(spec_path)
    assert _headings(reloaded)[0] == "Characterization evidence"
    # And re-merging the written file is still idempotent.
    merge_spec_file(spec_path, frag_path, position="prepend")
    assert load_spec(spec_path) == reloaded


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_merged_spec_renders_with_characterization_first(tmp_path: Path) -> None:
    # End-to-end: build a fake session tree, merge, render, and assert the
    # characterization section leads and its figure resolves.
    root = tmp_path / "session"
    _touch(root / "plan" / "figures" / "projection_explorer.html")
    _touch(root / "artifacts" / "wd" / "m" / "baseline" / "accuracy.png")
    _touch(
        root
        / "artifacts"
        / "wd"
        / "m"
        / "subspace"
        / "fixed_k8"
        / "res"
        / "features_3d.html"
    )

    merged = merge_sections(_plan_base_spec(), _pipeline_fragment(), position="prepend")
    out = root / "result" / "artifact_viewer"
    html = build_viewer(merged, root, out, copy_assets=True).read_text(encoding="utf-8")

    # Characterization heading is rendered before the Baseline heading.
    assert "Characterization evidence" in html and "Baseline" in html
    assert html.index("Characterization evidence") < html.index("Baseline")
    # The characterization figure (from plan/figures/) resolved -> its caption is
    # rendered; a figure whose candidates don't resolve is dropped caption-and-all.
    assert "Subspace projection explorer" in html  # resolved (plan/figures/ hit)
    assert "Webtext peak-token distribution" not in html  # dropped (no file touched)
    # An asset was copied under figures/ for the resolved iframe.
    assert re.search(r'(?:src|href)="figures/[^"]+\.html"', html)
