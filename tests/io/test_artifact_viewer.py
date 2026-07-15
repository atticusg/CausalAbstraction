"""Unit tests for the generic spec-driven HTML artifact viewer.

Pure filesystem + string work — no GPU, no model load. Fake artifact trees are
built with tiny byte payloads under ``tmp_path``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from causalab.io.artifact_viewer import build_viewer, load_spec

pytestmark = pytest.mark.unit


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build(spec: dict, root: Path, tmp_path: Path, *, copy_assets: bool = True) -> str:
    out = tmp_path / "viewer"
    index = build_viewer(spec, root, out, copy_assets=copy_assets)
    return index.read_text(encoding="utf-8")


def _src_refs(html: str) -> list[str]:
    """All src=/href= URLs in the rendered page."""
    return re.findall(r'(?:src|href)="([^"]+)"', html)


def _fig_refs(html: str) -> list[str]:
    """Only the copied-asset refs (under figures/)."""
    return [r for r in _src_refs(html) if r.startswith("figures/")]


class TestFiletypePreference:
    def test_html_over_png_over_pdf(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "a" / "fig.pdf")
        _touch(root / "a" / "fig.png")
        html_path = root / "a" / "fig.html"
        _touch(html_path)
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "c", "candidates": ["**/fig.*"]}],
                }
            ],
        }

        # The dest asset preserves the *source extension* (stem becomes the
        # caption slug), so assert on the chosen extension, not the source name.

        # all three present -> html wins (iframe)
        html = _build(spec, root, tmp_path)
        assert _fig_refs(html)[0].endswith(".html")
        assert "<iframe" in html

        # drop html -> png wins (img)
        html_path.unlink()
        html = _build(spec, root, tmp_path / "b")
        assert _fig_refs(html)[0].endswith(".png")
        assert "<img" in html

        # drop png -> pdf wins (iframe, last resort)
        (root / "a" / "fig.png").unlink()
        html = _build(spec, root, tmp_path / "c")
        assert _fig_refs(html)[0].endswith(".pdf")
        assert "<iframe" in html

    def test_candidate_order_beats_filetype(self, tmp_path: Path) -> None:
        # First candidate matches only a pdf; second matches an html.
        # Candidate order is authoritative -> pdf wins despite html being "better".
        root = tmp_path / "art"
        _touch(root / "first" / "only.pdf")
        _touch(root / "second" / "better.html")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {
                            "caption": "c",
                            "candidates": ["**/first/*.*", "**/second/*.*"],
                        }
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        # pdf from the first candidate wins over the html from the second
        assert _fig_refs(html)[0].endswith(".pdf")

    def test_unrenderable_only_falls_through(self, tmp_path: Path) -> None:
        # A candidate matching only a .json (unrenderable) is skipped for the next.
        root = tmp_path / "art"
        _touch(root / "x" / "data.json")
        _touch(root / "y" / "plot.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {"caption": "c", "candidates": ["**/*.json", "**/*.png"]}
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        assert _fig_refs(html)[0].endswith(".png")
        assert "<img" in html


class TestDropUnresolved:
    def test_unresolved_figure_dropped_no_placeholder(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "present.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "Has one",
                    "items": [
                        {"caption": "present fig", "candidates": ["**/present.png"]},
                        {"caption": "absent fig", "candidates": ["**/nope.html"]},
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        # the missing slot leaves no placeholder behind ...
        assert "has not been created" not in html
        assert "absent fig" not in html
        # ... but the resolvable sibling still renders
        assert "present fig" in html

    def test_section_with_no_resolvable_items_dropped(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "All missing",
                    "items": [{"caption": "absent", "candidates": ["**/nope.html"]}],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        # an empty section is dropped entirely: no heading, no TOC, no asset dir
        assert "All missing" not in html
        assert "<nav>" not in html
        assert not (tmp_path / "viewer" / "figures").exists()

    def test_partial_row_collapses_to_single_figure(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "left.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {
                            "row": [
                                {"caption": "left", "candidates": ["left.png"]},
                                {"caption": "right", "candidates": ["missing.png"]},
                            ]
                        }
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        # one survivor -> standalone figure, not a half-empty 2-col row
        assert "left" in html
        assert "right" not in html
        assert 'class="row"' not in html

    def test_drop_is_warned_on_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "art"
        _touch(root / "present.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {"caption": "present", "candidates": ["present.png"]},
                        {"caption": "ghost", "candidates": ["ghost.png"]},
                    ],
                }
            ],
        }
        _build(spec, root, tmp_path)
        err = capsys.readouterr().err
        assert "ghost" in err and "dropped" in err


class TestLayout:
    def test_row_side_by_side(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "left.png")
        _touch(root / "right.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {
                            "row": [
                                {"caption": "left", "candidates": ["left.png"]},
                                {"caption": "right", "candidates": ["right.png"]},
                            ]
                        }
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        row = re.search(r'<div class="row">(.*?)</div>', html, re.DOTALL)
        assert row is not None
        assert row.group(1).count("<figure>") == 2

    def test_toc_anchors_match_sections(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "f.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "First section",
                    "items": [{"caption": "c", "candidates": ["f.png"]}],
                },
                {
                    "heading": "Second section",
                    "items": [{"caption": "c", "candidates": ["f.png"]}],
                },
            ],
        }
        html = _build(spec, root, tmp_path)
        assert "<nav>" in html
        toc_targets = re.findall(r"<nav>.*?</nav>", html, re.DOTALL)[0]
        hrefs = re.findall(r'href="#([^"]+)"', toc_targets)
        assert len(hrefs) == 2
        for anchor in hrefs:
            assert f'id="{anchor}"' in html


class TestRepeat:
    def test_repeat_per_subdir(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "alpha" / "deep" / "plot.png")
        _touch(root / "beta" / "deep" / "plot.png")
        spec = {
            "title": "t",
            "sections": [
                {
                    "repeat": {
                        "over": "*/",
                        "label_from": {"kind": "dirname"},
                        "heading": "{label}",
                        "sections": [
                            {
                                "heading": "Plot",
                                "items": [
                                    {"caption": "p", "candidates": ["**/plot.png"]}
                                ],
                            }
                        ],
                    }
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        # two instances, each labelled by its dirname, each in the TOC
        assert ">alpha</h2>" in html
        assert ">beta</h2>" in html
        toc = re.findall(r"<nav>.*?</nav>", html, re.DOTALL)[0]
        assert toc.count("<li>") == 2
        # per-instance globs resolve within their own subtree (distinct copied assets)
        refs = [r for r in _src_refs(html) if r.startswith("figures/")]
        assert len(refs) == 2
        assert len(set(refs)) == 2

    def test_repeat_no_matches_notes_it(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        spec = {
            "title": "t",
            "sections": [
                {
                    "repeat": {
                        "over": "*/",
                        "sections": [
                            {
                                "heading": "P",
                                "items": [{"caption": "p", "candidates": ["**/x.png"]}],
                            }
                        ],
                    }
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        assert "No instances matched" in html


class TestAssetsRelative:
    def test_copied_and_relative(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "deep" / "nested" / "plot.png", b"imagedata")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "the plot", "candidates": ["**/plot.png"]}],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        fig_dir = tmp_path / "viewer" / "figures"
        copied = [p for p in fig_dir.rglob("*") if p.is_file()]
        assert len(copied) == 1
        for ref in _src_refs(html):
            if ref.startswith("figures/"):
                assert ".." not in ref
                assert not ref.startswith("/")
                assert not ref.startswith("data:")
                assert (tmp_path / "viewer" / ref).exists()


class TestEscaping:
    def test_caption_and_intro_escaped(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "f.png")
        spec = {
            "title": "Title & <co>",
            "intro": {"Key<>": 'val & "x"'},
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {
                            "caption": '<script>alert(1)</script> & "q"',
                            "candidates": ["f.png"],
                        }
                    ],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "Key&lt;&gt;" in html


class TestIntro:
    def test_intro_keys_rendered(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        spec = {
            "title": "t",
            "intro": {"Featurizer": "SAE", "Target LM": "Llama"},
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "c", "candidates": ["**/none.png"]}],
                }
            ],
        }
        html = _build(spec, root, tmp_path)
        assert 'id="overview"' in html
        assert "Featurizer" in html and "SAE" in html
        assert "Target LM" in html and "Llama" in html


class TestLoadSpec:
    def test_missing_sections_raises(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("title: t\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sections"):
            load_spec(spec_path)

    def test_bad_item_raises(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(
            "title: t\nsections:\n  - heading: S\n    items:\n      - not_a_figure: true\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="caption"):
            load_spec(spec_path)

    def test_valid_roundtrips(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(
            "title: t\n"
            "sections:\n"
            "  - heading: S\n"
            "    items:\n"
            "      - caption: c\n"
            "        candidates: ['**/x.png']\n",
            encoding="utf-8",
        )
        spec = load_spec(spec_path)
        assert spec["title"] == "t"
        assert spec["sections"][0]["heading"] == "S"


class TestHeight:
    def _spec(self, height: object | None = None) -> dict:
        fig: dict = {"caption": "c", "candidates": ["fig.*"]}
        if height is not None:
            fig["height"] = height
        return {"title": "t", "sections": [{"heading": "S", "items": [fig]}]}

    def test_default_iframe_height(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "fig.html")
        html = _build(self._spec(), root, tmp_path)
        assert "<iframe" in html
        assert "height:70vh" in html

    def test_per_figure_height_string(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "fig.html")
        html = _build(self._spec("45vh"), root, tmp_path)
        assert "height:45vh" in html
        assert "height:70vh" not in html

    def test_per_figure_height_int_is_px(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "fig.pdf")
        html = _build(self._spec(360), root, tmp_path)
        assert "height:360px" in html

    def test_image_ignores_height(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "fig.png")
        html = _build(self._spec("45vh"), root, tmp_path)
        # images fit their width; a height is never written onto an <img>
        assert "<img" in html
        assert "height:45vh" not in html


class TestMarkdownIntro:
    def _spec(self, intro: object) -> dict:
        return {
            "title": "t",
            "intro": intro,
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "c", "candidates": ["**/x.png"]}],
                }
            ],
        }

    def test_value_bold_is_unwrapped_inline(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        html = _build(self._spec({"Note": "see **this**"}), root, tmp_path)
        # rendered as Markdown, with the lone wrapping <p> stripped inside the <dd>
        assert "<dd>see <strong>this</strong></dd>" in html

    def test_value_table_rendered(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        table = "| a | b |\n|---|---|\n| 1 | 2 |"
        html = _build(self._spec({"Stats": table}), root, tmp_path)
        assert "<table>" in html
        assert "<th>a</th>" in html

    def test_string_intro_is_markdown_block(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        html = _build(self._spec("A *para* and a [link](http://x)."), root, tmp_path)
        assert 'class="intro-md"' in html
        assert "<em>para</em>" in html
        assert '<a href="http://x">link</a>' in html

    def test_intro_keys_still_escaped(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        root.mkdir()
        html = _build(self._spec({"K<>": "v"}), root, tmp_path)
        assert "K&lt;&gt;" in html


def _fig_inventory(out: Path) -> tuple[list[str], int]:
    """Sorted relative paths + total byte size of the copied figures/ tree."""
    fig_dir = out / "figures"
    files = (
        sorted(p for p in fig_dir.rglob("*") if p.is_file()) if fig_dir.exists() else []
    )
    rels = [p.relative_to(out).as_posix() for p in files]
    total = sum(p.stat().st_size for p in files)
    return rels, total


class TestIdempotentRebuild:
    def test_double_build_is_stable(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "a" / "plot.html", b"<html>A</html>")
        _touch(root / "b" / "plot.png", b"imgbytes")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [
                        {"caption": "html one", "candidates": ["**/a/plot.html"]},
                        {"caption": "png two", "candidates": ["**/b/plot.png"]},
                    ],
                }
            ],
        }
        out = tmp_path / "viewer"
        build_viewer(spec, root, out)
        rels1, total1 = _fig_inventory(out)
        build_viewer(spec, root, out)
        rels2, total2 = _fig_inventory(out)
        assert rels1 == rels2
        assert total1 == total2
        assert len(rels1) == 2

    def test_out_inside_root_does_not_accumulate(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "plot.png", b"imgbytes")
        out = root / "viewer"  # output dir lives *inside* the resolution root
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "c", "candidates": ["**/*.png"]}],
                }
            ],
        }
        build_viewer(spec, root, out)
        n1 = len([p for p in (out / "figures").rglob("*") if p.is_file()])
        build_viewer(spec, root, out)
        n2 = len([p for p in (out / "figures").rglob("*") if p.is_file()])
        assert n1 == n2 == 1  # the copied asset never re-matches the glob


class TestOutDirExclusion:
    def test_resolution_skips_out_dir_files(self, tmp_path: Path) -> None:
        root = tmp_path / "art"
        _touch(root / "zzz" / "plot.png", b"REALDATA")
        out = root / "viewer"
        # a leftover png directly under out (it survives the figures/ clear) whose
        # path sorts *before* the real source -> it would win without the exclusion
        _touch(out / "leftover.png", b"STALEDATA")
        spec = {
            "title": "t",
            "sections": [
                {
                    "heading": "S",
                    "items": [{"caption": "c", "candidates": ["**/*.png"]}],
                }
            ],
        }
        build_viewer(spec, root, out)
        copied = [p for p in (out / "figures").rglob("*") if p.is_file()]
        assert len(copied) == 1
        assert copied[0].read_bytes() == b"REALDATA"
