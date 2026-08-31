"""The demos are checked, not trusted (``docs/demos.md`` §6).

A demo is prose wrapped around documents, and prose rots silently: a renamed
component, a moved script module or a retired metric kind leaves the markdown
reading exactly as before and the JSON beside it dead. So the mechanical half
of the format's checklist runs here —

* every document under ``demos/`` loads and validates against its own demo's
  data root, ``--data`` included, so a column a metric names has to exist;
* every workflow's steps resolve, which reaches into each inner document;
* every relative link in a demo's markdown points at a file that exists, and
  every committed figure is actually shown;
* every demo has the seven sections of ``docs/demos.md`` §2, in order, with a
  header table whose ``Reproduced`` field is one of the two legal forms;
* every digest a demo quotes is one its own documents and tables really have,
  which is what makes "pasted output, not typed by hand" a check.

What is deliberately *not* checked is the prose: whether a number has a floor,
whether a figure caption is honest, whether a verdict answers its question.
Those are review, and the checklist says so.

The tier is ``unit``: validation is pure — no weights, no network, no
accelerator (spec §9) — so this costs a second and runs in the CPU tier CI
selects.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
DEMOS = REPO / "demos"

#: §2 — the seven sections, in this order. The header table sits above the
#: first of them, so it is checked separately.
SECTIONS = (
    "TL;DR",
    "The protocol",
    "Run it",
    "Experimental design",
    "Results",
    "Limits",
    "Next",
)

#: §2 — ``Reproduced`` is a record or an admission, and nothing else.
REPRODUCED = re.compile(r"^\s*(✓|⚠)\s+\S")

#: A markdown link whose target is a path rather than a URL or an anchor.
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|#)([^)\s]+)\)")


def _demo_dirs() -> list[Path]:
    return sorted(p for p in DEMOS.iterdir() if p.is_dir())


def _markdown() -> list[Path]:
    """Every demo file: the markdown under a demo directory, minus the index."""
    return sorted(p for d in _demo_dirs() for p in d.glob("*.md"))


def _documents() -> list[Path]:
    return sorted(DEMOS.glob("*/protocols/*.json")) + sorted(
        DEMOS.glob("*/workflows/*.json")
    )


def _env(document: Path) -> ResolutionEnv:
    """A demo carries its own tables, so the data root is its own ``data/``.

    Artifacts resolve against the repo root: a demo document that loads a
    fitted featurizer names it by a repo-relative path, and one that does not
    never touches the store.
    """
    demo = document.parents[1]
    return ResolutionEnv(
        datasets=FileDatasets(root=demo / "data"),
        artifacts=FileArtifacts(root=REPO),
    )


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


class TestDocuments:
    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_validates(self, document: Path) -> None:
        """Load and validate, columns included.

        ``--data`` is the half that catches the drift a demo is most likely to
        acquire: a metric naming a column the table stopped emitting reads as
        valid structurally and produces nothing at run time.
        """
        from causalab.protocol.loader import check_data_columns, load

        raw = json.loads(document.read_text())
        env = _env(document)
        if "steps" in raw:
            from causalab.workflow.document import load_workflow

            loaded = load_workflow(document, env)
            for name in loaded.order:
                inner = loaded.inner.get(name)
                if inner is not None:
                    check_data_columns(inner, env)
        else:
            check_data_columns(load(document, env), env)

    @pytest.mark.parametrize("document", _documents(), ids=_ids(_documents()))
    def test_has_a_description(self, document: Path) -> None:
        """JSON has no comments, which is why ``description`` exists
        (spec §7). A demo document without one is a document whose reason to
        exist lives only in the markdown beside it."""
        raw = json.loads(document.read_text())
        assert raw.get("description"), f"{document.name} declares no description"


class TestFormat:
    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_sections_in_order(self, demo: Path) -> None:
        headings = re.findall(r"^## (.+)$", demo.read_text(), flags=re.MULTILINE)
        present = [h for h in headings if h in SECTIONS]
        assert present == list(SECTIONS), (
            f"{demo.name}: expected the seven sections of docs/demos.md §2 in "
            f"order, got {present}"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_header_table_is_complete(self, demo: Path) -> None:
        text = demo.read_text()
        for field in (
            "Question",
            "Method",
            "Model",
            "Data",
            "Documents",
            "Cost",
            "Reproduced",
        ):
            assert f"**{field}**" in text, f"{demo.name}: header table has no {field}"

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_reproduced_is_a_legal_value(self, demo: Path) -> None:
        """✓ with a provenance, or ⚠ with what is stale. A demo whose figures
        predate its documents is useful; one that hides it is not."""
        row = re.search(r"\*\*Reproduced\*\*\s*\|(.+)", demo.read_text())
        assert row is not None, f"{demo.name}: no Reproduced row"
        value = row.group(1).strip().rstrip("|").strip()
        assert REPRODUCED.match(value), (
            f"{demo.name}: Reproduced is '{value}' — docs/demos.md §2 allows "
            "'✓ <provenance>' or '⚠ <what is stale>'"
        )

    @pytest.mark.parametrize("demo", _markdown(), ids=_ids(_markdown()))
    def test_links_resolve(self, demo: Path) -> None:
        missing = [
            target
            for target in LINK.findall(demo.read_text())
            if not (demo.parent / target.split("#", 1)[0]).exists()
        ]
        assert not missing, f"{demo.name}: dead links {missing}"


class TestPastedOutput:
    """§6 — "every ``explain`` block is pasted output, not typed by hand".

    A digest is the one thing in a demo that cannot be *nearly* right: it is a
    function of the document's canonical bytes, so a stale one is proof that
    the prose and the JSON have diverged. Editing a document's ``description``
    is enough to move it, which is exactly the edit a careful author makes
    without thinking to re-paste.
    """

    @pytest.mark.parametrize("demo_dir", _demo_dirs(), ids=_ids(_demo_dirs()))
    def test_quoted_digests_are_current(self, demo_dir: Path) -> None:
        from causalab.protocol.loader import load
        from causalab.workflow.document import load_workflow

        real: set[str] = set()
        for document in sorted(demo_dir.glob("*/*.json")):
            if document.parent.name not in ("protocols", "workflows"):
                continue
            env = _env(document)
            if "steps" in json.loads(document.read_text()):
                workflow = load_workflow(document, env)
                real.add(workflow.digest)
                # a step's digest is its document's *with `set` applied*, so it
                # differs from the same document loaded standalone
                real.update(workflow.inner_digests.values())
                real.update(workflow.step_digests.values())
            else:
                loaded = load(document, env)
                real.add(loaded.document_digest)
                real.update(loaded.point_digests)
        # a demo also quotes the content digest a table was built at — the
        # same sha256 over the file's bytes that a ref resolves to
        # (causalab.protocol.resolve.FileDatasets.digest)
        real.update(
            hashlib.sha256(table.read_bytes()).hexdigest()
            for table in demo_dir.glob("data/*/*.json")
            if not table.name.endswith(".manifest.json")
        )

        quoted = {
            hexits
            for demo in demo_dir.glob("*.md")
            for hexits in re.findall(r"digest\s+([0-9a-f]{8,64})", demo.read_text())
        }
        stale = sorted(q for q in quoted if not any(d.startswith(q) for d in real))
        assert not stale, (
            f"{demo_dir.name}: digests {stale} match no document in this demo — "
            "re-paste the explain output"
        )


class TestFigures:
    @pytest.mark.parametrize("demo_dir", _demo_dirs(), ids=_ids(_demo_dirs()))
    def test_every_figure_is_shown(self, demo_dir: Path) -> None:
        """A figure carries no record (workflow spec §2.5), so an unreferenced
        one is a binary nobody can date or explain. §5 asks every figure to
        carry a caption; the cheap half of that is checking it is shown at
        all."""
        shown = "\n".join(p.read_text() for p in demo_dir.glob("*.md"))
        orphans = [
            p.name
            for p in sorted((demo_dir / "figures").glob("*"))
            if p.name not in shown
        ]
        assert not orphans, f"{demo_dir.name}/figures: never shown {orphans}"


class TestIndex:
    def test_every_demo_is_indexed(self) -> None:
        """``demos/README.md`` is the only place a reader looks first, so a
        demo missing from it is a demo nobody finds."""
        index = (DEMOS / "README.md").read_text()
        missing = [
            str(p.relative_to(DEMOS))
            for p in _markdown()
            if str(p.relative_to(DEMOS)) not in index
        ]
        assert not missing, f"demos/README.md does not link {missing}"
