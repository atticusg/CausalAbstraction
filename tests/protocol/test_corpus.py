"""The golden corpus (spec §7): every example loads, validates,
canonicalizes to a pinned digest, and derives the pinned execution shape.

The digests are pinned against the committed fixture tables in
``tests/protocol/fixtures`` (dataset content digests are part of the
canonical form, so the pins and the fixtures move together). Regenerate
with ``uv run python tests/protocol/update_corpus_digests.py`` and review
the diff — a digest change means the canonical form changed, which is a
loader migration event (spec §7), not a routine edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.backend import requires
from causalab.protocol.loader import load
from causalab.protocol.plan import closure_digest, plan_point

from tests.protocol._env import CORPUS_DIR

PINS_PATH = Path(__file__).parent / "corpus_digests.json"
PINS = json.loads(PINS_PATH.read_text())

#: (file, expected points, expected forwards per point, expected requires)
CORPUS_SHAPE = [
    ("01_harvest_im.json", 1, 1, set()),
    ("02_interchange_im.json", 1, 2, {"paired_forward"}),
    ("03_path_patching_im.json", 1, 4, {"paired_forward"}),
    ("04_das_im.json", 1, 2, {"grad", "paired_forward"}),
    ("05_dbm_im.json", 1, 2, {"grad", "paired_forward"}),
    ("06_hydra_effect_im.json", 1, 7, {"paired_forward"}),
    ("07_weekdays_locate_scan_im.json", 64, 2, {"paired_forward"}),
    ("08_weekdays_das_sweep_im.json", 9, 2, {"grad", "paired_forward"}),
    ("09_das_apply_im.json", 1, 2, {"paired_forward"}),
]


class TestCorpusUnit:
    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize("name,n_points,n_forwards,needed", CORPUS_SHAPE)
    def test_loads_and_derives_shape(self, env, name, n_points, n_forwards, needed):
        loaded = load(CORPUS_DIR / name, env)
        assert len(loaded.expansion.points) == n_points
        doc = loaded.point_documents[0]
        assert plan_point(doc).num_forwards == n_forwards
        assert set(requires(doc)) == needed

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_document_digest_pin(self, env, name):
        loaded = load(CORPUS_DIR / name, env)
        assert loaded.document_digest == PINS[name]["document"], (
            f"{name}: canonical form drifted — if intended, regenerate the pins "
            "(update_corpus_digests.py) and treat it as a loader migration (§7)"
        )

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_point_digest_pins(self, env, name):
        loaded = load(CORPUS_DIR / name, env)
        assert list(loaded.point_digests) == PINS[name]["points"]

    def test_sweep_points_are_distinct(self, env):
        loaded = load(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env)
        assert len(set(loaded.point_digests)) == 64

    def test_das_sweep_interns_one_harvest(self, env):
        """§3's forcing example: 9 fits (k × seed) share ONE counterfactual harvest —
        the original/counterfactual forward group has one content digest across all
        points, while the patched groups are 9 distinct fits."""
        loaded = load(CORPUS_DIR / "08_weekdays_das_sweep_im.json", env)
        harvest, patched, v_cf = set(), set(), set()
        identity = {"base": "d", "counterfactual": "d"}
        for pdoc in loaded.point_documents:
            for group in plan_point(pdoc, data_identity=identity).groups:
                (harvest if group.model == "original" else patched).add(group.digest)
            v_cf.add(closure_digest(pdoc, "v_cf"))
        assert len(harvest) == 1
        assert len(patched) == 9
        assert len(v_cf) == 9

    def test_locate_scan_shares_per_layer_harvests(self, env):
        """07: the 64 points span 32 layers × 2 positions; the counterfactual-side
        harvest group of a point depends on nothing swept (taps differ, the
        forward doesn't), so all 64 original/counterfactual groups intern to one."""
        loaded = load(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env)
        identity = {"base": "d", "counterfactual": "d"}
        harvest = {
            group.digest
            for pdoc in loaded.point_documents
            for group in plan_point(pdoc, data_identity=identity).groups
            if group.model == "original"
        }
        assert len(harvest) == 1


class TestCorpusCompleteness:
    pytestmark = pytest.mark.unit

    def test_every_corpus_file_is_covered(self):
        """A new *_im.json must enter CORPUS_SHAPE and the pins file — the
        corpus, the shape table, and the digests move together."""
        files = sorted(path.name for path in CORPUS_DIR.glob("*_im.json"))
        assert files == sorted(row[0] for row in CORPUS_SHAPE)
        assert files == sorted(PINS)


class TestCorpusProperty:
    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_load_is_deterministic(self, env, name):
        first = load(CORPUS_DIR / name, env)
        second = load(CORPUS_DIR / name, env)
        assert first.document_digest == second.document_digest
        assert first.point_digests == second.point_digests
        assert first.canonical_document == second.canonical_document
