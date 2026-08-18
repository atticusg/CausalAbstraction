"""The CLI verbs (spec §9) — validate / explain / digest, plus --set."""

from __future__ import annotations

import json

import pytest

from causalab.protocol.cli import main
from causalab.protocol.loader import check_data_columns, load

from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.unit


def _argv(verb: str, name: str, artifacts_root, *extra: str) -> list[str]:
    return [
        verb,
        str(CORPUS_DIR / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(artifacts_root),
        *extra,
    ]


def test_validate_ok(capsys, artifacts_root):
    assert main(_argv("validate", "02_interchange_im.json", artifacts_root)) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_data_checks_columns(capsys, artifacts_root):
    code = main(_argv("validate", "02_interchange_im.json", artifacts_root, "--data"))
    assert code == 0


def test_validate_data_catches_missing_column(env):
    loaded = load(CORPUS_DIR / "02_interchange_im.json", env)
    raw = json.loads(json.dumps(dict(loaded.raw)))
    raw["metrics"]["logit_diff"]["a"] = "not_a_column"
    reloaded = load(raw, env)
    with pytest.raises(Exception) as err:
        check_data_columns(reloaded, env)
    assert "not_a_column" in str(err.value)


def test_digest_prints_the_document_digest(capsys, env, artifacts_root):
    assert main(_argv("digest", "04_das_im.json", artifacts_root)) == 0
    printed = capsys.readouterr().out.strip()
    assert printed == load(CORPUS_DIR / "04_das_im.json", env).document_digest


def test_explain_reports_plan(capsys, artifacts_root):
    assert main(_argv("explain", "03_path_patching_im.json", artifacts_root)) == 0
    out = capsys.readouterr().out
    assert "forwards  4 per point" in out
    assert "paired_forward" in out


def test_explain_sweep_reports_point_count(capsys, artifacts_root):
    assert (
        main(_argv("explain", "07_weekdays_locate_scan_im.json", artifacts_root)) == 0
    )
    out = capsys.readouterr().out
    assert "points    64" in out


def test_set_override_changes_digest(capsys, env, artifacts_root):
    assert (
        main(
            _argv(
                "digest",
                "02_interchange_im.json",
                artifacts_root,
                "--set",
                "sites.target.layer=5",
            )
        )
        == 0
    )
    overridden = capsys.readouterr().out.strip()
    assert (
        overridden != load(CORPUS_DIR / "02_interchange_im.json", env).document_digest
    )


def test_refusal_exits_nonzero(capsys, artifacts_root):
    code = main(
        _argv(
            "validate",
            "02_interchange_im.json",
            artifacts_root,
            "--set",
            "sites.target.layer=99",
        )
    )
    assert code == 1
    assert "refused" in capsys.readouterr().err
