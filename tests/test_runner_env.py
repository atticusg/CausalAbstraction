"""Tests for ``causalab.runner.env.load_project_dotenv`` (issue #221).

The runner loads a project ``.env`` at entry so credentials such as
``OPENROUTER_API_KEY`` reach the process with no manual ``export``. These
tests pin the three behaviours that matter: it loads from a discoverable
``.env``, it never clobbers an already-set variable, and it degrades to a
no-op (returning ``None``) when no ``.env`` exists.
"""

from __future__ import annotations

import os

import pytest

from causalab.runner import env as env_module
from causalab.runner.env import load_project_dotenv

pytestmark = pytest.mark.unit


class TestLoadProjectDotenv:
    def test_loads_key_from_cwd_dotenv(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Canonical POSIX-sourceable form (no spaces around `=`).
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-file\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        path = load_project_dotenv()

        assert path is not None
        assert os.path.basename(path) == ".env"
        assert os.environ["OPENROUTER_API_KEY"] == "sk-from-file"

    def test_does_not_override_existing_env_var(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An explicit export / sbatch --export must win over the file.
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-file\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-already-set")

        load_project_dotenv()

        assert os.environ["OPENROUTER_API_KEY"] == "sk-already-set"

    def test_returns_none_when_no_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When nothing is discoverable, it's a graceful no-op (no raise) so a
        # key supplied purely via the ambient environment / CI still works.
        monkeypatch.setattr(env_module, "find_dotenv", lambda *a, **k: "")

        assert load_project_dotenv() is None
