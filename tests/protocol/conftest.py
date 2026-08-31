"""Fixtures for the protocol-layer tests: the resolution environment and
the golden corpus, resolved against the committed fixture tables."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.protocol._env import CORPUS_DIR, FIXTURES, build_env, write_rot_fixture

CORPUS_FILES = sorted(p.name for p in CORPUS_DIR.glob("*_im.json"))


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("protocol-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    return root


@pytest.fixture(scope="session")
def env(artifacts_root: Path):
    return build_env(artifacts_root)
