"""Fixtures for the step-script tests.

The resolution environment is the same one the protocol-layer tests use — a
script step's *loader* still needs it, even though most of these tests call
``main`` directly and need nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.protocol._env import FIXTURES, build_env, write_rot_fixture


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("steps-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    return root


@pytest.fixture(scope="session")
def env(artifacts_root: Path):
    return build_env(artifacts_root)
