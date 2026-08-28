"""Shared resolution-environment construction for the protocol tests.

One helper used by both the pytest fixtures (conftest.py) and the
digest-pin regeneration script (update_corpus_digests.py), so the pinned
digests and the asserting tests are guaranteed to resolve against identical
fixture content.

The single generated fixture is ``rot_k8.safetensors`` (corpus file 09's
``file_path`` featurizer): its bytes are deterministic — sorted-key header
JSON, zero-filled weight — so the content digest inside 09's canonical form
is stable across machines and sessions.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from causalab.protocol.resolve import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    build_artifact_identity,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_DIR = Path(__file__).parent.parent / "protocols"
ROT_FIXTURE_RELPATH = "artifacts/weekdays/llama31_8b/subspace/rot_k8.safetensors"


def write_rot_fixture(artifacts_root: Path) -> Path:
    """A deterministic fitted-DAS bundle matching 09_das_apply_im.json:
    stamped identity for (Llama-3.1-8B @ main in fp32, block_output L18, k=8,
    cayley, fp32 params), weight zeros. Load-time checks read only the
    header."""
    target = artifacts_root / ROT_FIXTURE_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = build_artifact_identity(
        produced_by="0" * 64,
        model_key="meta-llama/Llama-3.1-8B",
        model_revision="main",
        model_dtype="fp32",
        tokenizer="meta-llama/Llama-3.1-8B",
        site={"component": "block_output", "layer": 18},
        k=8,
        parametrization="cayley",
        dtype="fp32",
        trained_on="weekdays/train",
        trained_on_digest="0" * 64,
        engine="pytorch_hooks",
        commit="fixture",
    )
    # This fixture is 09's "previously fitted" artifact, and artifacts fitted
    # before the backend→engine rename carry the old stamp key. Keeping the old
    # key keeps 09's content_digest (hence its pinned canonical form)
    # byte-stable across the rename, and keeps the loader's tolerance of
    # pre-rename bundles under test. Flip to "engine" only with a corpus
    # re-pin.
    identity["backend"] = identity.pop("engine")
    n_bytes = 4096 * 8 * 4
    header = {
        "__metadata__": identity,
        "weight": {"dtype": "F32", "shape": [4096, 8], "data_offsets": [0, n_bytes]},
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with target.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(bytes(n_bytes))
    return target


def build_env(artifacts_root: Path) -> ResolutionEnv:
    """The test resolution environment: committed JSON fixture tables for
    datasets, ``artifacts_root`` (a copy of fixtures/artifacts plus the
    generated bundle) for artifacts."""
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts_root),
    )
