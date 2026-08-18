"""End-to-end corpus execution on tiny-random (CPU) through the real CLI.

The golden corpus authors Llama-3.1-8B; ``--set`` overrides (spec §9)
retarget the model key and layer/head indices at tiny scale — the
documents' semantics are untouched, which is exactly what the override
mechanism is for. The sweep/train corpus files (04, 05, 07, 08) stay off
the CPU budget per the phase plan; the fit→apply roundtrip below covers
the train path end to end at tiny scale instead, including the
ArtifactIdentity stamp-and-check cycle that corpus 09 specifies.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from safetensors.torch import load_file

from causalab.protocol.cli import main

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.neural.pytorch_hooks._drive import base_data_section  # noqa: F401  (tier anchor)
from tests.neural.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    from tests.protocol._env import write_rot_fixture

    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    return FIXTURES / "data", artifacts


def _run(name: str, roots: tuple[Path, Path], out: Path, *overrides: str) -> int:
    data_root, artifacts_root = roots
    argv = [
        "run",
        str(CORPUS_DIR / name),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--set",
        f"model.key={TINY_LLAMA}",
    ]
    for item in overrides:
        argv += ["--set", item]
    return main(argv)


def test_01_harvest_runs(roots, tmp_path):
    code = _run(
        "01_harvest_im.json",
        roots,
        tmp_path,
        "sites.L8.layer=0",
        "sites.L24.layer=1",
    )
    assert code == 0
    acts = load_file(str(tmp_path / "acts_L8_ans.safetensors"))
    assert acts["acts_L8_ans"].shape == (4, 1, 16)
    ragged = load_file(str(tmp_path / "acts_L8_sub.safetensors"))
    assert "acts_L8_sub.widths" in ragged  # multi-token subjects are ragged


def test_02_interchange_runs_and_scores(roots, tmp_path):
    code = _run("02_interchange_im.json", roots, tmp_path, "sites.target.layer=1")
    assert code == 0
    iia = pd.read_parquet(tmp_path / "iia.parquet")
    assert len(iia) == 4  # one row per example
    assert set(iia["value"]).issubset({0.0, 1.0})  # match is an indicator
    ld = pd.read_parquet(tmp_path / "logit_diff.parquet")
    assert len(ld) == 4 and ld["value"].dtype.kind == "f"


def test_03_path_patching_runs(roots, tmp_path):
    code = _run(
        "03_path_patching_im.json",
        roots,
        tmp_path,
        "sites.sender.layer=0",
        "sites.sender.head=1",
        "sites.receiver.layer=1",
        "sites.a10.layer=0",
        "sites.a11.layer=1",
    )
    assert code == 0
    ld = pd.read_parquet(tmp_path / "logit_diff.parquet")
    assert len(ld) == 3


def test_06_hydra_effect_runs(roots, tmp_path):
    code = _run(
        "06_hydra_effect_im.json",
        roots,
        tmp_path,
        "sites.abl.layer=0",
        "sites.probe14.layer=0",
        "sites.probe20.layer=1",
        "sites.resid_final.layer=1",
    )
    assert code == 0
    for rel in (
        "te_clean.parquet",
        "te_abl.parquet",
        "de14_clean.parquet",
        "de20_abl.parquet",
    ):
        assert (tmp_path / rel).is_file()


def test_fit_then_apply_roundtrip(roots, tmp_path):
    """Corpus 04 → 09 at tiny scale: train a DAS rotation, save the stamped
    bundle, then load it through a 09-shaped document — the
    ArtifactIdentity written at save must satisfy the check at load, and a
    doctored k must refuse."""
    data_root, artifacts_root = roots
    fit_out = tmp_path / "fit"
    code = _run(
        "04_das_im.json",
        (data_root, artifacts_root),
        fit_out,
        "sites.target.layer=1",
        "featurizers.rot.k=4",
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0
    bundle_path = fit_out / "rot.safetensors"
    assert bundle_path.is_file()
    fitted = load_file(str(bundle_path))
    assert fitted["weight"].shape == (16, 4)

    # stage the fitted bundle as 09's artifact and apply it
    target = artifacts_root / "artifacts/tiny/rot_k4.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_path, target)
    apply_out = tmp_path / "apply"
    code = _run(
        "09_das_apply_im.json",
        (data_root, artifacts_root),
        apply_out,
        "sites.target.layer=1",
        "featurizers.rot.k=4",
        "featurizers.rot.file_path=artifacts/tiny/rot_k4.safetensors",
    )
    assert code == 0
    iia = pd.read_parquet(apply_out / "iia.parquet")
    assert len(iia) == 2  # the weekdays/test split

    # a doctored declaration must refuse against the stamp (§2.5)
    code = _run(
        "09_das_apply_im.json",
        (data_root, artifacts_root),
        tmp_path / "mismatch",
        "sites.target.layer=1",
        "featurizers.rot.k=8",
        "featurizers.rot.file_path=artifacts/tiny/rot_k4.safetensors",
    )
    assert code == 1


def test_explain_and_digest_work_on_every_corpus_file(roots, capsys):
    data_root, artifacts_root = roots
    for path in sorted(CORPUS_DIR.glob("*_im.json")):
        for verb in ("digest", "explain", "validate"):
            code = main(
                [
                    verb,
                    str(path),
                    "--data-root",
                    str(data_root),
                    "--artifacts-root",
                    str(artifacts_root),
                ]
            )
            assert code == 0, f"{verb} failed on {path.name}"
    capsys.readouterr()


def test_run_output_is_stamped(roots, tmp_path):
    assert (
        _run(
            "01_harvest_im.json",
            roots,
            tmp_path,
            "sites.L8.layer=0",
            "sites.L24.layer=1",
        )
        == 0
    )
    from causalab.protocol.resolve import read_safetensors_metadata

    meta = read_safetensors_metadata(tmp_path / "acts_L24_ans.safetensors")
    assert meta is not None
    assert meta["model_key"] == TINY_LLAMA
    assert meta["backend"] == "pytorch_hooks"
    assert len(meta["produced_by"]) == 64
