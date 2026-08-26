"""End-to-end corpus execution on tiny-random (CPU) through the real CLI.

The golden corpus authors Llama-3.1-8B; ``--set`` overrides (spec §9)
retarget the model key and layer/head indices at tiny scale — the
documents' semantics are untouched, which is exactly what the override
mechanism is for. The sweep/train corpus files (04, 05, 07) stay off the
CPU budget per the phase plan; the fit→apply roundtrip below covers the
train path end to end at tiny scale instead, including the
ArtifactIdentity stamp-and-check cycle that corpus 09 specifies.

Corpus 08 is the one exception, run at a single ``k`` and one epoch (~10 s):
it is the only document in the corpus that sweeps ``train.seed``, and a
seed sweep is the one thing a single-point fit cannot check — the stage
cache is keyed by featurizer name alone, so a cache shared across points
would hand seed 0's rotation to every other seed and no unit test on
``build_stack`` would notice. See ``test_featurizer_seed.py``.
"""

from __future__ import annotations

import json
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


def test_08_seed_sweep_fits_three_genuinely_different_rotations(roots, tmp_path):
    """A ``{"sweep": [0,1,2]}`` on ``train.seed`` must fit three rotations
    from three *different* starting points (§2.11).

    The bug this guards end to end: ``train.seed`` never reached the
    ``subspace`` init, so all three points started from the identical
    rotation and drifted apart only by batch order. The three fits still
    differed — by ~1e-3, a subspace distance of ~0.015 — so nothing looked
    obviously wrong, but a sweep read as evidence of stability was really
    measuring data-order jitter around one initialisation.

    Asserted on the *column space*, ``‖QaQaᵀ − QbQbᵀ‖_F``, which ignores the
    arbitrary basis inside each frame: 0 is the same subspace, ``√(2k)`` ≈
    2.83 orthogonal ones. Pre-fix this measured ≤ 0.016 for every pair;
    the threshold below is an order of magnitude above that and an order of
    magnitude below what independent inits give (~2.4).
    """
    import torch

    out = tmp_path / "sweep"
    code = _run(
        "08_weekdays_das_sweep_im.json",
        roots,
        out,
        "sites.target.layer=1",
        "featurizers.rot.k=4",  # one k, so the sweep is over seed alone
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0
    fitted = load_file(str(out / "rot.safetensors"))
    assert sorted(fitted) == [f"weight[seed={s}]" for s in (0, 1, 2)]

    names = sorted(fitted)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            qa, qb = fitted[a], fitted[b]
            distance = float((qa @ qa.T - qb @ qb.T).norm())
            assert distance > 0.5, f"{a} and {b} fit the same subspace ({distance:.4f})"
            # each is still a frame, not just noise
            torch.testing.assert_close(
                qa.T @ qa, torch.eye(qa.shape[1]), atol=1e-5, rtol=1e-4
            )


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


def test_11_probe_generate_runs_and_scores(roots, tmp_path):
    """The exploration probe end to end: decode under a steer, score the last
    generated token with an ordinary metric."""
    code = _run("11_probe_generate_im.json", roots, tmp_path, "sites.target.layer=1")
    assert code == 0
    probe = pd.read_parquet(tmp_path / "probe.parquet")
    assert len(probe) == 4  # one row per example
    for value in probe["value"]:
        top = json.loads(value)
        assert len(top["tokens"]) == 1 and len(top["probs"]) == 1


def test_12_probe_variable_scores_every_step_and_reports_what_was_said(roots, tmp_path):
    """PR-2's surface end to end: a metric per decode step, an ids-domain
    metric that never touches the vocabulary, and a `variable` anchor whose
    misses come back as data.

    Tiny-random says nothing resembling a weekday, so `said_answer` matches
    nowhere — which is the case worth pinning: the run finishes, the rows
    survive, and `matched` says why the values are null.
    """
    code = _run("12_probe_variable_im.json", roots, tmp_path, "sites.target.layer=1")
    assert code == 0

    per_step = pd.read_parquet(tmp_path / "per_step.parquet")
    examples = per_step["example"].nunique()
    assert len(per_step) == examples * 8  # one row per (example, decode step)
    assert sorted(per_step["step"].unique()) == list(range(8))
    assert per_step["matched"].all()

    said = pd.read_parquet(tmp_path / "said.parquet")
    assert len(said) == examples  # decode reduces the window to one string
    assert said["step"].isna().all()  # no single step owns a joined string
    assert said["value"].notna().all()

    where = load_file(tmp_path / "where.safetensors")
    # every row missed, so every row addressed zero positions: the harvest is
    # empty but still shaped per example, and nothing about the run failed
    assert where["where"].shape[:2] == (examples, 0)
