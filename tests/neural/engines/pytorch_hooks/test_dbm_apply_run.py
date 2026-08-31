"""DBM's held-out path: fit a gate, save it, apply it (spec §2.5 ``file_path``).

The gap this closes. ``FEATURIZER_SLOTS["gate"] = ("theta",)`` makes a trained
gate **saveable**, and §2.5 says ``file_path`` is legal on every kind — but
``_build_stage`` dispatched ``subspace``/``pca``/``standardize``/``sae`` and
then raised, so a saved gate could never be loaded back. DBM therefore had no
apply document and no held-out number at all, while DAS had both
(``weekdays_das_sweep`` → ``weekdays_das_apply``). Two independent A3B runs hit
it and reported a DBM fit's own ``iia.json`` — a *train* score — as a
localization result.

The load-bearing assertion is the last one: an apply against the fit's own
split must reproduce the fit's number **exactly**. Anything less than exact
means the reloaded mask is not the mask that was scored — a soft σ(θ/T)
instead of the hard ``θ > 0`` split, a re-initialised θ, or a silently
truncated one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.shared.featurizers import Gate, build_stack
from causalab.neural.shared.services import TensorBundle
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import FeaturizerSpec

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
METHODS = str(REPO / "causalab/configs/protocols")

# tiny-random on CPU runs fp32 while dbm.json declares bf16 — and the
# realization is part of a fit bundle's identity (§8), so fit and apply have
# to name the same one
TINY = {"model.key": TINY_LLAMA, "model.dtype": "fp32"}

WIDTH = 8


# --------------------------------------------------------------------------- #
# the unit: what comes back off disk
# --------------------------------------------------------------------------- #


def _gate_loader(theta: torch.Tensor):
    def load_tensors(_path: str) -> TensorBundle:
        return TensorBundle(tensors={"theta": theta}, entry_coords={})

    return load_tensors


def _loaded_gate(theta: torch.Tensor, *, width: int = WIDTH) -> Gate:
    stack = build_stack(
        "gate",
        {"gate": FeaturizerSpec(kind="gate", file_path="gate.safetensors")},
        width=width,
        load_tensors=_gate_loader(theta),
        stage_cache={},
    )
    (stage,) = stack.stages
    return stage


def test_a_loaded_gate_selects_exactly_the_fitted_coordinates() -> None:
    """``build_stack`` puts every stage in eval mode, so a loaded gate is the
    hard ``θ > 0`` split — the one a fit's reported score was computed
    through, not the soft mask the train loop optimizes."""
    theta = torch.tensor([1.0, -1.0, 0.5, -0.25, 3.0, -2.0, 0.0, 0.125])
    gate = _loaded_gate(theta)
    assert isinstance(gate, Gate)
    x = torch.ones(WIDTH)
    kept, _ = gate.featurize(x)
    assert torch.equal(kept.nonzero().flatten(), (theta > 0).nonzero().flatten())
    # θ == 0 is *not* selected: the split is strict, as Gate._mask writes it
    assert kept[6] == 0.0


def test_a_loaded_gate_is_the_same_object_a_trained_one_is_after_eval() -> None:
    """The apply path must not re-derive anything. A gate trained in-process
    and put in eval mode, and the same θ round-tripped through a file, have to
    mask identically — otherwise an apply number is not comparable with the
    fit it came from."""
    theta = torch.randn(WIDTH, generator=torch.Generator().manual_seed(0))
    trained = Gate(WIDTH)
    with torch.no_grad():
        trained.theta.copy_(theta)
    trained.eval()
    x = torch.randn(4, WIDTH, generator=torch.Generator().manual_seed(1))
    assert torch.equal(trained.featurize(x)[0], _loaded_gate(theta).featurize(x)[0])
    assert torch.equal(trained.featurize(x)[1], _loaded_gate(theta).featurize(x)[1])


def test_a_loaded_gate_is_not_trainable() -> None:
    """Applying a mask is not resuming a fit — §2.5 forbids a ``file_path``
    featurizer in ``train.params``, and the stage says so itself."""
    assert not _loaded_gate(torch.ones(WIDTH)).theta.requires_grad


def test_a_gate_fitted_at_another_width_refuses() -> None:
    """A mask is a set of coordinates of one activation. Loading a 2048-wide
    gate at a 4096-wide site used to reach the swap and die in the featurize
    matmul; declared width against real tensor is the check that catches it
    (§2.5, widths derive from the site)."""
    with pytest.raises(ProtocolError, match="wide but the site here"):
        _loaded_gate(torch.ones(WIDTH + 1))


# --------------------------------------------------------------------------- #
# end to end: dbm.json fits, dbm_apply.json applies
# --------------------------------------------------------------------------- #


def _pipeline(apply_set: dict | None = None) -> dict:
    return {
        "version": "1",
        "description": "fit a DBM gate, then apply it without re-fitting",
        "output_dir": "dbm",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/dbm.json",
                "set": {
                    **TINY,
                    "sites.target.layer": 0,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "apply": {
                "type": "intervention_protocol",
                "document": f"{METHODS}/dbm_apply.json",
                "set": {
                    **TINY,
                    "sites.target.layer": 0,
                    # score the split the fit reported on, so the two numbers
                    # are the same question asked twice
                    "data.base.dataset": "weekdays/train",
                    "data.counterfactual.dataset": "weekdays/train",
                    **(apply_set or {}),
                },
            },
        },
    }


def _run_workflow(base: Path, document: dict) -> tuple[int, Path]:
    artifacts = base / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    wf_dir = base / "workflows"
    wf_dir.mkdir()
    path = wf_dir / "wf.json"
    path.write_text(json.dumps(document, indent=2))
    out = base / "run"
    code = main(
        [
            "run",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    return code, out / document["output_dir"]


@pytest.fixture(scope="module")
def dbm_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    code, run = _run_workflow(tmp_path_factory.mktemp("dbm"), _pipeline())
    assert code == 0
    return run


def test_the_fit_saves_a_gate_the_apply_can_read(dbm_run: Path) -> None:
    theta = load_file(str(dbm_run / "fit/gate.safetensors"))["theta"]
    assert theta.ndim == 1
    manifest = json.loads((dbm_run / "workflow.json").read_text())
    assert manifest["steps"]["apply"]["status"] == "completed"


def test_the_applied_mask_reproduces_the_fit_exactly(dbm_run: Path) -> None:
    """The whole point of an apply document: the same mask, scored again.

    On the fit's own split the two must agree to the bit. When the apply
    document then names a *held-out* split, any difference is the
    generalization the study is after — and not, as before, the difference
    between a train score and nothing."""
    fitted = table_frame(dbm_run / "fit/iia.json")
    applied = table_frame(dbm_run / "apply/iia.json")
    assert len(applied) == len(fitted) == 4  # the weekdays/train fixture rows
    assert list(applied["value"]) == pytest.approx(list(fitted["value"]), abs=0.0)


def test_a_gate_fitted_at_another_site_refuses(tmp_path: Path, capsys) -> None:
    """The ArtifactIdentity check (§2.5), same as the ``subspace`` case: the
    stamped site is part of what the fit is, so applying an L0 mask at L1 is
    refused rather than scored — and refused for that reason, not because
    something downstream tripped over a shape."""
    code, _ = _run_workflow(tmp_path, _pipeline({"sites.target.layer": 1}))
    assert code == 1
    err = capsys.readouterr().err
    assert "[V15]" in err and "ArtifactIdentity mismatch on 'site'" in err
