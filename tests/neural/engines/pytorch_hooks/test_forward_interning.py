"""Cross-point forward interning in the reference engine (spec §3, §4).

§3 promises that a swept document's shared sub-values are content-deduped —
"shared harvests and forwards fall out automatically" — and corpus 07's own
description spends the promise: a 32-layer x 2-position scan plans "64 patched
forwards plus one shared counterfactual-harvest forward". The planner has
always said so (``tests/protocol/test_corpus.py`` pins those digests), but a
flat per-point execution loop cannot claim it: it re-runs the shared harvest
once per point, because taps are deliberately absent from a group's digest and
each point taps a different layer.

Ground truth for "a forward happened" is a pre-hook on the loaded model
itself, not the engine's own tally, so a bookkeeping bug cannot make the
suite agree with itself; ``RunResult.forwards`` is then checked against it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pytest

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import campaign_plans
from causalab.protocol.engine import ExecutionRequest
from causalab.protocol.loader import LoadedProtocol, load
from causalab.protocol.plan import interned_groups, plan_point
from causalab.protocol.resolve import ResolutionEnv

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES, build_env, write_rot_fixture
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

#: Corpus 07 at tiny scale. The document authors a 32-layer x 2-position grid
#: on Llama-3.1-8B; tiny-random has 2 layers, and its rows say nothing a
#: ``{"variable": ...}`` anchor can find, so the two swept axes become 2 layers
#: x 2 indices. The shape that matters survives: several points whose patched
#: forward differs and whose counterfactual harvest does not.
OVERRIDES = {
    "model.key": TINY_LLAMA,
    "sites.target.layer": {"sweep": [0, 1]},
    "positions.tap": {"sweep": [{"index": -1}, {"index": -2}]},
}


@pytest.fixture(scope="module")
def scan_env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    root = tmp_path_factory.mktemp("interning-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    return build_env(root)


@pytest.fixture(scope="module")
def scan(scan_env: ResolutionEnv) -> LoadedProtocol:
    return load(
        CORPUS_DIR / "07_weekdays_locate_scan_im.json", scan_env, overrides=OVERRIDES
    )


def _request(
    loaded: LoadedProtocol,
    env: ResolutionEnv,
    out: Path,
    selected: Sequence[int] | None = None,
) -> ExecutionRequest:
    """One execution request over a chosen subset of the campaign's points.

    The same lockstep slice ``--points`` takes (``causalab.protocol.cli``), so
    a one-point request is exactly the shard an external scheduler hands a
    worker — and, usefully here, a run with nothing to intern against."""
    index = range(len(loaded.expansion.points)) if selected is None else selected
    return ExecutionRequest(
        points=tuple(loaded.expansion.points[i].raw for i in index),
        canonical=tuple(loaded.canonical_points[i] for i in index),
        digests=tuple(loaded.point_digests[i] for i in index),
        coords=tuple(loaded.expansion.points[i].coords for i in index),
        document_digest=loaded.document_digest,
        env=env,
        output_dir=out,
    )


@pytest.fixture()
def forwards() -> Iterator[list[int]]:
    """Every top-level call of the loaded model, counted at the model.

    ``load_model`` is memoized on its exact call form — ``quantization``
    included — so asking for it the way the engine's ``_executor`` does hands
    back the very object the engine will run; a call that differs in one
    keyword would silently hand back a second, unhooked model. No patching,
    and nothing private touched."""
    bundle = load_model(
        TINY_LLAMA, "main", dtype="fp32", device="cpu", quantization=None
    )
    calls: list[int] = []
    handle = bundle.model.register_forward_pre_hook(
        lambda _module, _args: calls.append(1)
    )
    try:
        yield calls
    finally:
        handle.remove()


def test_the_plan_shares_a_group_the_points_do_not(scan: LoadedProtocol) -> None:
    """The premise, stated as data: the campaign's forward-group instances
    outnumber its distinct digests, and the whole surplus is counterfactual
    harvest — one digest, one tap per layer."""
    plans = campaign_plans(scan.point_documents)
    groups = interned_groups(plans)
    assert sum(plan.num_forwards for plan in plans) == 8  # 4 points x 2 groups
    assert len(groups) == 5  # 4 distinct patched + 1 shared harvest
    (harvest,) = [group for group in groups if group.model == "original"]
    # one tap per layer: the position axis moves the gather, not the forward
    assert len(harvest.taps) == 2
    # the shared pass has to reach every layer any point taps, so the depth it
    # may elide at is the deepest of the union rather than of one point (§4)
    assert harvest.stop_after == max(tap.depth for tap in harvest.taps)


def test_a_shared_forward_group_runs_once(
    scan: LoadedProtocol, scan_env: ResolutionEnv, forwards: list[int], tmp_path: Path
) -> None:
    """The gap this closes: the engine runs the campaign's distinct forward
    groups, not one per point per group.

    Before interning this counted 8 — the flat per-point loop re-ran the
    shared counterfactual harvest for every one of the four points.
    """
    plans = campaign_plans(scan.point_documents)
    owed = len(interned_groups(plans))

    result = PytorchHooksEngine().execute(_request(scan, scan_env, tmp_path))

    assert len(forwards) == owed, (
        f"{len(forwards)} forwards for a campaign whose plan has {owed} "
        "distinct groups — the shared harvest ran more than once"
    )
    assert len(forwards) < sum(plan.num_forwards for plan in plans)
    # the engine's own tally must agree with what the model actually saw
    assert result.forwards == len(forwards)


def test_interning_changes_no_number(
    scan: LoadedProtocol, scan_env: ResolutionEnv, tmp_path: Path
) -> None:
    """Interning is a pure performance change: one campaign request must
    produce exactly the table that one request per point produces.

    A one-point request has nothing to intern against, so this is a real
    before/after — the sharded runs execute the harvest once per point, the
    whole run executes it once, and every value and every ``produced_by``
    stamp has to survive that unchanged.
    """
    whole = PytorchHooksEngine().execute(_request(scan, scan_env, tmp_path / "whole"))
    shards = [
        PytorchHooksEngine().execute(
            _request(scan, scan_env, tmp_path / f"point{i}", [i])
        )
        for i in range(len(scan.expansion.points))
    ]
    assert whole.forwards < sum(shard.forwards for shard in shards)

    for name in ("iia.json", "logit_diff.json"):
        interned = table_frame(tmp_path / "whole" / name)
        sharded = pd.concat(
            [table_frame(tmp_path / f"point{i}" / name) for i in range(len(shards))],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(interned, sharded)


def test_a_lone_point_still_runs_its_own_groups(
    scan: LoadedProtocol, scan_env: ResolutionEnv, forwards: list[int], tmp_path: Path
) -> None:
    """Nothing is shared *within* a point, so a single-point request still
    pays the plan's per-point ``num_forwards`` — interning must never drop a
    group a point genuinely needs."""
    result = PytorchHooksEngine().execute(_request(scan, scan_env, tmp_path, [0]))
    assert plan_point(scan.point_documents[0]).num_forwards == 2
    assert result.forwards == 2
    assert len(forwards) == 2
