"""Paper-golden tier: real-model runs asserted against paper-provenance values.

Every asserted number in tests/golden/paper_goldens.json traces to a
published paper figure/table or to the VeriFires task package encoding it
(environments/silico_research/tasks/) — never to a pinning run of this
stack. Documents live in tests/golden/protocols/ (identity pinned in
golden_digests.json); fixture datasets are seeded, committed JSON produced
by tests/golden/fixtures/generators/.

Run on a GPU box:

    uv run pytest tests/golden -m golden

Gated models (meta-llama/Llama-3.1-8B, google/gemma-2-2b-it) need a
licensed ``HF_TOKEN`` in the environment — without one the load 401s with
nothing naming the cause. CUDA is preferred; Apple MPS works for the
smaller documents (the 1,152-row hours document wants ~35GB free).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from causalab.protocol.cli import main

from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS, GOLDENS_FILE

pytestmark = pytest.mark.golden

GOLDENS = json.loads(GOLDENS_FILE.read_text())["goldens"]

# golden ids each test claims; test_structural cross-checks this registry
# against paper_goldens.json so no value can go silently unasserted
COVERED: dict[str, tuple[str, ...]] = {
    "test_ioi_clean_logit_diff": ("ioi.clean_logit_diff", "ioi.io_over_s_rate"),
    "test_hours_baseline_accuracy": ("arithmetic.hours_baseline_acc",),
    "test_rome_average_total_effect": ("rome.ate",),
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("golden tier needs an accelerator (cuda or mps)")


def assert_golden(golden_id: str, measured: float) -> None:
    """One assertion path for every golden: prints measured next to the
    band so a near-miss is diagnosable straight from the log."""
    entry = GOLDENS[golden_id]
    if entry["sidedness"] == "at_least":
        floor = entry["floor"]
        print(f"{golden_id}: measured {measured:.4f}, floor {floor}")
        assert measured >= floor, f"{golden_id}: {measured:.4f} < floor {floor}"
    else:
        lo, hi = entry["band"]
        print(f"{golden_id}: measured {measured:.4f}, band [{lo}, {hi}]")
        assert lo <= measured <= hi, (
            f"{golden_id}: {measured:.4f} outside [{lo}, {hi}] "
            f"(paper value {entry['value']})"
        )


def run_document(name: str, out: Path, *, dtype: str = "fp32", **extra: str) -> None:
    argv = [
        "run",
        str(GOLDEN_PROTOCOLS / name),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(out / "artifacts"),
        "--out",
        str(out),
        "--device",
        _device(),
        "--dtype",
        dtype,
    ]
    for key, value in extra.items():
        argv += ["--set", f"{key}={value}"]
    assert main(argv) == 0


def test_ioi_clean_logit_diff(tmp_path):
    run_document("ioi_clean_logit_diff_im.json", tmp_path)
    ld = pd.read_parquet(tmp_path / "logit_diff.parquet")["value"]
    assert len(ld) == 512
    assert_golden("ioi.clean_logit_diff", float(ld.mean()))
    assert_golden("ioi.io_over_s_rate", float((ld > 0).mean()))


def test_hours_baseline_accuracy(tmp_path):
    run_document("hours_baseline_im.json", tmp_path, dtype="bf16")
    acc = pd.read_parquet(tmp_path / "acc.parquet")["value"]
    assert len(acc) == 1152
    assert_golden("arithmetic.hours_baseline_acc", float(acc.mean()))


def test_rome_average_total_effect(tmp_path):
    """ATE = mean over facts and noise seeds of P_clean - P_corrupted,
    recovered from the cross_entropy parquets (p = exp(-ce)); the clean
    read is off the seed axis, so its per-seed rows are identical and the
    per-example mean collapses them.

    One document per subject-token width (the backend refuses ragged
    edits); pooling the per-fact effects across the width shards restores
    the loose-filter width distribution — a single-width sample biases the
    ATE (diagnosed +16pts on the width-4 bucket alone)."""
    import numpy as np

    effects = []
    clean_all, corr_all = [], []
    for width in (2, 3, 4, 5):
        out = tmp_path / f"w{width}"
        run_document(f"rome_ate_w{width}_im.json", out)
        ce_clean = pd.read_parquet(out / "ce_clean.parquet")
        ce_corr = pd.read_parquet(out / "ce_corr.parquet")
        p_clean = np.exp(-ce_clean.groupby("example")["value"].mean())
        p_corr = (
            ce_corr.assign(p=np.exp(-ce_corr["value"])).groupby("example")["p"].mean()
        )
        effects.append(p_clean - p_corr)
        clean_all.append(p_clean)
        corr_all.append(p_corr)
    pooled = pd.concat(effects)
    print(
        f"n={len(pooled)}; clean baseline {pd.concat(clean_all).mean():.4f}, "
        f"corrupted {pd.concat(corr_all).mean():.4f} (paper: 0.270 / 0.0847)"
    )
    assert_golden("rome.ate", float(pooled.mean() * 100))
